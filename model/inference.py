from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer
from f5_tts.model import CFM, DiT, UNetT
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder, save_spectrogram, load_model, remove_silence_for_generated_wav, infer_process
import f5_tts.infer.utils_infer as utils_infer
import torchaudio
import torch.nn.functional as F
from scipy import signal
import whisper
import pyworld as pw
from scipy.io.wavfile import write
from flask import Flask, request, jsonify
import librosa
import soundfile as sf
import torch
import traceback
import numpy as np
import time
import json
import tempfile
from cached_path import cached_path
import os
# For time stretching audio
import pyrubberband as pyrb
# Enable MPS fallback to CPU for operations not supported by MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False

# F5-TTS components

app = Flask(__name__)
# Get the absolute path to the `upload` folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "express-typescript-app", "../../uploads"))

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create model cache directory
os.makedirs("./model_cache", exist_ok=True)

# Global variables for models
whisper_model = None  # Whisper model
whisper_device = "cpu"  # Device for Whisper - always CPU to avoid sparse tensor issues

# F5-TTS model components
f5_model = None
f5_vocoder = None
f5_tokenizer = None

# Memory optimization settings
MAX_AUDIO_LENGTH = 10  # Maximum audio length in seconds to process at once
CHUNK_SIZE = 200  # Maximum number of words to process at once

DEFAULT_TTS_MODEL = "F5-TTS"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16,
                    ff_mult=2, text_dim=512, conv_layers=4)),
]


def load_f5tts(ckpt_path=str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.safetensors"))):
    F5TTS_model_cfg = dict(dim=1024, depth=22, heads=16,
                           ff_mult=2, text_dim=512, conv_layers=4)
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


try:
    whisper_device = "cpu"
    print(f"Using {whisper_device} specifically for Whisper model")
    # Using base model for faster loading, use "medium" or "large" for better accuracy
    whisper_model = whisper.load_model("turbo", device=whisper_device)
    print("✅ Whisper model loaded successfully")
    vocoder = load_vocoder()
    print("✅ Vocoder model loaded successfully")
    F5TTS_ema_model = load_f5tts()
    print("✅ F5_TTS model loaded successfully")
except Exception as e:
    print(f"❌ sumthin broke: {e}")


def infer(
        ref_audio,
        ref_text,
        gen_text,
        segment_duration,
        model,
        remove_silence,
        cross_fade_duration=0.15,
        nfe_step=32,
        speed=1,
):
    if not ref_audio:
        print("Please provide reference audio.")
        return

    if not gen_text.strip():
        print("Please enter text to generate.")
        return

    ema_model = F5TTS_ema_model
    # duration = librosa.get_duration(path=ref_audio)

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        cfg_strength=2.0,
        fix_duration=segment_duration,
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    # with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
    # 	spectrogram_path = tmp_spectrogram.name
    # 	save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave)


def show_info(msg):
    print(msg)


utils_infer.show_info = show_info


def rip(src_audio_path, tgt_audio_path, output_path):
    try:
        # Get source audio duration
        src_duration = 0
        try:
            src_duration = librosa.get_duration(path=src_audio_path)
            print(f"Source audio duration: {src_duration:.2f} seconds")
        except Exception as dur_error:
            print(f"Error getting source duration: {dur_error}")
            # Continue anyway, just won't do time stretching
        global whisper_device
        decode_options = {
            "fp16": False,
        }
        # Transcribe with timestamps enabled
        transcription_result = whisper_model.transcribe(
            src_audio_path,
            **decode_options
        )
        try:
            # Create a serializable version with only what you need
            serializable_data = {
                "text": transcription_result["text"],
                "language": transcription_result["language"],
                "segments": []
            }

            # Process segments to ensure all values are serializable
            for segment in transcription_result["segments"]:
                clean_segment = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }

                # Add words if they exist
                if "words" in segment:
                    clean_segment["words"] = [
                        {
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"]
                        }
                        for word in segment["words"]
                    ]

                serializable_data["segments"].append(clean_segment)

            # Write to file with proper JSON formatting
            # with open(transcription_path, 'w', encoding='utf-8') as f:
            #     f.write(json.dumps(serializable_data, indent=2))

            # print(f"Transcription saved to {transcription_path}")
        except Exception as e:
            print(f"Error saving transcription: {e}")
            # Continue with processing even if saving fails

        print(serializable_data)
        # Still get target audio transcription for reference
        tgt_transcription = whisper_model.transcribe(
            tgt_audio_path, **decode_options)

        text = transcription_result["text"]
        tgt_text = tgt_transcription["text"]
        language = transcription_result["language"]

        # final_sample_rate, final_wave = infer(
        #     tgt_audio_path, tgt_text, text, F5TTS_ema_model, remove_silence=False)
        # Use the timestamp-based synchronization
        sync_output_path = sync_with_timestamps(
            src_audio_path,
            tgt_audio_path,
            tgt_text,
            output_path,
            serializable_data  # Pass the full transcription with timestamps
        )

        # if isinstance(final_wave, np.ndarray):
        #     # If it's a numpy array, convert to tensor first
        #     final_wave = torch.from_numpy(final_wave)
        # # Add channel dimension if it's 1D
        # if final_wave.dim() == 1:
        #     # Add channel dimension [1, samples]
        #     final_wave = final_wave.unsqueeze(0)

        return {
            "success": True,
            "output_path": output_path,
            "text": text,
            "language": language,
            "transcript": serializable_data
        }
    except Exception as e:
        print(f"❌ Error in voice conversion: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def calculate_dynamic_speed(source_segment, reference_audio_path, reference_text):
    """
    Calculate the ideal speed parameter by comparing speaking rates.

    Returns: float - speed parameter for F5-TTS
    """
    # 1. Calculate source speaking rate
    source_text = source_segment["text"]
    source_duration = source_segment["end"] - source_segment["start"]
    source_word_count = len(source_text.split())
    source_speaking_rate = source_word_count / source_duration  # words per second

    # 2. Calculate reference speaking rate from a small sample
    if not hasattr(calculate_dynamic_speed, "ref_speaking_rate"):
        # Only calculate this once and cache it
        try:
            # Load a small portion of the reference audio
            ref_y, ref_sr = librosa.load(
                reference_audio_path, sr=None)
            ref_duration = len(ref_y) / ref_sr

            # Get word count from reference text
            ref_word_count = len(reference_text.split())

            # Calculate speaking rate
            ref_speaking_rate = ref_word_count / ref_duration

            # Cache for future calls
            calculate_dynamic_speed.ref_speaking_rate = ref_speaking_rate
            print(
                f"Reference speaking rate: {ref_speaking_rate:.2f} words/sec")
        except Exception as e:
            print(f"Error calculating reference rate: {e}")
            # Default fallback
            calculate_dynamic_speed.ref_speaking_rate = 2.5  # typical speaking rate

    # 3. Calculate ratio (how much faster/slower source is compared to reference)
    speed_ratio = source_speaking_rate / calculate_dynamic_speed.ref_speaking_rate
    print(f"Speed Ratio before squeeze: {speed_ratio:.2f}")
    # 4. Apply limits to avoid extreme values
    speed_parameter = min(max(speed_ratio, 0.8), 5)  # Keep between 0.8-1.8
    print(f"Source rate: {source_speaking_rate:.2f} words/sec, "
          f"Speed parameter: {speed_parameter:.2f}")

    return speed_parameter


def sync_with_timestamps(src_audio_path, tgt_audio_path, tgt_text, output_path, transcription):
    # Load source and reference info first
    src_y, src_sr = librosa.load(src_audio_path, sr=24000)
    src_duration = len(src_y) / src_sr

    # Process reference audio once
    ref_audio, ref_text = utils_infer.preprocess_ref_audio_text(
        tgt_audio_path, tgt_text, clip_short=False)

    # Check if ref_audio is a string (file path) and load it if needed
    if isinstance(ref_audio, str):
        # ref_audio is a file path, need to load it
        ref_audio, _ = librosa.load(ref_audio, sr=24000)

    # Calculate reference audio duration for passing to F5-TTS
    ref_audio_samples = len(ref_audio) if isinstance(
        ref_audio, np.ndarray) else ref_audio.shape[-1]
    ref_audio_duration = ref_audio_samples / 24000  # Standard F5-TTS sample rate
    print(f"Reference audio duration: {ref_audio_duration:.2f}s")

    # Get all segments
    segments = []
    for segment in transcription["segments"]:
        segments.append({
            "text": segment["text"],
            "start": segment["start"],
            "end": segment["end"],
            "duration": segment["end"] - segment["start"]
        })

    # Calculate total original segment duration
    total_original_segment_duration = sum(s["duration"] for s in segments)

    # Calculate dynamic expansion budget
    # This is the critical factor that prevents overall drift
    expansion_budget = src_duration / total_original_segment_duration
    # Cap at 1.5x to avoid extreme expansion
    expansion_budget = min(expansion_budget, 1.5)
    print(f"Global expansion budget: {expansion_budget:.2f}x")

    # Create output buffer matching source duration exactly
    final_audio = np.zeros(int(src_duration * src_sr))

    # Track the end of previous segment for overlap detection
    previous_segment_end = 0

    # Process each segment
    for i, segment in enumerate(segments):
        # Calculate timing parameters
        segment_duration = segment["duration"]
        word_count = len(segment["text"].split())
        word_density = word_count / segment_duration

        # Calculate speed parameter
        speed = calculate_dynamic_speed(segment, tgt_audio_path, ref_text)

        # Calculate quality-preserving multiplier based on speech characteristics
        if word_density > 5.0 or speed > 2.5:
            # Fast/dense speech needs more expansion
            quality_multiplier = 1.4
        elif word_density > 3.5 or speed > 1.8:
            # Moderately fast speech
            quality_multiplier = 1.2
        else:
            # Normal speech
            quality_multiplier = 1.1

        # Apply global budget constraint
        # This ensures we don't expand beyond what would fit in the source duration
        final_multiplier = min(quality_multiplier, expansion_budget)

        # Calculate adjusted segment duration
        adjusted_segment_duration = segment_duration * final_multiplier

        # CRITICAL: Include reference audio duration for F5-TTS
        # But we'll only use the actual segment portion in final output
        total_duration_for_f5tts = ref_audio_duration + adjusted_segment_duration

        print(
            f"Segment {i+1}/{len(segments)}: Original={segment_duration:.2f}s, Adjusted={adjusted_segment_duration:.2f}s (x{final_multiplier:.2f})")
        print(
            f"Total duration for F5-TTS: {total_duration_for_f5tts:.2f}s (including {ref_audio_duration:.2f}s reference audio)")

        # Generate the audio with F5-TTS
        segment_result = infer(
            tgt_audio_path,
            ref_text,
            segment["text"],
            total_duration_for_f5tts,  # Pass total duration INCLUDING reference audio
            F5TTS_ema_model,
            remove_silence=False,
            speed=speed
        )

        # F5-TTS already removes the reference audio portion, so we have just our segment
        segment_audio = segment_result[1]
        segment_sr = segment_result[0]

        # Calculate timestamp-based position
        start_idx = int(segment["start"] * src_sr)

        # Check for potential overlap with previous segment
        if start_idx < previous_segment_end:
            # Add a small gap and place after previous segment
            start_idx = previous_segment_end + int(0.05 * src_sr)
            print(f"Repositioned segment due to overlap: {start_idx}")

            # If we're running out of space, we need to fit the remaining segments
            remaining_segments = len(segments) - i
            if remaining_segments > 1:
                # Are we past 80% of the source duration?
                if start_idx > 0.8 * len(final_audio):
                    # Time-stretch this segment to be shorter
                    compression_factor = min(
                        0.9, (len(final_audio) - start_idx) / (1.2 * len(segment_audio)))
                    segment_audio = pyrb.time_stretch(
                        segment_audio, segment_sr, 1/compression_factor)
                    print(
                        f"Compressed segment by {compression_factor:.2f}x to fit within source duration")

        # Insert the segment at the calculated position
        available_length = min(
            len(segment_audio), len(final_audio) - start_idx)
        final_audio[start_idx:start_idx +
                    available_length] = segment_audio[:available_length]

        # Track where this segment ends
        previous_segment_end = start_idx + available_length

        # Stop if we've reached the end of the source duration
        if previous_segment_end >= len(final_audio) - int(0.1 * src_sr):
            print(
                f"Reached end of output buffer at segment {i+1}/{len(segments)}")
            break

    # Save final audio
    sf.write(output_path, final_audio, src_sr)
    print(f"✅ Synchronized audio saved to: {output_path}")
    return output_path


@app.route("/inference", methods=["POST"])
def inference():
    try:
        print("📩 Received a new inference request.")

        # Check if models are loaded
        if whisper_model is None:
            print("❌ Whisper model failed to load.")
            return jsonify({"error": "Speech recognition model failed to load"}), 500

        # if f5_model is None:
        #     print("❌ F5-TTS model failed to load.")
        #     return jsonify({"error": "Voice synthesis model failed to load"}), 500

        # Ensure we have the device variables available in this context
        global device, whisper_device

        # Get request data
        data = request.json
        src_path = data.get("src_path")
        tgt_path = data.get("tgt_path")
        src_filename = data.get("src_filename")
        tgt_filename = data.get("tgt_filename")

        # Get conversion settings
        output_format = data.get("output_format", "wav")  # Output audio format

        # Additional settings
        # Whether to match source duration
        keep_original_timing = data.get("keep_original_timing", False)

        # Validate inputs
        if not src_path:
            print("❌ No source audio provided in request.")
            return jsonify({"error": "No source audio provided"}), 400

        src_audio_path = os.path.join(UPLOAD_FOLDER, src_filename)
        print(f"🔍 Checking if source file exists: {src_audio_path}")

        if not src_audio_path or not os.path.exists(src_audio_path):
            print(f"❌ Source audio file not found: {src_audio_path}")
            return jsonify({"error": "Source audio file is missing or invalid"}), 400

        tgt_audio_path = os.path.join(UPLOAD_FOLDER, tgt_filename)
        print(f"🔍 Checking if target file exists: {tgt_audio_path}")

        if not tgt_audio_path or not os.path.exists(tgt_audio_path):
            print(f"❌ Target audio file not found: {tgt_audio_path}")
            return jsonify({"error": "Target audio file is missing or invalid"}), 400

        # Prepare output paths
        output_wav_path = os.path.join(
            UPLOAD_FOLDER, f"{src_filename}_converted.wav")

        transcription_path = os.path.join(
            UPLOAD_FOLDER, f"transcription-{time.time()}.json")

        # Final output path might be a different format
        final_output_path = output_wav_path
        if output_format != "wav":
            final_output_path = os.path.join(
                UPLOAD_FOLDER, f"{src_filename}_converted.{output_format}")

        print(f"🚀 Starting voice conversion process to: {output_wav_path}")

        # Perform voice conversion
        start_time = time.time()

        # Use F5-TTS for voice cloning via transcription
        print("Using F5-TTS for voice synthesis")
        # result = convert_voice(
        # src_audio_path, tgt_audio_path, output_wav_path, keep_original_timing)
        result = rip(src_audio_path, tgt_audio_path, output_wav_path)
        # Convert to requested format if needed
        if result["success"] and output_format != "wav":
            try:
                # print(f"Converting output to {output_format} format...")
                # import subprocess

                # Use FFmpeg to convert the format
                # command = [
                #     "ffmpeg", "-y", "-i", output_wav_path,
                #     "-c:a", "libmp3lame" if output_format == "mp3" else "copy",
                #     final_output_path
                # ]
                # subprocess.run(command, check=True)

                # Update the output path in the result
                result["output_path"] = final_output_path

                # Clean up the WAV file if conversion succeeded
                # if os.path.exists(final_output_path) and os.path.exists(output_wav_path):
                # 	os.remove(output_wav_path)

            except Exception as format_err:
                print(f"Error converting format: {format_err}")
                # Keep the original WAV file if conversion failed
                pass

        end_time = time.time()
        conversion_time = f"{end_time - start_time:.2f}"
        print(f"🎉 Voice conversion completed in {conversion_time} seconds.")

        if result["success"]:
            print("✅ Voice conversion successful!")

            # Basic response data
            response_data = {
                "output_path": result["output_path"],
                "time": conversion_time
            }

            # Add additional data from F5-TTS if available
            if "text" in result:
                response_data["text"] = result["text"]
            if "language" in result:
                response_data["language"] = result["language"]
            if "duration" in result:
                response_data["duration"] = result["duration"]
            # if "transcript" in result:
                # with open(transcription_path, 'w'):
                #     json.dumps(result["transcript"])
            return jsonify(response_data)
        else:
            print(f"❌ Error: {result['error']}")
            return jsonify({"error": result["error"]}), 500

    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True)
