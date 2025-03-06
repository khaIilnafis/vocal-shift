import json
import os
import time
import numpy as np
import traceback
import torch
import soundfile as sf
import librosa
from flask import Flask, request, jsonify
from scipy.io.wavfile import write

# Import FreeVC components
import sys
sys.path.append('./FreeVC')  # Add FreeVC directory to path
from FreeVC.wavlm import WavLM, WavLMConfig  # NOQA
from utils import get_hparams_from_file, load_checkpoint, get_vocoder, get_cmodel, get_content, get_vocoder, transform, load_checkpoint, load_wav_to_torch  # NOQA
from FreeVC.text import text_to_sequence  # NOQA
from FreeVC.models import SynthesizerTrn  # NOQA
from FreeVC.hifigan.models import Generator  # NOQA
from FreeVC.hifigan import AttrDict  # NOQA
from FreeVC.pitch import FreeVCPitchExtractor  # NOQA
from FreeVC.speaker_encoder.voice_encoder import SpeakerEncoder  # NOQA
from FreeVC.mel_processing import mel_spectrogram_torch  # NOQA
app = Flask(__name__)
# Get the absolute path to the `upload` folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "express-typescript-app", "../../uploads"))

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create model cache directory
os.makedirs("./model_cache", exist_ok=True)

# Global variables for models
net_g = None
content_model = None
vocoder = None
pitch_extractor = None
cmodel = None
smodel = None
hps = None
speaker_embeddings = {}

try:
    # Initialize FreeVC model
    print("Loading FreeVC model...")
    ptfile = "./FreeVC/pretrained_models/freevc.pth"
    # Load config (adjust paths as needed)
    hps = get_hparams_from_file('./FreeVC/configs/freevc.json')

    # Initialize the model
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).to('cpu')

    # Load the pretrained model
    _ = net_g.eval()
    _ = load_checkpoint(ptfile, net_g, None, True)

    print("✅ FreeVC model loaded successfully")
    # Load WavLM model for content extraction
    print("Loading WavLM model...")
    cmodel = get_cmodel(0)

    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("Using MPS (Apple Metal) acceleration")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("Using CUDA acceleration")
    # else:
    #     device = torch.device("cpu")
    #     print("Using CPU only")
    # device = torch.device("cpu")
    # content_model = get_content_model(device)
    print("✅ WavLM model loaded successfully")
    if hps.model.use_spk:
        print("Loading speaker encoder...")
        encoder_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "FreeVC", "speaker_encoder", "ckpt"))
        print(f"encoder dir: {encoder_dir}")
        encoder_model = os.path.join(
            encoder_dir, "pretrained_bak_5805000.pt")
        print(f"encoder model: {encoder_model}")
        smodel = SpeakerEncoder(encoder_model)
    # Initialize the pitch extractor
    # print("Loading pitch extractor...")
    # try:
    # 	pitch_extractor = FreeVCPitchExtractor(
    # 		hop_length=hps.data.hop_length).to(device)
    # 	print("✅ Pitch extractor loaded successfully")
    # except Exception as e:
    # 	print(f"⚠️ Could not load pitch extractor: {e}")
    # 	pitch_extractor = None

    # Load HiFiGAN vocoder for higher quality synthesis
    # print("Loading HiFiGAN vocoder...")
    # try:
    # 	# Try to load the vocoder
    # 	with open("./FreeVC/hifigan/config_v3.json", "r") as f:
    # 		config = json.load(f)
    # 	config = AttrDict(config)
    # 	vocoder = Generator(config)
    # 	ckpt = torch.load("./FreeVC/hifigan/generator_v3", map_location=device)
    # 	vocoder.load_state_dict(ckpt["generator"])
    # 	vocoder.eval()
    # 	vocoder.remove_weight_norm()
    # 	vocoder = vocoder.to(device)
    # 	print("✅ HiFiGAN v3 vocoder loaded successfully")
    # except Exception as e:
    # 	print(f"Warning: Could not load HiFiGAN v3, trying v1: {e}")
    # 	try:
    # 		# Fallback to v1 if available
    # 		with open("./FreeVC/hifigan/config.json", "r") as f:
    # 			config = json.load(f)
    # 		config = AttrDict(config)
    # 		vocoder = Generator(config)
    # 		ckpt = torch.load("./FreeVC/hifigan/generator_v1",
    # 						  map_location=device)
    # 		vocoder.load_state_dict(ckpt["generator"])
    # 		vocoder.eval()
    # 		vocoder.remove_weight_norm()
    # 		vocoder = vocoder.to(device)
    # 		print("✅ HiFiGAN v1 vocoder loaded successfully")
    # 	except Exception as e2:
    # 		print(f"Warning: Could not load HiFiGAN vocoder: {e2}")
    # 		vocoder = None
    # 		print("⚠️ Continuing without HiFiGAN vocoder - audio quality may be reduced")

    # Load speaker embeddings from a predefined set
    # For simplicity, we'll create a mapping of voice types
    speaker_embeddings = {
        "kid": torch.load('./FreeVC/speaker_embeddings/kid.pt'),
        # "kid2": torch.load('./FreeVC/speaker_embeddings/kid2.pt'),
        "obama": torch.load('./FreeVC/speaker_embeddings/obama.pt'),
        # "male": torch.load('./FreeVC/speaker_embeddings/male.pt')
    }

    # Print the dimensions of loaded embeddings
    for speaker_type, embedding in speaker_embeddings.items():
        print(f"Loaded {speaker_type} embedding with shape: {embedding.shape}")

    print("✅ Speaker embeddings loaded successfully")

    # Check if the model is speaker-aware
    if not hasattr(hps.model, 'use_spk') or not hps.model.use_spk:
        print("⚠️ Model doesn't appear to support speaker conditioning!")

    # Add this code to see if the embeddings are meaningfully different
    for name1, emb1 in speaker_embeddings.items():
        for name2, emb2 in speaker_embeddings.items():
            if name1 != name2:
                similarity = torch.nn.functional.cosine_similarity(
                    emb1.flatten(), emb2.flatten(), dim=0)
                print(
                    f"Similarity between {name1} and {name2}: {similarity.item()}")

except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    net_g = None
    content_model = None
    vocoder = None
    cmodel = None
    smodel = None


def get_speaker_embedding(speaker_type):
    """Get speaker embedding based on desired voice type."""
    # Return the pre-loaded speaker embedding
    if speaker_type in speaker_embeddings:
        return speaker_embeddings[speaker_type]
    else:
        print(f"Speaker type {speaker_type} not found, falling back to 'kid'")
        return speaker_embeddings["kid"]


def change_voice(audio_path, output_path, speaker_embedding, strength):
    try:
        print(f"🔹 Starting voice conversion for: {audio_path}")
        start_time = time.time()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"🚨 Audio file not found: {audio_path}")

        # Load input audio
        source_audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Convert to tensor and move to CUDA if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        source = torch.FloatTensor(source_audio).unsqueeze(
            0).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract content features using WavLM
            # c = extract_content(content_model, source)
            c = utils.get_cmodel(0)
            # Extract pitch information if available - for logging/analysis only
            # Note: FreeVC doesn't directly use these pitch values in inference,
            # but we extract them for debugging and potential future features
            if pitch_extractor is not None:
                print("Extracting pitch information for analysis...")
                try:
                    # Extract f0 (fundamental frequency) and uv (voiced/unvoiced)
                    f0, uv = pitch_extractor(source.squeeze(1))
                    print(f"Pitch extraction successful. Shape: {f0.shape}")
                    # We won't pass these to the model - pitch is encoded in content features
                except Exception as pitch_error:
                    print(f"Error in pitch extraction: {pitch_error}")
                    f0, uv = None, None
            else:
                f0, uv = None, None

            # Use the provided target speaker embedding and adapt it
            tgt_embedding = speaker_embedding.to(device)

            # Adapt the embedding to match the expected dimensions (256 channels)
            # tgt_embedding = adapt_speaker_embedding(
            #     tgt_embedding, target_dim=256)

            # Properly normalize the embedding to unit length (better than just multiplying)
            # This preserves the direction of the embedding vector while controlling magnitude
            # norm = torch.norm(tgt_embedding, p=2, dim=1, keepdim=True)
            # tgt_embedding = tgt_embedding / (norm + 1e-8) * strength

            tgt_embedding * strength
            print(f"Applied strength {strength} with proper normalization")
            print(f"Shape after processing: {tgt_embedding.shape}")

            # Get global hps variable
            global hps

            # Perform voice conversion using the infer method
            if hasattr(hps.model, 'use_spk') and hps.model.use_spk:
                # NOTE: FreeVC's SynthesizerTrn.infer() method doesn't accept f0 and uv parameters
                # The pitch information is already captured in the content features (c)
                print("Using standard inference without explicit pitch")
                mel = freevc_model.infer(c, g=tgt_embedding)
            else:
                # Fallback case for models that need mel for speaker encoding
                mel = freevc_model.infer(c, g=tgt_embedding, mel=source)

            # After inference
            print(f"Model output shape: {mel.shape}")

            # ========== SIMPLIFIED APPROACH ==========
            # Use direct output from FreeVC model without HiFiGAN
            # if len(mel.shape) >= 3:
            #     # If output is [batch, channel, time] or [batch, 1, channel, time]
            #     waveform_np = mel.squeeze().cpu().numpy()
            # else:
            #     # Handle unexpected shapes
            #     waveform_np = mel.cpu().numpy()

            # print(f"Using direct model output. Shape: {waveform_np.shape}")

        # Simple normalization without aggressive processing
        # waveform_np = waveform_np / (np.max(np.abs(waveform_np)) + 1e-6) * 0.9

        # Save as mono audio, ensuring we have a 1D array
        # if len(waveform_np.shape) > 1:
            # waveform_np = waveform_np.squeeze()

        # Final check to ensure we have a valid 1D waveform
        # if len(waveform_np.shape) > 1:
        #     print(
        #         f"WARNING: Unexpected audio shape: {waveform_np.shape}. Attempting to fix.")
        #     if waveform_np.shape[0] == 1:
        #         waveform_np = waveform_np[0]
        #     else:
        #         # Average across all dimensions
        #         waveform_np = waveform_np.mean(axis=0)

        sf.write(output_path, mel, 16000, subtype="PCM_16")

        end_time = time.time()
        print(
            f"🎉 Voice conversion completed in {end_time - start_time:.2f} seconds.")
        return {"success": True, "output_audio": output_path}
    except Exception as e:
        print(f"❌ Error in voice conversion: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def change_voice_real(src_audio_path, tgt_audio_path, output_path, strength):
    print(f"🔹 Starting voice conversion for: {src_audio_path}")
    start_time = time.time()
    try:
        if not os.path.exists(src_audio_path):
            raise FileNotFoundError(f"🚨 Audio file not found: {audio_path}")

        print(f"🔊 Processing target audio.")
        wav_tgt, _ = librosa.load(tgt_audio_path, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        if hps.model.use_spk:
            print(f"Use SPK")
            g_tgt = smodel.embed_utterance(wav_tgt) * strength
            print(f"Applied strength {strength} with proper normalization")
            g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to('cpu')
        else:
            print(f"Fallback, numpy spectrogram")
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to('cpu')
            mel_tgt = mel_spectrogram_torch(
                wav_tgt,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

        # process src
        print(f"🔊 Processing source audio.")
        wav_src, _ = librosa.load(src_audio_path, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).to('cpu')
        c = get_content(cmodel, wav_src)

        if hps.model.use_spk:
            print("🤖 Using spk inference.")
            audio = net_g.infer(c, g=g_tgt)
        else:
            print("🤖 Using fallback inference")
            audio = net_g.infer(c, mel=mel_tgt)
        audio = audio[0][0].data.cpu().float().numpy()

        write(output_path, hps.data.sampling_rate, audio)
        end_time = time.time()
        conversion_time = f"{end_time - start_time:.2f}"
        print(
            f"🎉 Voice conversion completed in {conversion_time} seconds.")
        return {"success": True, "output_path": output_path, "time": conversion_time}
    except Exception as e:
        print(f"❌ Error in voice conversion: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def extract_speaker_embedding(audio):
    """Extract speaker embedding from audio."""
    # In a full implementation, this would use a speaker encoder
    # For simplicity, we'll return a placeholder
    # Adjust size based on FreeVC requirements
    return torch.zeros(512).unsqueeze(0)


@app.route("/inference", methods=["POST"])
def inference():
    try:
        print("📩 Received a new inference request.")

        if net_g is None:
            print("❌ Model failed to load.")
            return jsonify({"error": "Model failed to load"}), 500

        data = request.json
        src_path = data.get("src_path")
        tgt_path = data.get("tgt_path")
        src_filename = data.get("src_filename")
        tgt_filename = data.get("tgt_filename")
        # voice_type = data.get("voice_type", "kid")  # Default to kid voice

        # Get strength parameter from request or use default
        strength = float(data.get("strength", 2.5))  # Increased default to 2.5

        # print(f"Using voice type: {voice_type} with strength: {strength}")

        if not src_path:
            print("❌ No audio file provided in request.")
            return jsonify({"error": "No audio file provided"}), 400

        src_audio_path = os.path.join(UPLOAD_FOLDER, src_filename)
        print(f"🔍 Checking if file exists: {src_audio_path}")

        if not src_audio_path or not os.path.exists(src_audio_path):
            print(f"❌ Audio file not found: {src_audio_path}")
            return jsonify({"error": "Audio file is missing or invalid"}), 400

        tgt_audio_path = os.path.join(UPLOAD_FOLDER, tgt_filename)
        print(f"🔍 Checking if file exists: {tgt_path}")

        if not tgt_audio_path or not os.path.exists(tgt_audio_path):
            print(f"❌ Audio file not found: {tgt_audio_path}")
            return jsonify({"error": "Audio file is missing or invalid"}), 400
        # Get the appropriate speaker embedding
        # speaker_embedding = get_speaker_embedding(voice_type)

        output_audio_path = os.path.join(
            UPLOAD_FOLDER, f"{src_filename}_converted.wav")
        print(f"🚀 Starting voice conversion process...{output_audio_path}")

        # result = change_voice(audio_path, output_audio_path,
        #   speaker_embedding, strength)

        result = change_voice_real(
            src_audio_path, tgt_audio_path, output_audio_path, strength)

        if result["success"]:
            print("✅ Voice conversion successful!")
            return jsonify({"output_path": result["output_path"], "time": result["time"]})
        else:
            print(f"❌ Error: {result['error']}")
            return jsonify({"error": result["error"]}), 500

    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


# Memory optimization for MPS (Apple Silicon)
def optimize_for_mps():
    """Apply optimizations for MPS device (Apple Silicon)"""
    if torch.backends.mps.is_available():
        # Empty the MPS cache to free up memory
        torch.mps.empty_cache()
        # Configure tensor allocation to be more efficient
        # Less aggressive caching
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        # CRITICAL: Enable CPU fallback for operations not supported by MPS
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("Applied MPS optimizations for Apple Silicon")
        print("Enabled CPU fallback for unsupported MPS operations")


# Call the optimization function
# optimize_for_mps()

if __name__ == "__main__":
    app.run(port=8000, debug=True)
