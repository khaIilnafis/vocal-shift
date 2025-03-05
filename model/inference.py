import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
from datasets import load_dataset
from flask import Flask, request, jsonify
import librosa
import soundfile as sf
import os
import time
import numpy as np
import traceback

app = Flask(__name__)
# Get the absolute path to the `upload` folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "express-typescript-app", "../../uploads"))

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
try:
    # Load SpeechT5 Model
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
    model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")

    # Load target speaker embedding (A "kid-like" speaker from a dataset)
    dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(dataset[7306]["xvector"]).unsqueeze(
        0)  # Example speaker with a high-pitched voice
except Exception as e:
    print(f"Error loading model: {e}")
    processor, model, speaker_embedding = None, None, None  # Prevent crashes


def get_speaker_embedding(speaker_type="kid"):
    """Get speaker embedding based on desired voice type."""
    dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    
    # Dictionary mapping voice types to dataset indices
    voice_indices = {
        "kid": 7306,  # Current index you're using
        "female": 0,   # Find a good female index
        "male": 100,   # Find a good male index
        "high": 2000,  # Find a high-pitched voice
        "low": 3000,   # Find a low-pitched voice
    }
    
    # Get the index for the requested voice type
    index = voice_indices.get(speaker_type, 7306)  # Default to current
    
    return torch.tensor(dataset[index]["xvector"]).unsqueeze(0)


def change_voice(audio_path, output_path):
    try:
        print(f"🔹 Starting voice conversion for: {audio_path}")
        start_time = time.time()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"🚨 Audio file not found: {audio_path}")

        # Load input audio
        speech_array, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Convert to model format
        inputs = processor(audio=speech_array, sampling_rate=16000, return_tensors="pt")
        
        # Voice conversion
        with torch.no_grad():
            # Load the vocoder properly
            from transformers import SpeechT5HifiGan
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Generate speech features
            speech_features = model.generate_speech(inputs.input_values, speaker_embedding)
            print(f"Model output features shape: {speech_features.shape}")
            
            # Make sure the speech features are properly formatted for the vocoder
            # The vocoder expects [batch_size, sequence_length, hidden_size]
            if len(speech_features.shape) == 2:
                # If shape is [seq_len, hidden_size], add batch dimension
                speech_features = speech_features.unsqueeze(0)
            
            try:
                # Convert to waveform using vocoder
                waveform = vocoder(speech_features)
                print(f"Generated waveform shape: {waveform.shape}")
                
                # Get the waveform as numpy array
                waveform_np = waveform.squeeze().cpu().numpy()
                
                # Normalize to prevent clipping
                waveform_np = waveform_np / (np.max(np.abs(waveform_np)) + 1e-6)
            except Exception as vocoder_error:
                print(f"Vocoder error: {vocoder_error}")
                # Fallback: use direct output (might be noisy but better than nothing)
                waveform_np = speech_features.squeeze().cpu().numpy()
            
        # Save as mono audio with explicit format
        sf.write(output_path, waveform_np, 16000, subtype="PCM_16")
        
        end_time = time.time()
        print(f"🎉 Voice conversion completed in {end_time - start_time:.2f} seconds.")
        return {"success": True, "output_audio": output_path}
    except Exception as e:
        print(f"❌ Error in voice conversion: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.route("/inference", methods=["POST"])
def inference():
    try:
        print("📩 Received a new inference request.")

        if not processor or not model:
            print("❌ Model failed to load.")
            return jsonify({"error": "Model failed to load"}), 500

        data = request.json
        audio_filename = data.get("audio_filename")
        voice_type = data.get("voice_type", "kid")  # Default to kid voice

        if not audio_filename:
            print("❌ No audio file provided in request.")
            return jsonify({"error": "No audio file provided"}), 400

        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)
        print(f"🔍 Checking if file exists: {audio_path}")

        if not audio_path or not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return jsonify({"error": "Audio file is missing or invalid"}), 400

        # Get the appropriate speaker embedding
        speaker_embedding = get_speaker_embedding(voice_type)

        output_audio_path = f"{os.path.splitext(audio_path)[0]}_converted.wav"
        print("🚀 Starting voice conversion process...")

        result = change_voice(audio_path, output_audio_path)

        if result["success"]:
            print("✅ Voice conversion successful!")
            return jsonify({"output_audio": output_audio_path})
        else:
            print(f"❌ Error: {result['error']}")
            return jsonify({"error": result["error"]}), 500

    except Exception as e:
        print(f"❌ Server error: {e}")
        # Prevent server crash
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True)
