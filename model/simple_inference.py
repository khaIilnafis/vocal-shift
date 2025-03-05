import os
import time
import numpy as np
import traceback
import torch
import soundfile as sf
import librosa
from flask import Flask, request, jsonify

app = Flask(__name__)
# Get the absolute path to the `upload` folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "express-typescript-app", "../../uploads"))

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("./model_cache", exist_ok=True)
os.makedirs("./FreeVC/speaker_embeddings", exist_ok=True)

class SimplifiedVoiceConverter:
    """A simplified voice conversion implementation that doesn't require FreeVC"""
    
    def __init__(self):
        self.sr = 16000  # Sample rate
        print("✅ Simplified voice converter initialized")
    
    def voice_conversion(self, src_embedding, tgt_embedding, source):
        """
        Simple voice conversion using basic pitch shifting
        In a real implementation, this would use a neural model
        """
        # Convert to numpy
        audio = source.squeeze().numpy()
        
        # Get target characteristics from embedding
        # This is a simplification - in a real model, we'd use the embedding more effectively
        # Here we just use the embedding to determine how much to shift the pitch
        shift_factor = 4.0  # Default for child voice
        
        if torch.norm(tgt_embedding) > 0:
            # Use the embedding to determine shift direction and magnitude
            # This is just an example heuristic
            shift_factor = min(12, max(-12, float(torch.mean(tgt_embedding) * 10)))
        
        # Apply pitch shifting (simple method, not as good as neural conversion)
        try:
            # Get target pitch properties from embedding
            pitch_shift = shift_factor
            
            # Apply librosa pitch shift
            # Modified audio using pitch shifting
            audio_shifted = librosa.effects.pitch_shift(
                audio, sr=self.sr, n_steps=pitch_shift
            )
            
            # Formant modification to make it sound more like a child
            # (Basic approximation)
            if 'kid' in str(tgt_embedding):
                # Simple formant shifting by resampling
                audio_shifted = librosa.resample(
                    audio_shifted, orig_sr=self.sr, target_sr=int(self.sr * 1.2)
                )
                audio_shifted = librosa.resample(
                    audio_shifted, orig_sr=int(self.sr * 1.2), target_sr=self.sr
                )
            
            # Convert back to tensor format
            return torch.tensor(audio_shifted).unsqueeze(0).unsqueeze(0)
            
        except Exception as e:
            print(f"Error in voice conversion: {e}")
            traceback.print_exc()
            return source  # Return original audio if conversion fails

# Initialize the simplified voice converter
voice_converter = SimplifiedVoiceConverter()

# Load or create speaker embeddings
speaker_embeddings = {}
try:
    for voice_type in ['kid', 'female', 'male']:
        emb_path = f'./FreeVC/speaker_embeddings/{voice_type}.pt'
        if os.path.exists(emb_path):
            speaker_embeddings[voice_type] = torch.load(emb_path)
        else:
            # Create simple placeholder embeddings if they don't exist
            print(f"Creating placeholder embedding for {voice_type}")
            if voice_type == 'kid':
                # High values = higher pitch shift for kids
                emb = torch.ones(512) * 0.8
            elif voice_type == 'female':
                # Medium values for female
                emb = torch.ones(512) * 0.3
            elif voice_type == 'male':
                # Negative values for male (lower pitch)
                emb = torch.ones(512) * -0.2
                
            speaker_embeddings[voice_type] = emb.unsqueeze(0)
            torch.save(emb.unsqueeze(0), emb_path)
            
    print("✅ Speaker embeddings loaded successfully")
except Exception as e:
    print(f"Error loading embeddings: {e}")
    traceback.print_exc()


def get_speaker_embedding(speaker_type="kid"):
    """Get speaker embedding based on desired voice type."""
    if speaker_type in speaker_embeddings:
        return speaker_embeddings[speaker_type]
    else:
        print(f"Speaker type {speaker_type} not found, falling back to 'kid'")
        return speaker_embeddings["kid"]


def extract_speaker_embedding(audio):
    """Extract speaker embedding from audio."""
    # Simple placeholder method - in a real implementation, this would use a speaker encoder
    return torch.zeros(512).unsqueeze(0)


def change_voice(audio_path, output_path, speaker_embedding=None):
    try:
        print(f"🔹 Starting voice conversion for: {audio_path}")
        start_time = time.time()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"🚨 Audio file not found: {audio_path}")

        # Load input audio
        source_audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Convert to tensor
        source = torch.FloatTensor(source_audio).unsqueeze(0)
        
        # Extract source speaker embedding (simple placeholder)
        src_embedding = extract_speaker_embedding(source_audio)
        
        # Use the provided target speaker embedding
        tgt_embedding = speaker_embedding if speaker_embedding is not None else get_speaker_embedding("kid")
        
        # Perform voice conversion
        audio = voice_converter.voice_conversion(
            src_embedding, 
            tgt_embedding,
            source
        )
        
        # Convert to numpy array
        waveform_np = audio[0, 0].cpu().numpy()
            
        # Save as mono audio
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

        result = change_voice(audio_path, output_audio_path, speaker_embedding)

        if result["success"]:
            print("✅ Voice conversion successful!")
            return jsonify({"output_audio": output_audio_path})
        else:
            print(f"❌ Error: {result['error']}")
            return jsonify({"error": result["error"]}), 500

    except Exception as e:
        print(f"❌ Server error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(port=8000, debug=True) 