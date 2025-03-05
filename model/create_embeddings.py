import torch
import librosa
import numpy as np
import os
import sys
import subprocess
from fairseq import checkpoint_utils
import soundfile as sf
import tempfile

# Add FreeVC to path
sys.path.append('./FreeVC')

# Ensure output directory exists
os.makedirs('./FreeVC/speaker_embeddings', exist_ok=True)

def extract_speaker_embedding(media_path, target_name="kid", model_path="./FreeVC/pretrained_models/hubert_base.pt"):
    """
    Extract speaker embedding from an audio or video file using HuBERT model.
    
    Args:
        media_path: Path to the audio/video file (MP3, WAV, MP4, etc.)
        target_name: Name to save the embedding as (e.g., "kid", "female")
        model_path: Path to HuBERT model
    """
    print(f"Processing media file: {media_path}")
    
    # Check if the file is a video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    is_video = any(media_path.lower().endswith(ext) for ext in video_extensions)
    
    # If video, extract audio to a temporary file
    if is_video:
        print("Detected video file. Extracting audio...")
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            cmd = [
                'ffmpeg', '-i', media_path, 
                '-q:a', '0', '-map', 'a', 
                '-y', temp_audio
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Audio extracted to temporary file: {temp_audio}")
            audio_path = temp_audio
        except subprocess.CalledProcessError as e:
            print(f"Error extracting audio from video: {e}")
            raise
    else:
        audio_path = media_path
    
    try:
        # Load and preprocess audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        # Ensure sufficient length (at least 2 seconds)
        if len(wav) < 32000:
            wav = np.pad(wav, (0, 32000 - len(wav)), mode='constant')
        # Trim to 30 seconds max to avoid memory issues
        if len(wav) > 480000:  # 30 seconds at 16kHz
            wav = wav[:480000]
        
        # Convert to tensor
        wav = torch.FloatTensor(wav).unsqueeze(0)
        
        # Load HuBERT model
        print("Loading HuBERT model...")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path])
        hubert_model = models[0]
        hubert_model = hubert_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        hubert_model.eval()
        
        # Extract embedding
        print("Extracting speaker embedding...")
        with torch.no_grad():
            if torch.cuda.is_available():
                wav = wav.cuda()
            
            # Get HuBERT features
            feats = hubert_model.extract_features(
                source=wav, 
                padding_mask=None, 
                mask=False, 
                output_layer=9
            )[0]
            
            # Average the features to get a fixed-length embedding
            embedding = torch.mean(feats, dim=1)
            
        # Save the embedding
        output_path = f"./FreeVC/speaker_embeddings/{target_name}.pt"
        torch.save(embedding, output_path)
        print(f"✅ Speaker embedding saved to {output_path}")
        return embedding
    
    finally:
        # Clean up temporary file if we created one
        if is_video and os.path.exists(temp_audio):
            os.unlink(temp_audio)
            print("Temporary audio file removed")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=str, required=True, help="Path to audio/video file")
    parser.add_argument("--name", type=str, default="kid", help="Name for the embedding (kid, female, male)")
    parser.add_argument("--model", type=str, default="./FreeVC/pretrained_models/hubert_base.pt", 
                        help="Path to HuBERT model")
    args = parser.parse_args()
    
    extract_speaker_embedding(args.media, args.name, args.model) 