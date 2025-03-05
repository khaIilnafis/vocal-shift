import torch
import librosa
import numpy as np
import os
import sys
import subprocess
from fairseq import checkpoint_utils
import soundfile as sf
import tempfile
from torch import nn
import torch.nn.functional as F

# Add FreeVC to path
sys.path.append('./FreeVC')

# Get the absolute path to the workspace directory
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure output directory exists
os.makedirs(os.path.join(WORKSPACE_DIR,
            'FreeVC/speaker_embeddings'), exist_ok=True)


def preprocess_audio(wav, sr=16000, normalize=True, remove_silence=True, target_length=None):
    """
    Preprocess audio for better embedding quality.

    Args:
        wav: Audio waveform
        sr: Sample rate
        normalize: Whether to normalize audio volume
        remove_silence: Whether to remove silent parts
        target_length: Target length in samples (if None, keep the original length)

    Returns:
        Preprocessed audio waveform
    """
    # Convert to mono if needed
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)

    # Resample if needed
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Normalize volume
    if normalize:
        wav = wav / (np.max(np.abs(wav)) + 1e-6)

    # Remove silence
    if remove_silence:
        intervals = librosa.effects.split(wav, top_db=30)
        wav_parts = []
        for interval in intervals:
            wav_parts.append(wav[interval[0]:interval[1]])
        if wav_parts:
            wav = np.concatenate(wav_parts)

    # Adjust length if specified
    if target_length is not None:
        if len(wav) > target_length:
            # Take the middle section
            start = (len(wav) - target_length) // 2
            wav = wav[start:start + target_length]
        elif len(wav) < target_length:
            # Pad with silence
            wav = np.pad(wav, (0, target_length - len(wav)), mode='constant')

    return wav


def extract_speaker_embedding(media_path, target_name="kid", model_path="./FreeVC/pretrained_models/hubert_base.pt",
                              target_dim=256, max_duration=120, segment_length=30):
    """
    Extract speaker embedding from an audio or video file using HuBERT model.

    Args:
        media_path: Path to the audio/video file (MP3, WAV, MP4, etc.)
        target_name: Name to save the embedding as (e.g., "kid", "female")
        model_path: Path to HuBERT model
        target_dim: The target dimension for the speaker embedding (default: 256)
        max_duration: Maximum duration to process in seconds (default: 120)
        segment_length: Length of each segment in seconds for averaging (default: 30)
    """
    print(f"Processing media file: {media_path}")

    # Convert relative path to absolute path if needed
    if not os.path.isabs(model_path):
        abs_model_path = os.path.join(WORKSPACE_DIR, model_path)
    else:
        abs_model_path = model_path

    # Check if the model file exists
    if not os.path.exists(abs_model_path):
        print(f"🚨 Model file not found at: {abs_model_path}")
        print("Looking for alternative locations...")

        # Try alternative locations
        alt_paths = [
            os.path.join(
                WORKSPACE_DIR, 'FreeVC/pretrained_models/hubert_base.pt'),
            './model/FreeVC/pretrained_models/hubert_base.pt',
            '/Users/khalilbrown/express-typescript-app/model/FreeVC/pretrained_models/hubert_base.pt'
        ]

        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                print(f"✅ Found model at alternative location: {alt_path}")
                abs_model_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"Could not find HuBERT model file. Tried paths: {[abs_model_path] + alt_paths}")

    print(f"Using model file: {abs_model_path}")

    # Check if the file is a video file
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    is_video = any(media_path.lower().endswith(ext)
                   for ext in video_extensions)

    # If video, extract audio to a temporary file
    if is_video:
        print("Detected video file. Extracting audio...")
        temp_audio = tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False).name
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
        # Load audio
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Limit duration to max_duration seconds to avoid excessive memory usage
        max_samples = max_duration * 16000
        if len(wav) > max_samples:
            print(
                f"Audio longer than {max_duration}s, trimming to first {max_duration}s")
            wav = wav[:max_samples]

        # Preprocess audio
        wav = preprocess_audio(wav, sr=sr, normalize=True, remove_silence=True)

        # Ensure sufficient length (at least 2 seconds)
        if len(wav) < 32000:
            print("Warning: Audio too short, padding to 2 seconds")
            wav = np.pad(wav, (0, 32000 - len(wav)), mode='constant')

        # Convert to tensor
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0)

        # Load HuBERT model
        print("Loading HuBERT model...")
        models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([
                                                                          abs_model_path])
        hubert_model = models[0]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hubert_model = hubert_model.to(device)
        hubert_model.eval()

        # Process in segments for longer audio
        segment_samples = segment_length * 16000
        num_segments = max(1, len(wav) // segment_samples)
        print(f"Processing audio in {num_segments} segments...")

        embeddings = []
        for i in range(num_segments):
            start = i * segment_samples
            end = min(start + segment_samples, len(wav))

            if end - start < 16000:  # Skip segments shorter than 1 second
                continue

            segment = wav[start:end]
            segment_tensor = torch.FloatTensor(segment).unsqueeze(0).to(device)

            with torch.no_grad():
                # Get HuBERT features for this segment
                feats = hubert_model.extract_features(
                    source=segment_tensor,
                    padding_mask=None,
                    mask=False,
                    # Use layer 11 instead of 9 for stronger speaker characteristics
                    # Layer 11 captures more of the unique speaker voice qualities
                    output_layer=11  # Changed from 9 to 11 for better voice character capture
                )[0]

                # Average the features to get a fixed-length embedding for this segment
                segment_embedding = torch.mean(feats, dim=1)
                embeddings.append(segment_embedding)

        # Average all segment embeddings
        if embeddings:
            embedding = torch.mean(
                torch.cat(embeddings, dim=0), dim=0, keepdim=True)
        else:
            # Fallback if no segments were processed
            with torch.no_grad():
                wav_tensor = wav_tensor.to(device)
                feats = hubert_model.extract_features(
                    source=wav_tensor,
                    padding_mask=None,
                    mask=False,
                    output_layer=9
                )[0]
                embedding = torch.mean(feats, dim=1)

        # Reshape embedding to target dimension if needed
        orig_dim = embedding.shape[1]
        if orig_dim != target_dim:
            print(
                f"Reshaping embedding from dimension {orig_dim} to {target_dim}")

            # Linear projection to target dimension
            if orig_dim > target_dim:
                projection = nn.Linear(orig_dim, target_dim).to(device)
                # Initialize with PCA-like weights for better preservation of information
                nn.init.orthogonal_(projection.weight)
                embedding = projection(embedding)
            else:
                # If we need to expand, use a simple projection
                projection = nn.Linear(orig_dim, target_dim).to(device)
                nn.init.normal_(projection.weight, std=0.02)
                embedding = projection(embedding)

        # Normalize the embedding to unit length
        embedding = F.normalize(embedding, p=2, dim=1)

        # Save the embedding
        output_path = os.path.join(
            WORKSPACE_DIR, f"FreeVC/speaker_embeddings/{target_name}.pt")
        torch.save(embedding, output_path)
        print(
            f"✅ Speaker embedding saved to {output_path} with shape {embedding.shape}")
        return embedding

    finally:
        # Clean up temporary file if we created one
        if is_video and os.path.exists(temp_audio):
            os.unlink(temp_audio)
            print("Temporary audio file removed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--media", type=str, required=True,
                        help="Path to audio/video file")
    parser.add_argument("--name", type=str, default="kid",
                        help="Name for the embedding (kid, female, male)")
    parser.add_argument("--model", type=str, default="./FreeVC/pretrained_models/hubert_base.pt",
                        help="Path to HuBERT model")
    parser.add_argument("--dimension", type=int, default=256,
                        help="Target dimension for the speaker embedding")
    parser.add_argument("--max-duration", type=int, default=120,
                        help="Maximum duration to process in seconds")
    parser.add_argument("--segment-length", type=int, default=30,
                        help="Length of each segment in seconds for averaging")
    args = parser.parse_args()

    extract_speaker_embedding(
        args.media,
        args.name,
        args.model,
        target_dim=args.dimension,
        max_duration=args.max_duration,
        segment_length=args.segment_length
    )
