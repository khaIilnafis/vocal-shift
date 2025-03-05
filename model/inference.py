import json
import os
import time
import numpy as np
import traceback
import torch
import soundfile as sf
import librosa
from flask import Flask, request, jsonify

# Import FreeVC components
import sys
sys.path.append('./FreeVC')  # Add FreeVC directory to path
from FreeVC.wavlm import WavLM, WavLMConfig  # NOQA
from utils import get_hparams_from_file, load_checkpoint, get_vocoder  # NOQA
from FreeVC.text import text_to_sequence  # NOQA
from FreeVC.models import SynthesizerTrn  # NOQA
from FreeVC.hifigan.models import Generator  # NOQA
from FreeVC.hifigan import AttrDict  # NOQA

app = Flask(__name__)
# Get the absolute path to the `upload` folder
UPLOAD_FOLDER = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "express-typescript-app", "../../uploads"))

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create model cache directory
os.makedirs("./model_cache", exist_ok=True)

# Global variables for models
freevc_model = None
content_model = None
vocoder = None
speaker_embeddings = {}

# Custom function to load WavLM model from the correct path


def get_content_model(device='cuda'):
    # Use the correct path to the WavLM model
    wavlm_path = './FreeVC/wavlm/WavLM-Large.pt'
    if not os.path.exists(wavlm_path):
        raise FileNotFoundError(f"WavLM model not found at {wavlm_path}")

    checkpoint = torch.load(wavlm_path, map_location=device)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

# Custom function to extract content from audio


def extract_content(cmodel, audio):
    with torch.no_grad():
        # WavLM expects audio shape [batch_size, audio_length]
        # so we need to squeeze the channel dimension
        features = cmodel.extract_features(audio.squeeze(1))[0]
    # Transpose to get [batch_size, feature_dim, time]
    return features.transpose(1, 2)

# Function to convert raw audio to mel spectrogram


def audio_to_mel(audio, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=16000, n_mel_channels=80):
    """Convert raw audio to mel spectrogram compatible with HiFiGAN"""
    if torch.is_tensor(audio):
        audio = audio.squeeze().cpu().numpy()

    # Improve the mel-spectrogram calculation
    # Using librosa's defaults for clean speech
    stft = librosa.stft(audio,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window='hann',
                        center=True)

    # Get magnitudes and use power spectrogram (squared magnitudes)
    magnitudes = np.abs(stft) ** 2

    # Create mel filterbank with proper settings
    fmin = 0  # Lowest frequency in mel filterbank
    fmax = sampling_rate / 2  # Highest frequency (Nyquist frequency)

    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=n_fft,
        n_mels=n_mel_channels,
        fmin=fmin,
        fmax=fmax
    )

    # Apply mel filterbank and convert to log domain with proper scaling
    mel = np.log10(np.maximum(np.dot(mel_basis, magnitudes), 1e-9))

    # Normalize to match HiFiGAN's expected input range
    # Typically HiFiGAN expects values in a specific range
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    # Convert to tensor and add batch dimension
    # Output should be [1, n_mel_channels, time] for HiFiGAN
    mel_tensor = torch.FloatTensor(mel).unsqueeze(0)
    return mel_tensor

# Function to adapt speaker embeddings to the required dimensions


def adapt_speaker_embedding(embedding, target_dim=256):
    """
    Adapts a speaker embedding to the required dimension for the FreeVC model.

    Args:
        embedding: The original speaker embedding tensor
        target_dim: The target channel dimension required by the model

    Returns:
        A tensor with the required channel dimension
    """
    # Print shape to help with debugging
    print(f"Original embedding shape: {embedding.shape}")

    # Check the shape and adjust approach based on dimensionality
    if len(embedding.shape) == 2:
        # Shape is [batch_size, features]
        orig_dim = embedding.shape[1]

        # If dimensions already match, return as is
        if orig_dim == target_dim:
            return embedding

        print(
            f"Adapting 2D speaker embedding from dimension {orig_dim} to {target_dim}")

        # For downsampling (if orig_dim > target_dim)
        if orig_dim > target_dim:
            # Use a subset of the features
            return embedding[:, :target_dim]

        # For upsampling (if orig_dim < target_dim)
        else:
            # Pad with zeros
            padded = torch.zeros(
                embedding.shape[0], target_dim, device=embedding.device)
            padded[:, :orig_dim] = embedding
            return padded

    elif len(embedding.shape) == 3:
        # Shape is [batch_size, channels, time]
        orig_dim = embedding.shape[1]

        # If dimensions already match, return as is
        if orig_dim == target_dim:
            return embedding

        print(
            f"Adapting 3D speaker embedding from dimension {orig_dim} to {target_dim}")

        # For downsampling (if orig_dim > target_dim)
        if orig_dim > target_dim:
            # Use interpolation for smoother reduction
            return torch.nn.functional.interpolate(
                embedding,
                size=(target_dim, embedding.shape[2]),
                mode='bilinear',
                align_corners=False
            )

        # For upsampling (if orig_dim < target_dim)
        else:
            # Pad with zeros
            padded = torch.zeros(embedding.shape[0], target_dim, embedding.shape[2],
                                 device=embedding.device)
            padded[:, :orig_dim, :] = embedding
            return padded

    else:
        # Handle unexpected shapes - most likely just needs reshaping
        print(
            f"Unexpected embedding shape: {embedding.shape}, attempting to reshape")

        # Try to convert to expected 2D shape for the model
        if embedding.dim() == 1:
            # Single vector - add batch dimension and reshape to target
            embedding = embedding.unsqueeze(0)

            if len(embedding) != target_dim:
                # Need to resize
                if len(embedding) > target_dim:
                    return embedding[:target_dim]
                else:
                    padded = torch.zeros(target_dim, device=embedding.device)
                    padded[:len(embedding)] = embedding
                    return padded.unsqueeze(0)
            return embedding

        # If it's a strange shape, just flatten and resize
        flattened = embedding.reshape(1, -1)
        if flattened.shape[1] >= target_dim:
            return flattened[:, :target_dim]
        else:
            padded = torch.zeros(1, target_dim, device=embedding.device)
            padded[:, :flattened.shape[1]] = flattened
            return padded


try:
    # Initialize FreeVC model
    print("Loading FreeVC model...")

    # Load config (adjust paths as needed)
    hps = get_hparams_from_file('./FreeVC/configs/freevc.json')

    # Initialize the model
    freevc_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model)

    # Load the pretrained model
    _ = freevc_model.eval()
    _ = load_checkpoint(
        './FreeVC/pretrained_models/freevc.pth', freevc_model, None, True)

    print("✅ FreeVC model loaded successfully")

    # Load WavLM model for content extraction
    print("Loading WavLM model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    content_model = get_content_model(device)
    print("✅ WavLM model loaded successfully")

    # Load HiFiGAN vocoder for higher quality synthesis
    print("Loading HiFiGAN vocoder...")
    try:
        # Try to load the vocoder
        with open("./FreeVC/hifigan/config_v3.json", "r") as f:
            config = json.load(f)
        config = AttrDict(config)
        vocoder = Generator(config)
        ckpt = torch.load("./FreeVC/hifigan/generator_v3", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder = vocoder.to(device)
        print("✅ HiFiGAN v3 vocoder loaded successfully")
    except Exception as e:
        print(f"Warning: Could not load HiFiGAN v3, trying v1: {e}")
        try:
            # Fallback to v1 if available
            with open("./FreeVC/hifigan/config.json", "r") as f:
                config = json.load(f)
            config = AttrDict(config)
            vocoder = Generator(config)
            ckpt = torch.load("./FreeVC/hifigan/generator_v1",
                              map_location=device)
            vocoder.load_state_dict(ckpt["generator"])
            vocoder.eval()
            vocoder.remove_weight_norm()
            vocoder = vocoder.to(device)
            print("✅ HiFiGAN v1 vocoder loaded successfully")
        except Exception as e2:
            print(f"Warning: Could not load HiFiGAN vocoder: {e2}")
            vocoder = None
            print("⚠️ Continuing without HiFiGAN vocoder - audio quality may be reduced")

    # Load speaker embeddings from a predefined set
    # For simplicity, we'll create a mapping of voice types
    speaker_embeddings = {
        "kid": torch.load('./FreeVC/speaker_embeddings/kid.pt'),
        "kid2": torch.load('./FreeVC/speaker_embeddings/kid2.pt'),
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
    freevc_model = None
    content_model = None
    vocoder = None


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
            c = extract_content(content_model, source)

            # Strength value controls how much to apply the target voice characteristics
            # Higher values = stronger voice conversion but may introduce artifacts
            # Going back to a moderate value

            # Use the provided target speaker embedding and adapt it
            tgt_embedding = speaker_embedding.to(device)

            # Adapt the embedding to match the expected dimensions (256 channels)
            tgt_embedding = adapt_speaker_embedding(
                tgt_embedding, target_dim=256)

            # Apply the strength factor
            tgt_embedding = tgt_embedding * strength

            # Print debug info
            print(f"Shape after processing: {tgt_embedding.shape}")

            # Get global hps variable
            global hps

            # Perform voice conversion using the infer method
            if hasattr(hps.model, 'use_spk') and hps.model.use_spk:
                # This generates mel-spectrograms or audio
                mel = freevc_model.infer(c, g=tgt_embedding)
            else:
                # Fallback case
                mel = freevc_model.infer(c, g=tgt_embedding)

            # After inference
            print(f"Model output shape: {mel.shape}")

            # ========== SIMPLIFIED APPROACH ==========
            # Use direct output from FreeVC model without HiFiGAN
            if len(mel.shape) >= 3:
                # If output is [batch, channel, time] or [batch, 1, channel, time]
                waveform_np = mel.squeeze().cpu().numpy()
            else:
                # Handle unexpected shapes
                waveform_np = mel.cpu().numpy()

            print(f"Using direct model output. Shape: {waveform_np.shape}")

        # Simple normalization without aggressive processing
        waveform_np = waveform_np / (np.max(np.abs(waveform_np)) + 1e-6) * 0.9

        # Save as mono audio, ensuring we have a 1D array
        if len(waveform_np.shape) > 1:
            waveform_np = waveform_np.squeeze()

        # Final check to ensure we have a valid 1D waveform
        if len(waveform_np.shape) > 1:
            print(
                f"WARNING: Unexpected audio shape: {waveform_np.shape}. Attempting to fix.")
            if waveform_np.shape[0] == 1:
                waveform_np = waveform_np[0]
            else:
                # Average across all dimensions
                waveform_np = waveform_np.mean(axis=0)

        sf.write(output_path, waveform_np, 16000, subtype="PCM_16")

        end_time = time.time()
        print(
            f"🎉 Voice conversion completed in {end_time - start_time:.2f} seconds.")
        return {"success": True, "output_audio": output_path}
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

        if freevc_model is None:
            print("❌ Model failed to load.")
            return jsonify({"error": "Model failed to load"}), 500

        data = request.json
        audio_filename = data.get("audio_filename")
        voice_type = data.get("voice_type")  # Default to kid voice
        strength = data.get("strength")
        print(f"Voice selected: {voice_type}")
        print(f"Strength selected: {strength}")
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

        result = change_voice(audio_path, output_audio_path,
                              speaker_embedding, strength)

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
