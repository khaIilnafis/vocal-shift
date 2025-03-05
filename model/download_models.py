import os
import torch
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan, SpeechT5Config
from datasets import load_dataset

# Create cache directory
os.makedirs("./model_cache", exist_ok=True)

print("Downloading SpeechT5 model and vocoder...")

# Download main model
config = SpeechT5Config.from_pretrained("microsoft/speecht5_vc", cache_dir="./model_cache")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc", cache_dir="./model_cache", force_download=True)
model = SpeechT5ForSpeechToSpeech.from_pretrained(
    "microsoft/speecht5_vc", 
    config=config,
    cache_dir="./model_cache", 
    force_download=True
)

# Download vocoder
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cache_dir="./model_cache", force_download=True)

# Download speaker embeddings
dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(dataset[7306]["xvector"]).unsqueeze(0)

print("All models downloaded successfully!") 