import os
import gc
import torch
import nltk
from PIL import ImageFile

def setup_system():
    # Suppress Tokenizer Parallelism Warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Handle Truncated Images (Common in Medical Datasets)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Download NLTK data for BLEU/ROUGE if needed
    nltk.download('punkt', quiet=True)
    print("✅ System Settings Configured.")

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Memory Flushed.")
