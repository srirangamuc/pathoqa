import os
import gc
import torch
import nltk
from PIL import ImageFile

def setup_system():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    nltk.download('punkt', quiet=True)
    print("System Settings Configured.")

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory Flushed.")
