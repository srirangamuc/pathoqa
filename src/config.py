import torch
import argparse
import os

def get_default_config():
    return {
        # Paths
        "t5_path": "./final_pathology_model_v2",
        "clip_path": "jamessyx/pathgenclip-vit-large-patch14-hf",
        "output_dir": "./pathvqa_experiment_optimized",

        # Hyperparameters
        "epochs": 10,
        "batch_size": 8,
        "lr": 1e-4,
        "max_len": 128,
        "image_size": 336,

        # Hardware
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

def parse_args():
    parser = argparse.ArgumentParser(description="PathVQA Training/Evaluation Config")
    defaults = get_default_config()

    parser.add_argument("--t5_path", type=str, default=defaults["t5_path"], help="Path to T5 model")
    parser.add_argument("--clip_path", type=str, default=defaults["clip_path"], help="Path to CLIP model")
    parser.add_argument("--output_dir", type=str, default=defaults["output_dir"], help="Output directory")
    
    parser.add_argument("--epochs", type=int, default=defaults["epochs"], help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=defaults["lr"], help="Learning rate")
    parser.add_argument("--max_len", type=int, default=defaults["max_len"], help="Max sequence length")
    parser.add_argument("--image_size", type=int, default=defaults["image_size"], help="Image size")
    
    parser.add_argument("--device", type=str, default=defaults["device"], help="Device (cuda/cpu)")

    args = parser.parse_args()
    
    # ensure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    return vars(args)
