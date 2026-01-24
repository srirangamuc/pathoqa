import os
import torch
import numpy as np
from tqdm import tqdm

from src.config import parse_args
from src.model import PathVQA_Fusion_Model, FusionOutput
from src.dataset import get_processors, get_dataloaders
from src.evaluator import PathVQA_Evaluator

def main():
    config = parse_args()
    print(f"Config loaded. Device: {config['device']}")
    
    print("Preparing Data Processors...")
    tokenizer, processor = get_processors(config["t5_path"], config["clip_path"], config["image_size"])
    
    _, _, test_loader, _ = get_dataloaders(config, tokenizer, processor)
    
    print("\nIntializing Model...")
    model = PathVQA_Fusion_Model(config["t5_path"], config["clip_path"]).to(config["device"])
    
    print("\nFinal evaluation...")
    best_model_path = os.path.join(config["output_dir"], "best_model.pth")
    
    if os.path.exists(best_model_path):
        print(f"Loading weights from: {best_model_path}")
        state_dict = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Best model not found. Using current random weights (Expect poor performance).")
        
    model.eval()
    model.to(config["device"])
    
    evaluator = PathVQA_Evaluator(config["device"])
    
    all_preds = []
    all_truths = []
    all_questions = []
    
    print("🚀 Generating Answers on Test Set (Beam Search)...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = {k: v.to(config["device"]) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            vis_out = model.vision_encoder(batch["pixel_values"]).last_hidden_state
            vis_embeds = model.vis_project(vis_out)
            text_embeds = model.t5.encoder(batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
            
            fused = model.fusion_module(tgt=text_embeds, memory=vis_embeds)
            fused_output = FusionOutput(last_hidden_state=fused)
            
            gen_ids = model.t5.generate(
                encoder_outputs=fused_output, 
                max_new_tokens=50,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
                repetition_penalty=1.2
            )
            
            preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            
            all_preds.extend(preds)
            all_truths.extend(batch["raw_answers"])
            qs = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            all_questions.extend(qs)

    metrics = evaluator.compute_metrics(all_preds, all_truths)

    print(f"\nFINAL TEST REPORT")
    print(f"{'='*40}")
    print(f"Overall Score:{metrics['overall_score']:.4f}")
    print(f"Binary Accuracy:{metrics['binary_accuracy']:.2%} (N={metrics['binary_count']})")
    print(f"Open Exact Match:{metrics['open_exact_match']:.2%} (N={metrics['open_count']})")
    print(f"Open SBERT:{metrics['open_sbert']:.4f}")
    print(f"Open BLEU:{metrics['open_bleu']:.4f}")
    print(f"Open ROUGE-L:{metrics['open_rouge_l']:.4f}")
    print(f"{'='*40}")

    print("\nSAMPLES:")
    sample_size = min(5, len(all_preds))
    indices = np.random.choice(len(all_preds), sample_size, replace=False)

    for i in indices:
        q_text = all_questions[i].replace("Question:", "").strip()
        print(f"\nQ: {q_text}")
        print(f"Truth:{all_truths[i]}")
        print(f"Pred:{all_preds[i]}")

if __name__ == "__main__":
    main()
