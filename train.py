import os
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from src.config import parse_args
from src.utils import setup_system, flush_memory
from src.model import PathVQA_Fusion_Model, FusionOutput
from src.dataset import get_processors, get_dataloaders
from src.evaluator import PathVQA_Evaluator

def main():
    # 1. Config & Setup
    config = parse_args()
    print(f"✅ Config loaded. Device: {config['device']}")
    
    setup_system()
    flush_memory()
    
    # 2. Data & Model
    print("📦 Preparing Data Processors...")
    tokenizer, processor = get_processors(config["t5_path"], config["clip_path"], config["image_size"])
    print(f"✅ Processor enforced to {config['image_size']}px")
    
    train_loader, val_loader, test_loader, ds = get_dataloaders(config, tokenizer, processor)
    print(f"✅ Loaders ready. Train Size: {len(ds['train'])}")
    
    print("\nIntializing Model...")
    model = PathVQA_Fusion_Model(config["t5_path"], config["clip_path"]).to(config["device"])
    
    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    
    total_steps = len(train_loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )
    
    evaluator = PathVQA_Evaluator(config["device"])
    
    # --- CHECKPOINT LOADING ---
    start_epoch = 0
    best_score = -1.0
    checkpoint_path = os.path.join(config["output_dir"], "last_checkpoint.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"🔄 Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=config["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", -1.0)
        print(f"✅ Resuming from Epoch {start_epoch}")
    else:
        print("🆕 No checkpoint found. Starting from scratch.")
        
    print(f"✅ Setup Complete. Training for {total_steps} steps.")
    
    # 4. Training Loop
    print(f"\n🚀 STARTING TRAINING (Epochs: {config['epochs']})...")

    for epoch in range(start_epoch, config["epochs"]):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for batch in loop:
            optimizer.zero_grad()
            pixel_values = batch["pixel_values"].to(config["device"])
            input_ids = batch["input_ids"].to(config["device"])
            attention_mask = batch["attention_mask"].to(config["device"])
            labels = batch["labels"].to(config["device"])
            
            loss = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            ).loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
        # --- VALIDATE ---
        model.eval()
        all_preds, all_truths = [], []
        
        print("🔍 Validating...")
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = {k: v.to(config["device"]) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Efficient Explicit Fusion for Generation
                v_out = model.vision_encoder(batch["pixel_values"]).last_hidden_state
                v_emb = model.vis_project(v_out)
                
                t_emb = model.t5.encoder(batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                
                fused = model.fusion_module(tgt=t_emb, memory=v_emb)
                fused_output = FusionOutput(last_hidden_state=fused)
                
                gen_ids = model.t5.generate(
                    encoder_outputs=fused_output,
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=32
                )
                preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                
                all_preds.extend(preds)
                all_truths.extend(batch["raw_answers"])
                
        # Metrics
        metrics = evaluator.compute_metrics(all_preds, all_truths)
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   - Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"   - Overall Score: {metrics['overall_score']:.4f}")
        print(f"   - Binary Acc: {metrics['binary_accuracy']:.2%} ({metrics['binary_count']})")
        print(f"   - Open Exact: {metrics['open_exact_match']:.2%} (Bleu: {metrics['open_bleu']:.4f}) (SBERT: {metrics['open_sbert']:.4f})")
        
        # Save Best
        if metrics['overall_score'] > best_score:
            best_score = metrics['overall_score']
            print(f"   🌟 New Best Model! Saving to {config['output_dir']}...")
            torch.save(model.state_dict(), f"{config['output_dir']}/best_model.pth")
            
        # Save Last Checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_score": best_score
        }
        torch.save(checkpoint, os.path.join(config["output_dir"], "last_checkpoint.pth"))
        print(f"   💾 Checkpoint saved for Epoch {epoch+1}")
        
    print("\n✅ TRAINING FINISHED.")

if __name__ == "__main__":
    main()
