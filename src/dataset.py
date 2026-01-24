from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, CLIPImageProcessor
from datasets import load_dataset
import torch

def get_processors(t5_path, clip_path, image_size):
    tokenizer = T5TokenizerFast.from_pretrained(t5_path)
    processor = CLIPImageProcessor.from_pretrained(clip_path)

    if hasattr(processor, "size"):
        processor.size = {"height": image_size, "width": image_size}
    if hasattr(processor, "crop_size"):
        processor.crop_size = {"height": image_size, "width": image_size}
    
    return tokenizer, processor

def vqa_collate_fn(batch, tokenizer, processor, max_len):
    images = []
    for x in batch:
        img = x['image']
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    pixel_values = processor(images, return_tensors="pt").pixel_values

    questions = ["Question: " + x['question'] for x in batch]
    inputs = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    answers = [str(x['answer']) for x in batch]
    labels = tokenizer(
        answers,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    ).input_ids

    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels,
        "raw_answers": answers
    }

def get_dataloaders(config, tokenizer, processor):
    ds = load_dataset("flaviagiammarino/path-vqa")
    def collate_wrapper(batch):
        return vqa_collate_fn(batch, tokenizer=tokenizer, processor=processor, max_len=config["max_len"])

    train_loader = DataLoader(
        ds["train"],
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_wrapper,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        ds["validation"],
        batch_size=config["batch_size"],
        collate_fn=collate_wrapper,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        ds["test"],
        batch_size=config["batch_size"],
        collate_fn=collate_wrapper,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, ds
