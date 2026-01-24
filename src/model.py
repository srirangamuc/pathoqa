import torch
import torch.nn as nn
from transformers import (
    T5ForConditionalGeneration,
    CLIPVisionModel,
)
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model

class FusionOutput(BaseModelOutput):
    pass

class PathVQA_Fusion_Model(nn.Module):
    def __init__(self, t5_path, clip_path):
        super().__init__()

        # A. Vision Encoder (PathGen-CLIP with LoRA)
        print("👁️ Loading Vision Encoder...")
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_path)

        # Apply LoRA to handle "Textbook Domain Gap"
        peft_config = LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none"
        )
        self.vision_encoder = get_peft_model(self.vision_encoder, peft_config)
        self.vision_encoder.print_trainable_parameters()

        # B. The Bridge (Transformer Fusion)
        print("🌉 Building Transformer Bridge...")
        self.vis_project = nn.Linear(1024, 768) # 1024 (ViT-L) -> 768 (T5)

        # 3-Layer Cross-Attention
        fusion_layer = nn.TransformerDecoderLayer(d_model=768, nhead=8, batch_first=True)
        self.fusion_module = nn.TransformerDecoder(fusion_layer, num_layers=3)

        # C. The Brain (T5)
        print("🧠 Loading T5 Brain...")
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_path)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # 1. Vision Features
        vision_out = self.vision_encoder(pixel_values).last_hidden_state
        vision_embeds = self.vis_project(vision_out)

        # 2. Text Features
        text_embeds = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # 3. Fusion (Text queries Image)
        fused_embeds = self.fusion_module(tgt=text_embeds, memory=vision_embeds)

        # 4. Generate Answer
        outputs = self.t5(
            encoder_outputs=(fused_embeds,), # Bypass T5 Encoder
            labels=labels
        )
        return outputs
