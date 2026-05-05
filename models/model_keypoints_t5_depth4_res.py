import os
import torch
import torchvision
import torch.nn.functional as F

from utils.definition import *
from hpman.m import _
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from peft import LoraConfig, get_peft_model

import numpy as np

import torch
import torch.nn as nn
from typing import Optional


import torch
import torch.nn as nn


import torch
import torch.nn as nn

# ---------------------------
# MLP
# ---------------------------
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------
# Transformer Block (mask 지원)
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, N, D)
        key_padding_mask: (B, N)  (True = padding)
        """

        x_norm = self.norm1(x)

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))

        return x


# ---------------------------
# Spatio-Temporal Transformer
# ---------------------------
class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=1024,
        num_heads=8,
        spatial_depth=4,     # 6 -> 4
        temporal_depth=4,    # 6 -> 4
        num_joints=75,
        max_time=32,
        residual_scale=0.3,  # stack-level residual scale
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.max_time = max_time
        self.residual_scale = residual_scale

        # input projection
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # positional embeddings
        self.joint_pos_embed = nn.Parameter(
            torch.randn(1, 1, num_joints, embed_dim) * 0.02
        )
        self.time_pos_embed = nn.Parameter(
            torch.randn(1, max_time, 1, embed_dim) * 0.02
        )

        # spatial transformer
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(spatial_depth)
        ])

        # temporal transformer
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(temporal_depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src_input):
        """
        src_input:
            input_kps: (B, T, V, C)
            attention_mask: (B, T)  True = padding
        """

        x = src_input["input_kps"]                 # (B, T, V, C)
        padding_mask = src_input["attention_mask"] # (B, T)

        B, T, V, C = x.shape

        # ---------------------------
        # 1. Input Projection
        # ---------------------------
        x = self.input_proj(x)  # (B, T, V, D)

        # ---------------------------
        # 2. Spatial Pos Embedding
        # ---------------------------
        x = x + self.joint_pos_embed[:, :, :V, :]

        # ---------------------------
        # 3. Spatial Transformer
        # (joint-wise attention)
        # ---------------------------
        x = x.reshape(B * T, V, self.embed_dim)

        spatial_res = x
        for blk in self.spatial_blocks:
            x = blk(x)
        x = x + self.residual_scale * spatial_res

        x = x.reshape(B, T, V, self.embed_dim)

        # ---------------------------
        # 4. Temporal Pos Embedding
        # ---------------------------
        x = x + self.time_pos_embed[:, :T, :, :]

        # ---------------------------
        # 5. Temporal Transformer
        # (time-wise attention)
        # ---------------------------
        x = x.permute(0, 2, 1, 3)   # (B, V, T, D)
        x = x.reshape(B * V, T, self.embed_dim)

        temporal_mask = padding_mask.unsqueeze(1).expand(-1, V, -1)
        temporal_mask = temporal_mask.reshape(B * V, T)

        temporal_res = x
        for blk in self.temporal_blocks:
            x = blk(x, key_padding_mask=temporal_mask)
        x = x + self.residual_scale * temporal_res

        # ---------------------------
        # 6. reshape back
        # ---------------------------
        x = x.reshape(B, V, T, self.embed_dim)
        x = x.permute(0, 2, 1, 3)   # (B, T, V, D)

        # ---------------------------
        # 7. final norm
        # ---------------------------
        x = self.norm(x)

        return x  # (B, T, V, D)

class Projector(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Projector, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class MMLP(torch.nn.Module):
    def __init__(self, embed_dim=1024):
        super(MMLP, self).__init__()
        self.model_image = SpatioTemporalTransformer(input_dim=6, embed_dim=1024, num_heads=8, spatial_depth=4, temporal_depth=4, num_joints=75, max_time=32)
        self.model_image_projection = Projector(input_dim=1024, hidden_dim=embed_dim, output_dim=embed_dim)
        self.model_text_projection = Projector(input_dim=1024, hidden_dim=embed_dim, output_dim=embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        # self.attn_weight = nn.Parameter(torch.randn(1024, 1))

    def forward(self, src_input, tgt_input):
        # tgt_input: (B, text_feat_dim) precomputed T5 feature
        text_features = tgt_input.to(src_input["input_kps"].device).float()

        image_features = self.model_image(src_input)  # (B, T, J, D)

        padding_mask = src_input['attention_mask']
        valid_mask = (~padding_mask).float()

        image_features = image_features.mean(dim=2)   # (B, T, D)

        mask_expanded = valid_mask.unsqueeze(-1)
        sum_features = torch.sum(image_features * mask_expanded, dim=1)
        num_valid_frames = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        image_features = sum_features / num_valid_frames

        image_features = self.model_image_projection(image_features)
        text_features = self.model_text_projection(text_features)

        norm_images = F.normalize(image_features, p=2, dim=-1)
        norm_text = F.normalize(text_features, p=2, dim=-1)

        return norm_text, norm_images

class MZSSLR(torch.nn.Module):
    def __init__(self, inplanes=768, planes=1024, pretrain=None,):
        super(MZSSLR, self).__init__()
        
        model_id = "google-t5/t5-large"
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            model_id,
            local_files_only=os.environ.get("MMSLT_T5_LOCAL_FILES_ONLY", "0") == "1",
        )
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q","k","v","o"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_2_SEQ_LM"
        )
        self.t5 = get_peft_model(self.t5, lora_config)

        self.model_image = SpatioTemporalTransformer(input_dim=6, embed_dim=1024, num_heads=8, spatial_depth=4, temporal_depth=4, num_joints=75, max_time=32)
        self.encoder_proj = nn.Sequential(
            nn.Linear(self.model_image.num_joints * self.model_image.embed_dim, 2048),
            nn.GELU(),
            nn.LayerNorm(2048),
            nn.Dropout(0.1),
            nn.Linear(2048, self.t5.config.d_model),
            nn.LayerNorm(self.t5.config.d_model),
            nn.Dropout(0.1),
        )
        
        for name, param in self.t5.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False

        # encoder 쪽 LoRA도 혹시 포함되었으면 다시 freeze
        for name, param in self.t5.named_parameters():
            if "encoder." in name:
                param.requires_grad = False

        print("=" * 80)
        for name, param in self.t5.named_parameters():
            if param.requires_grad:
                print(name)
        print("=" * 80)

    def share_forward(self, src_input):
        device = next(self.parameters()).device

        input_kps = src_input["input_kps"].to(device, non_blocking=True)
        attention_mask = src_input["attention_mask"].to(device, non_blocking=True)

        src_input = {
            **src_input,
            "input_kps": input_kps,
            "attention_mask": attention_mask,
        }

        # (B, T, J, C)
        visual_tokens = self.model_image(src_input)

        padding_mask = src_input["attention_mask"].bool()  # True = padding
        valid_mask = (~padding_mask).unsqueeze(-1).unsqueeze(-1).float()

        # padded frame zero-out
        visual_tokens = visual_tokens * valid_mask

        B, T, J, C = visual_tokens.shape

        # (B, T, J, C) -> (B, T, J*C)
        visual_tokens = visual_tokens.reshape(B, T, J * C)

        # (B, T, J*C) -> (B, T, H)
        visual_tokens = self.encoder_proj(visual_tokens)

        # (B, T), 1 = valid, 0 = padding
        encoder_attention_mask = (~padding_mask).long()

        return visual_tokens, encoder_attention_mask

    def forward(self, src_input, tgt_input):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)
        encoder_outputs = BaseModelOutput(last_hidden_state=inputs_embeds)

        labels = tgt_input["input_ids"].clone().to(inputs_embeds.device)
        labels[tgt_input["attention_mask"].to(inputs_embeds.device) == 0] = -100
                
        out = self.t5(encoder_outputs = encoder_outputs,
                    attention_mask = attention_mask,
                    # decoder_input_ids = tgt_input['input_ids'].cuda(),
                    labels = labels,
                    decoder_attention_mask = tgt_input['attention_mask'].to(inputs_embeds.device),
                    return_dict = True,
                    )

        return out['logits']

    def generate(self,src_input,max_new_tokens,num_beams,num_return_sequences):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)
        encoder_outputs = BaseModelOutput(last_hidden_state=inputs_embeds)

        out = self.t5.generate(encoder_outputs = encoder_outputs,
                    attention_mask=attention_mask.to(inputs_embeds.device),
                    max_new_tokens=max_new_tokens,
                    num_beams = num_beams,
                    num_return_sequences=num_return_sequences,
                    )
        return out
