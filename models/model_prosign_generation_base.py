import os
import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from models.model_prosign_pretrain_base import SpatioTemporalTransformer


def use_local_t5_files_only():
    """Returns whether the current process should rely on locally cached T5 weights only."""

    return os.environ.get("MMSLT_T5_LOCAL_FILES_ONLY", "0") == "1"


class ProSign_generation(nn.Module):
    """Shared implementation for ProSign_generation."""

    def __init__(
        self,
        t5_model_id='google-t5/t5-large',
        input_dim=6,
        encoder_embed_dim=1024,
        num_heads=8,
        spatial_depth=4,
        temporal_depth=4,
        num_joints=75,
        max_time=32,
    ):
        super().__init__()
        self.t5_model_id = t5_model_id
        self.t5 = T5ForConditionalGeneration.from_pretrained(
            t5_model_id,
            local_files_only=use_local_t5_files_only(),
        )
        self.t5 = get_peft_model(
            self.t5,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "k", "v", "o"],
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ_2_SEQ_LM",
            ),
        )

        self.model_image = SpatioTemporalTransformer(
            input_dim=input_dim,
            embed_dim=encoder_embed_dim,
            num_heads=num_heads,
            spatial_depth=spatial_depth,
            temporal_depth=temporal_depth,
            num_joints=num_joints,
            max_time=max_time,
        )
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
            if "encoder." in name:
                param.requires_grad = False

    def share_forward(self, src_input):
        device = next(self.parameters()).device
        input_kps = src_input["input_kps"].to(device, non_blocking=True)
        attention_mask = src_input["attention_mask"].to(device, non_blocking=True)

        visual_tokens = self.model_image(
            {
                **src_input,
                "input_kps": input_kps,
                "attention_mask": attention_mask,
            }
        )

        padding_mask = attention_mask.bool()
        valid_mask = (~padding_mask).unsqueeze(-1).unsqueeze(-1).float()
        visual_tokens = visual_tokens * valid_mask

        batch_size, num_frames, num_joints, channels = visual_tokens.shape
        visual_tokens = visual_tokens.reshape(batch_size, num_frames, num_joints * channels)
        visual_tokens = self.encoder_proj(visual_tokens)
        encoder_attention_mask = (~padding_mask).long()
        return visual_tokens, encoder_attention_mask

    def forward(self, src_input, tgt_input):
        inputs_embeds, attention_mask = self.share_forward(src_input)
        labels = tgt_input["input_ids"].clone().to(inputs_embeds.device)
        labels[tgt_input["attention_mask"].to(inputs_embeds.device) == 0] = -100

        outputs = self.t5(
            encoder_outputs=BaseModelOutput(last_hidden_state=inputs_embeds),
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=tgt_input["attention_mask"].to(inputs_embeds.device),
            return_dict=True,
        )
        return outputs["logits"]

    def generate(self, src_input, max_new_tokens, num_beams, num_return_sequences):
        inputs_embeds, attention_mask = self.share_forward(src_input)
        return self.t5.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=inputs_embeds),
            attention_mask=attention_mask.to(inputs_embeds.device),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
