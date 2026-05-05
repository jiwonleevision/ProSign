import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional


class MLP(nn.Module):
    """Feed-forward block used inside the transformer encoder."""

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


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm self-attention and MLP layers."""

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
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class SpatioTemporalTransformer(nn.Module):
    """Applies spatial and temporal transformer blocks over keypoint sequences."""

    def __init__(
        self,
        input_dim,
        embed_dim=1024,
        num_heads=8,
        spatial_depth=4,
        temporal_depth=4,
        num_joints=75,
        max_time=32,
        residual_scale=0.3,
        use_spatial=True,
        use_temporal=True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.max_time = max_time
        self.residual_scale = residual_scale
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.joint_pos_embed = nn.Parameter(
            torch.randn(1, 1, num_joints, embed_dim) * 0.02
        )
        if use_temporal:
            self.time_pos_embed = nn.Parameter(
                torch.randn(1, max_time, 1, embed_dim) * 0.02
            )
        else:
            self.register_parameter("time_pos_embed", None)

        self.spatial_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(spatial_depth)]
        ) if use_spatial else nn.ModuleList()
        self.temporal_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads) for _ in range(temporal_depth)]
        ) if use_temporal else nn.ModuleList()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, src_input):
        x = src_input["input_kps"]
        padding_mask = src_input["attention_mask"].bool()

        batch_size, num_frames, num_joints, _ = x.shape

        x = self.input_proj(x)
        x = x + self.joint_pos_embed[:, :, :num_joints, :]

        if self.use_spatial:
            x = x.reshape(batch_size * num_frames, num_joints, self.embed_dim)
            spatial_residual = x
            for block in self.spatial_blocks:
                x = block(x)
            x = x + self.residual_scale * spatial_residual
            x = x.reshape(batch_size, num_frames, num_joints, self.embed_dim)

        if self.use_temporal:
            x = x + self.time_pos_embed[:, :num_frames, :, :]
            x = x.permute(0, 2, 1, 3).reshape(
                batch_size * num_joints, num_frames, self.embed_dim
            )

            temporal_mask = padding_mask.unsqueeze(1).expand(-1, num_joints, -1)
            temporal_mask = temporal_mask.reshape(batch_size * num_joints, num_frames)

            temporal_residual = x
            for block in self.temporal_blocks:
                x = block(x, key_padding_mask=temporal_mask)
            x = x + self.residual_scale * temporal_residual

            x = x.reshape(batch_size, num_joints, num_frames, self.embed_dim)
            x = x.permute(0, 2, 1, 3)

        x = self.norm(x)
        return x


class LinearProjector(nn.Module):
    """Applies a single linear projection."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.encoder(x)


class NonLinearProjector(nn.Module):
    """Applies a two-layer non-linear projection."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)


def build_projector(projector_type, input_dim, output_dim, hidden_dim=None):
    """Builds the projector used for image and text features."""

    if projector_type == "linear":
        return LinearProjector(input_dim=input_dim, output_dim=output_dim)
    if projector_type == "nonlinear":
        hidden_dim = hidden_dim if hidden_dim is not None else output_dim
        return NonLinearProjector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
    raise ValueError(f"Unsupported projector_type: {projector_type}")


@dataclass(frozen=True)
class ProSignPretrainConfig:
    use_spatial: bool = True
    use_temporal: bool = True
    projector_type: str = "nonlinear"
    projector_output_dim: int = 1024
    projector_hidden_dim: Optional[int] = None
    input_dim: int = 6
    encoder_embed_dim: int = 1024
    num_heads: int = 8
    spatial_depth: int = 4
    temporal_depth: int = 4
    num_joints: int = 75
    max_time: int = 32
    residual_scale: float = 0.3


class ProSign_pretrain(nn.Module):
    """Shared implementation for ProSign_pretrain variants."""

    def __init__(self, config: ProSignPretrainConfig):
        super().__init__()
        self.config = config
        self.model_image = SpatioTemporalTransformer(
            input_dim=config.input_dim,
            embed_dim=config.encoder_embed_dim,
            num_heads=config.num_heads,
            spatial_depth=config.spatial_depth,
            temporal_depth=config.temporal_depth,
            num_joints=config.num_joints,
            max_time=config.max_time,
            residual_scale=config.residual_scale,
            use_spatial=config.use_spatial,
            use_temporal=config.use_temporal,
        )
        self.model_image_projection = build_projector(
            projector_type=config.projector_type,
            input_dim=config.encoder_embed_dim,
            output_dim=config.projector_output_dim,
            hidden_dim=config.projector_hidden_dim,
        )
        self.model_text_projection = build_projector(
            projector_type=config.projector_type,
            input_dim=1024,
            output_dim=config.projector_output_dim,
            hidden_dim=config.projector_hidden_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, src_input, tgt_input):
        text_features = tgt_input.to(src_input["input_kps"].device).float()

        image_features = self.model_image(src_input)
        padding_mask = src_input["attention_mask"].bool()
        valid_mask = (~padding_mask).float()

        image_features = image_features.mean(dim=2)
        mask_expanded = valid_mask.unsqueeze(-1)
        summed_features = torch.sum(image_features * mask_expanded, dim=1)
        num_valid_frames = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        image_features = summed_features / num_valid_frames

        image_features = self.model_image_projection(image_features)
        text_features = self.model_text_projection(text_features)

        norm_images = F.normalize(image_features, p=2, dim=-1)
        norm_text = F.normalize(text_features, p=2, dim=-1)
        return norm_text, norm_images
