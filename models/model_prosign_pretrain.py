from models.model_prosign_pretrain_base import ProSign_pretrain as BaseProSignPretrain, ProSignPretrainConfig


class ProSign_pretrain(BaseProSignPretrain):
    def __init__(self):
        super().__init__(
            ProSignPretrainConfig(
                use_spatial=True,
                use_temporal=True,
                spatial_depth=4,
                temporal_depth=4,
                projector_type="nonlinear",
                projector_output_dim=1024,
            )
        )
