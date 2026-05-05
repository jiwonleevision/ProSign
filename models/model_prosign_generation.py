from models.model_prosign_generation_base import ProSign_generation as BaseProSignGeneration


T5_MODEL_ID = "google-t5/t5-large"


class ProSign_generation(BaseProSignGeneration):
    def __init__(self, inplanes=768, planes=1024, pretrain=None):
        super().__init__(
            t5_model_id=T5_MODEL_ID,
            spatial_depth=4,
            temporal_depth=4,
        )
