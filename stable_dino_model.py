"""Numerical-policy variant: BF16 on CUDA in both training and inference."""
import torch
from dino_model import DinoDPT

ARCHITECTURE = 'dpt_dinov2_small_detail_bf16'


class StableDinoDPT(DinoDPT):
    def __init__(self, architecture=ARCHITECTURE):
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        super().__init__('dpt_dinov2_small_detail')
        self.architecture = architecture

    def forward(self, x):
        # Override the shared tiler's FP16 context as well as training context.
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            return super().forward(x)


def load_stable_dino(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = StableDinoDPT(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
