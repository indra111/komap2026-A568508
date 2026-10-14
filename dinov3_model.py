"""DINOv3 Base, standard stride16 DPT and original-resolution detail, CUDA BF16."""
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from pretrained_model import MEAN, STD

ARCHITECTURE = 'dpt_dinov3_base_detail_bf16'


class DinoV3DPT(torch.nn.Module):
    def __init__(self, architecture=ARCHITECTURE):
        super().__init__()
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        self.architecture = architecture
        self.net = smp.DPT(encoder_name='tu-vit_base_patch16_dinov3.lvd1689m', encoder_weights=None,
            classes=4, decoder_intermediate_channels=(64, 128, 256, 256),
            decoder_fusion_channels=128, dynamic_img_size=True)
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))
        self.detail = torch.nn.Sequential(torch.nn.Conv2d(7, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(32, 4, 1))
        torch.nn.init.zeros_(self.detail[-1].weight)
        torch.nn.init.zeros_(self.detail[-1].bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            normalized = (x - self.input_mean) / self.input_std
            # Even patch16 token grid; no resize of the observed microstructure.
            padded = F.pad(normalized, (0, (-w) % 32, 0, (-h) % 32), mode='replicate')
            logits = self.net(padded)
            if logits.shape[-2:] != padded.shape[-2:]:
                raise RuntimeError('Stride16 DPT output geometry differs from its input')
            logits = logits[..., :h, :w].float()
            return logits + self.detail(torch.cat([normalized, logits.softmax(1)], 1)).float()


def load_dinov3(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = DinoV3DPT(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
