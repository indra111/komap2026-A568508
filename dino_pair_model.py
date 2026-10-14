"""Conditional Al3Ni/Eutectic correction preserving matrix/Primary probabilities and total pair mass."""
import torch

from dino_texture_model import ParallelTexture
from stable_dino_model import StableDinoDPT

ARCHITECTURE = 'dpt_dinov2_small_pair_bf16'


def redistribute_pair(logits, correction):
    if logits.ndim != 4 or logits.shape[1] != 4 or correction.shape != logits[:, :1].shape:
        raise ValueError('Expected four logits and one aligned correction channel')
    logits, correction = logits.float(), correction.float()
    original_pair = logits[:, 2:4]
    adjusted_pair = torch.cat([logits[:, 2:3], logits[:, 3:4] + correction], 1)
    shift = torch.logsumexp(adjusted_pair, 1, keepdim=True) - torch.logsumexp(original_pair, 1, keepdim=True)
    # At correction==0, identical reductions subtract to exactly zero.
    # Preserving p0,p1 does not freeze argmax labels: redistributing p2+p3 can
    # cause a pair member to overtake a previously leading non-pair class.
    return torch.cat([logits[:, :2], adjusted_pair - shift], 1)


class PairDinoDPT(StableDinoDPT):
    def __init__(self, architecture=ARCHITECTURE):
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        super().__init__()
        self.architecture = architecture
        self.texture = ParallelTexture()
        self.texture.fuse[-1] = torch.nn.Conv2d(32, 1, 1)
        torch.nn.init.zeros_(self.texture.fuse[-1].weight)
        torch.nn.init.zeros_(self.texture.fuse[-1].bias)

    def forward(self, x):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            logits = super().forward(x)
            normalized = (x - self.input_mean) / self.input_std
            features = torch.cat([normalized, logits.float().softmax(1)], 1)
            return redistribute_pair(logits, self.texture(features))


def load_pair_dino(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = PairDinoDPT(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
