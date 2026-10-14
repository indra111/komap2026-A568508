"""Native-pixel dilated texture residual; preserve the existing DINO parent at initialization."""
import torch

from dino_model import build_dino_optimizer
from pretrained_model import tensor_fingerprint
from stable_dino_model import StableDinoDPT, ARCHITECTURE as PARENT_ARCHITECTURE

ARCHITECTURE = 'dpt_dinov2_small_texture_bf16'
DILATIONS = (1, 2, 4, 8)


class ParallelTexture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Conv2d(7, 16, 3, padding=d, dilation=d), torch.nn.GELU())
            for d in DILATIONS])
        self.fuse = torch.nn.Sequential(torch.nn.Conv2d(64, 32, 1), torch.nn.GELU(), torch.nn.Conv2d(32, 4, 1))
        torch.nn.init.zeros_(self.fuse[-1].weight)
        torch.nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x):
        return self.fuse(torch.cat([branch(x) for branch in self.branches], 1))


class TextureDinoDPT(StableDinoDPT):
    def __init__(self, architecture=ARCHITECTURE):
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        super().__init__()
        self.architecture = architecture
        self.texture = ParallelTexture()

    def forward(self, x):
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            logits = super().forward(x)
            normalized = (x - self.input_mean) / self.input_std
            inputs = torch.cat([normalized, logits.float().softmax(1)], 1)
            return logits.float() + self.texture(inputs).float()


def initialize_parent(model, source):
    if source['architecture'] != PARENT_ARCHITECTURE:
        raise ValueError('Expected the stable Small detail parent')
    texture = hasattr(model, 'texture')
    missing, unexpected = model.load_state_dict(source['model'], strict=not texture)
    if unexpected or (texture and set(missing) != {k for k in model.state_dict() if k.startswith('texture.')}):
        raise ValueError('Unexpected missing or extra parent weights')
    inherited = {k: v for k, v in model.state_dict().items() if not k.startswith('texture.')}
    fingerprint = tensor_fingerprint(inherited)
    if fingerprint != tensor_fingerprint(source['model']):
        raise ValueError('Parent tensors were not copied exactly')
    if texture and any(torch.count_nonzero(p) for p in model.texture.fuse[-1].parameters()):
        raise ValueError('Texture correction must initially be zero')
    return {'inherited_parent_fingerprint': fingerprint, 'all_parent_tensors_identical': True,
            'zero_initial_correction': texture, 'extra_trainable_parameters': sum(p.numel() for p in model.texture.parameters()) if texture else 0}


def build_texture_optimizer(model):
    optimizer = build_dino_optimizer(model)
    if hasattr(model, 'texture'):
        optimizer.add_param_group({'params': model.texture.parameters(), 'lr': 1e-4})
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Optimizer must cover each trainable parameter exactly once')
    return optimizer


def load_texture_dino(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = TextureDinoDPT(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
