"""DINOv2 small features with a DPT decoder and optional native-pixel detail."""
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.dpt.decoder import DPTDecoder
from safetensors.torch import load_file
from core import file_hash
from pretrained_model import MEAN, STD

ARCHITECTURES = ('dpt_dinov2_small', 'dpt_dinov2_small_detail')


class DinoDPT(torch.nn.Module):
    def __init__(self, architecture='dpt_dinov2_small'):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError('Unknown DINO architecture')
        self.architecture = architecture
        self.net = smp.DPT(encoder_name='tu-vit_small_patch14_dinov2.lvd142m', encoder_weights=None,
            classes=4, decoder_intermediate_channels=(64, 128, 256, 256), decoder_fusion_channels=128,
            dynamic_img_size=True)
        # SMP 0.5 truncates fractional14/4,14/8 reassembly strides. Build the
        # standard [4,2,1,1/2] token-grid pyramid explicitly, then map its
        # nominal stride16 output back to the real stride14 image in forward.
        self.net.decoder = DPTDecoder(encoder_out_channels=self.net.encoder.out_channels,
            encoder_output_strides=(16, 16, 16, 16), encoder_has_prefix_tokens=True,
            readout='cat', intermediate_channels=(64, 128, 256, 256), fusion_channels=128)
        self.net.initialize()
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))
        if architecture.endswith('_detail'):
            self.detail = torch.nn.Sequential(torch.nn.Conv2d(7, 32, 3, padding=1), torch.nn.GELU(),
                torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(32, 4, 1))
            torch.nn.init.zeros_(self.detail[-1].weight)
            torch.nn.init.zeros_(self.detail[-1].bias)

    def forward(self, x):
        h, w = x.shape[-2:]
        normalized = (x - self.input_mean) / self.input_std
        # Patch14 features need an even grid for DPT's 2x fusion. Pad, never
        # distort the image; restore DPT's nominal stride16 output to stride14.
        padded = F.pad(normalized, (0, (-w) % 28, 0, (-h) % 28), mode='replicate')
        logits = self.net(padded)
        logits = F.interpolate(logits, size=padded.shape[-2:], mode='bilinear', align_corners=False)[..., :h, :w]
        if hasattr(self, 'detail'):
            logits = logits + self.detail(torch.cat([normalized, logits.float().softmax(1)], dim=1))
        return logits


def initialize_dino(model, path, expected_sha256):
    if file_hash(path) != expected_sha256:
        raise ValueError('Pretrained SHA256 differs')
    source = load_file(str(path), device='cpu')
    expected = model.net.encoder.state_dict()
    subset = {}
    used = set()
    for key, value in expected.items():
        original = key.removeprefix('model.')
        if original not in source or source[original].shape != value.shape or source[original].dtype != value.dtype:
            raise ValueError('DINO feature mismatch: ' + key)
        subset[key] = source[original]
        used.add(original)
    extra = set(source) - used
    # SMP uses intermediate block features; its wrapper may prune final norm.
    if extra not in [set(), {'norm.weight', 'norm.bias'}]:
        raise ValueError('Unexpected unused DINO tensors: ' + str(sorted(extra)))
    model.net.encoder.load_state_dict(subset, strict=True)
    assert all(torch.equal(v, subset[k]) for k, v in model.net.encoder.state_dict().items())
    return {'sha256': expected_sha256, 'feature_tensors_loaded': len(subset),
            'excluded_final_norm': sorted(extra), 'feature_tensors_identical': True, 'offline_loading': True}


def build_dino_optimizer(model):
    groups = [{'params': model.net.encoder.parameters(), 'lr': 2e-5},
              {'params': model.net.decoder.parameters(), 'lr': 1e-3},
              {'params': model.net.segmentation_head.parameters(), 'lr': 1e-3}]
    if hasattr(model, 'detail'):
        groups.append({'params': model.detail.parameters(), 'lr': 1e-3})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def load_dino(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = DinoDPT(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
