"""Pinned DINOv3 Large full fine-tuning with native-resolution DPT output.

The Base implementation and earlier experiments remain unchanged. An adapted
checkpoint contains the full encoder and can be loaded without public weights.
"""
import torch
import segmentation_models_pytorch as smp
from core import file_hash
from dino_model import initialize_dino
from dinov3_model import DinoV3DPT
from pretrained_model import MEAN, STD

ARCHITECTURE = 'dpt_dinov3_large_detail_full_bf16_v1'
PUBLIC_FILE = 'data/pretrained/dinov3-large/model.safetensors'
PUBLIC_SHA = '45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
PUBLIC_REVISION = '30c1109559f65dea34316b0d4842d35c5771fe11'
OUTPUT_INDICES = (5, 11, 17, 23)
# The SMP encoder requests intermediate features without applying final norm.
UNUSED_ENCODER_PARAMETERS = frozenset(('norm.weight', 'norm.bias'))


class DinoV3LargeDPT(DinoV3DPT):
    def __init__(self, weights=None):
        torch.nn.Module.__init__(self)
        self.architecture = ARCHITECTURE
        self.net = smp.DPT(
            encoder_name='tu-vit_large_patch16_dinov3.lvd1689m',
            encoder_weights=None, encoder_output_indices=list(OUTPUT_INDICES),
            classes=4, decoder_intermediate_channels=(64, 128, 256, 256),
            decoder_fusion_channels=128, dynamic_img_size=True)
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))
        self.detail = torch.nn.Sequential(
            torch.nn.Conv2d(7, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 4, 1))
        torch.nn.init.zeros_(self.detail[-1].weight)
        torch.nn.init.zeros_(self.detail[-1].bias)
        encoder = self.net.encoder.model
        assert len(encoder.blocks) == 24 and encoder.num_features == 1024
        assert tuple(encoder.reg_token.shape) == (1, 4, 1024)
        assert tuple(self.net.encoder._output_indices) == OUTPUT_INDICES
        self.initialization_audit = None
        if weights is not None:
            self.initialization_audit = initialize_dino(self, weights, PUBLIC_SHA)
        for name, parameter in encoder.named_parameters():
            parameter.requires_grad_(name not in UNUSED_ENCODER_PARAMETERS)


def build_dinov3_large_optimizer(model):
    groups = [
        {'params': [p for p in model.net.encoder.parameters() if p.requires_grad], 'lr': 2e-5},
        {'params': list(model.net.decoder.parameters()), 'lr': 1e-3},
        {'params': list(model.net.segmentation_head.parameters()), 'lr': 1e-3},
        {'params': list(model.detail.parameters()), 'lr': 1e-3}]
    selected = [id(p) for group in groups for p in group['params']]
    assert len(selected) == len(set(selected))
    assert set(selected) == {id(p) for p in model.parameters() if p.requires_grad}
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def load_dinov3_large(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCHITECTURE or saved['config']['public_sha256'] != PUBLIC_SHA:
        raise ValueError('Wrong DINOv3 Large architecture or initialization')
    model = DinoV3LargeDPT()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
