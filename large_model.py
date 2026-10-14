"""Offline ConvNeXt Base/ImageNet22k U-Net; old architecture files stay frozen."""
import torch
import segmentation_models_pytorch as smp
from advanced_model import initialize_advanced
from pretrained_model import MEAN, STD

ARCHITECTURE = 'unet_convnext_base22k'


class LargeUNet(torch.nn.Module):
    def __init__(self, architecture=ARCHITECTURE):
        super().__init__()
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        self.net = smp.Unet(encoder_name='tu-convnext_base.fb_in22k_ft_in1k', encoder_weights=None,
                            in_channels=3, classes=4)
        self.architecture = architecture
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.input_mean) / self.input_std)


def initialize_large(model, path, expected_sha256):
    # The original feature-key mapping is identical; every tensor is checked.
    return initialize_advanced(model, path, expected_sha256)


def load_large(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = LargeUNet(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
