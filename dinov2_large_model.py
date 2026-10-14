"""DINOv2 Large with the tested DPT/native-detail geometry and CUDA BF16."""
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.dpt.decoder import DPTDecoder
from dino_model import DinoDPT
from pretrained_model import MEAN, STD

ARCHITECTURE = 'dpt_dinov2_large_detail_bf16'


class LargeDinoDPT(DinoDPT):
    def __init__(self, architecture=ARCHITECTURE):
        torch.nn.Module.__init__(self)
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        self.architecture = architecture
        self.net = smp.DPT(encoder_name='tu-vit_large_patch14_dinov2.lvd142m', encoder_weights=None,
            encoder_output_indices=[5, 11, 17, 23],
            classes=4, decoder_intermediate_channels=(64, 128, 256, 256), decoder_fusion_channels=128,
            dynamic_img_size=True)
        self.net.decoder = DPTDecoder(encoder_out_channels=self.net.encoder.out_channels,
            encoder_output_strides=(16, 16, 16, 16), encoder_has_prefix_tokens=True,
            readout='cat', intermediate_channels=(64, 128, 256, 256), fusion_channels=128)
        self.net.initialize()
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))
        self.detail = torch.nn.Sequential(torch.nn.Conv2d(7, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(32, 4, 1))
        torch.nn.init.zeros_(self.detail[-1].weight)
        torch.nn.init.zeros_(self.detail[-1].bias)

    def forward(self, x):
        # Reuse the preserved patch14/even-grid padding and native-pixel detail.
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            return super().forward(x)


def load_large_dino(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = LargeDinoDPT(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
