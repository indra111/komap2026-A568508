"""Explicit, offline-loadable alternate feature extractors and decoders."""
import re
import torch
import segmentation_models_pytorch as smp
from safetensors.torch import load_file
from core import file_hash
from pretrained_model import MEAN, STD, initialize_encoder

ARCHITECTURES = ('unet_convnext_tiny', 'unetplusplus_resnet34')


class AdvancedUNet(torch.nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if architecture == 'unet_convnext_tiny':
            self.net = smp.Unet(encoder_name='tu-convnext_tiny.fb_in1k', encoder_weights=None, in_channels=3, classes=4)
        elif architecture == 'unetplusplus_resnet34':
            self.net = smp.UnetPlusPlus(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=4)
        else:
            raise ValueError('Unknown architecture: '+architecture)
        self.architecture = architecture
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1,3,1,1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1,3,1,1))

    def forward(self, x):
        return self.net((x-self.input_mean)/self.input_std)


def initialize_advanced(model, path, expected_sha256):
    if model.architecture == 'unetplusplus_resnet34':
        return initialize_encoder(model,path,expected_sha256)
    if file_hash(path) != expected_sha256:
        raise ValueError('Pretrained SHA256 differs')
    source=load_file(str(path),device='cpu')
    expected=model.net.encoder.state_dict()
    subset={}
    used=set()
    for key, value in expected.items():
        original=re.sub(r'^(stem|stages)_(\d+)\.',r'\1.\2.',key.removeprefix('model.'))
        if original not in source or source[original].shape != value.shape or source[original].dtype != value.dtype:
            raise ValueError('Pretrained feature mismatch: '+key+' <- '+original)
        subset[key]=source[original];used.add(original)
    extra=set(source)-used
    if extra != {'head.norm.weight','head.norm.bias','head.fc.weight','head.fc.bias'}:
        raise ValueError('Unexpected non-feature tensors: '+repr(sorted(extra)))
    model.net.encoder.load_state_dict(subset,strict=True)
    if not all(torch.equal(v,subset[k]) for k,v in model.net.encoder.state_dict().items()):
        raise ValueError('Loaded encoder mismatch')
    return {'sha256':expected_sha256,'feature_tensors_loaded':len(subset),'excluded_classification_head':sorted(extra),
            'feature_tensors_identical':True,'offline_loading':True}


def load_advanced(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=AdvancedUNet(saved['architecture'])
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval(),saved
