"""Warm-start normalized local detail with spatially aligned whole-image context.

Uses the already tested coordinate sampler/inference of ContextUNet. The fusion
starts at zero, so a loaded parent's initial local predictions are unchanged.
"""
import torch
import torch.nn.functional as F
from context_model import ContextUNet
from pretrained_model import MEAN, STD, build_optimizer as base_optimizer
from expansion import multiclass_loss
ARCHITECTURE = 'normalized_unet_resnet34_local_global'


class PretrainedContextNet(ContextUNet):
    def __init__(self):
        super().__init__()
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))

    @property
    def net(self):
        return self.local

    def load_parent(self, state):
        self.local.load_state_dict({k.removeprefix('net.'): v for k, v in state.items() if k.startswith('net.')}, strict=True)
        self.input_mean.copy_(state['input_mean'])
        self.input_std.copy_(state['input_std'])

    def encode_context(self, global_image):
        return self.context_encoder(2 * global_image - 1)

    def forward(self, local_image, global_image=None, boxes=None, context_features=None, auxiliary=False):
        return super().forward((local_image - self.input_mean) / self.input_std, global_image,
                               boxes, context_features, auxiliary)


def build_optimizer(model):
    optimizer = base_optimizer(model)
    optimizer.add_param_group({'params': list(model.context_encoder.parameters()) +
        list(model.fusion.parameters()) + list(model.coarse_head.parameters()), 'lr': 1e-3})
    return optimizer


def training_loss(output, target, coarse_target, method):
    logits, coarse = output
    loss = multiclass_loss(logits, target)
    if method == 'context':
        return loss
    if method != 'context_aux':
        raise ValueError(method)
    return loss + .2 * F.cross_entropy(coarse.float(), coarse_target, ignore_index=255)


def load_pretrained_context(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCHITECTURE:
        raise ValueError('Unexpected pretrained context architecture')
    model = PretrainedContextNet()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
