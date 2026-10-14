"""Native four-class residual over fixed full-image teacher probabilities.

The teacher is external and frozen. Its probabilities are inputs, not parameters
of this head; inference must include the separately recorded teacher recipe.
"""
import torch
from torch import nn
import torch.nn.functional as F

from patch_refiner import LocalResidual, starts, PATCH, STRIDE, ZOOM, CHUNK

ARCHITECTURE = 'cached_four_class_patch_residual_bf16'


class CachedRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.refiner = LocalResidual()
        self.register_buffer('input_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def correction(self, images, probabilities):
        if images.ndim != 4 or images.shape[1:] != (3, PATCH, PATCH):
            raise ValueError('Expected aligned native64 RGB/probability patches')
        if probabilities.shape != (images.shape[0], 4, PATCH, PATCH):
            raise ValueError('Probability coordinates differ')
        features = torch.cat([(images-self.input_mean)/self.input_std, probabilities.float()], 1)
        features = F.interpolate(features, scale_factor=ZOOM, mode='bilinear', align_corners=False)
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.device.type == 'cuda'):
            delta = self.refiner(features)
        return F.interpolate(delta.float(), size=(PATCH, PATCH), mode='bilinear', align_corners=False)

    def training_logits(self, images, probabilities):
        return probabilities.float().clamp_min(1e-7).log() + self.correction(images, probabilities)

    def forward(self, images, probabilities, strength=1.):
        if images.ndim != 4 or images.shape[1] != 3 or probabilities.shape != (images.shape[0], 4, *images.shape[-2:]):
            raise ValueError('Expected aligned full-resolution RGB and four-class probabilities')
        if strength not in (.5, 1.):
            raise ValueError('Unreviewed residual strength')
        h, w = images.shape[-2:]
        padding = (0, max(0, PATCH-w), 0, max(0, PATCH-h))
        image_pad = F.pad(images, padding, mode='replicate')
        prob_pad = F.pad(probabilities.float(), padding, mode='replicate')
        summed = torch.zeros_like(prob_pad)
        count = torch.zeros_like(prob_pad[:, :1])
        boxes = [(b, y, x) for b in range(images.shape[0])
                 for y in starts(image_pad.shape[-2]) for x in starts(image_pad.shape[-1])]
        for first in range(0, len(boxes), CHUNK):
            part = boxes[first:first+CHUNK]
            x = torch.stack([image_pad[b, :, y:y+PATCH, z:z+PATCH] for b, y, z in part])
            p = torch.stack([prob_pad[b, :, y:y+PATCH, z:z+PATCH] for b, y, z in part])
            delta = self.correction(x, p)
            for change, (b, y, z) in zip(delta, part):
                summed[b, :, y:y+PATCH, z:z+PATCH] += change
                count[b, :, y:y+PATCH, z:z+PATCH] += 1
        if (count == 0).any():
            raise RuntimeError('Uncovered native pixels')
        delta = (summed/count)[..., :h, :w]
        return (probabilities.float().clamp_min(1e-7).log() + strength*delta).softmax(1)


def load_refiner(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCHITECTURE:
        raise ValueError('Wrong residual architecture')
    model = CachedRefiner()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
