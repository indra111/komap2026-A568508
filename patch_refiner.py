"""Frozen Base segmenter plus a trainable four-class local zoom residual.

Inspired by BPR's image-plus-mask local refinement, not a reproduction of its
binary instance model. All inference tiles are refined, including confident
regions; only training patch sampling varies between uniform and boundaries.
"""
import torch
from torch import nn
import torch.nn.functional as F

from large_dino_model import BaseDinoDPT, ARCHITECTURE as PARENT_ARCHITECTURE
from pretrained_model import tensor_fingerprint

ARCHITECTURE = 'frozen_dinov2_base_patch_refiner_bf16'
PATCH, STRIDE, ZOOM, TRAIN_PATCHES, CHUNK = 64, 32, 2, 8, 16
METHODS = ('uniform', 'boundary')


def starts(length, patch=PATCH, stride=STRIDE):
    if length < patch or not 0 < stride <= patch:
        raise ValueError('Invalid patch geometry')
    return list(dict.fromkeys([*range(0, length-patch+1, stride), length-patch]))


def predicted_boundary(labels):
    if labels.ndim != 3:
        raise ValueError('Expected BxHxW predicted labels')
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    vertical = labels[:, 1:] != labels[:, :-1]
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    boundary[:, 1:] |= vertical; boundary[:, :-1] |= vertical
    boundary[:, :, 1:] |= horizontal; boundary[:, :, :-1] |= horizontal
    return boundary


def sample_boxes(logits, method, count=TRAIN_PATCHES):
    if method not in METHODS or count < 1 or logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError('Invalid training patch request')
    b, _, h, w = logits.shape
    if min(h, w) < PATCH:
        raise ValueError('Training context must contain a native patch')
    boundary = predicted_boundary(logits.detach().argmax(1)).cpu()
    boxes = []
    for index in range(b):
        points = boundary[index].nonzero() if method == 'boundary' else None
        for _ in range(count):
            if points is not None and len(points):
                point = points[int(torch.randint(len(points), (1,)))]
                y = max(0, min(h-PATCH, int(point[0])-PATCH//2))
                x = max(0, min(w-PATCH, int(point[1])-PATCH//2))
            else:
                y = int(torch.randint(h-PATCH+1, (1,)))
                x = int(torch.randint(w-PATCH+1, (1,)))
            boxes.append((index, y, x))
    return boxes


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(4, cout), nn.GELU())


class LocalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc0 = block(7, 16)
        self.enc1 = block(16, 32)
        self.enc2 = block(32, 64)
        self.dec1 = block(96, 32)
        self.dec0 = block(48, 16)
        self.out = nn.Conv2d(16, 4, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, features):
        a = self.enc0(features)
        b = self.enc1(F.avg_pool2d(a, 2))
        c = self.enc2(F.avg_pool2d(b, 2))
        b = self.dec1(torch.cat([b, F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False)], 1))
        a = self.dec0(torch.cat([a, F.interpolate(b, size=a.shape[-2:], mode='bilinear', align_corners=False)], 1))
        return self.out(a)


class PatchRefiner(nn.Module):
    def __init__(self, architecture=ARCHITECTURE):
        super().__init__()
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        self.architecture = architecture
        self.base = BaseDinoDPT()
        self.base.requires_grad_(False).eval()
        self.refiner = LocalResidual()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def correction(self, images, coarse):
        if images.shape[0] != coarse.shape[0] or images.shape[-2:] != (PATCH, PATCH) or coarse.shape[1:] != (4, PATCH, PATCH):
            raise ValueError('Expected aligned native RGB and four-logit patches')
        normalized = (images-self.base.input_mean) / self.base.input_std
        features = torch.cat([normalized, coarse.float().softmax(1)], 1)
        features = F.interpolate(features, scale_factor=ZOOM, mode='bilinear', align_corners=False)
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.device.type == 'cuda'):
            delta = self.refiner(features)
        return F.interpolate(delta.float(), size=(PATCH, PATCH), mode='bilinear', align_corners=False)

    def training_patches(self, images, labels, method):
        if labels.shape != (images.shape[0], *images.shape[-2:]):
            raise ValueError('Targets must match context images')
        self.base.eval()
        with torch.no_grad():
            coarse = self.base(images)
        boxes = sample_boxes(coarse, method)
        image_patches = torch.stack([images[b, :, y:y+PATCH, x:x+PATCH] for b,y,x in boxes])
        coarse_patches = torch.stack([coarse[b, :, y:y+PATCH, x:x+PATCH] for b,y,x in boxes])
        targets = torch.stack([labels[b, y:y+PATCH, x:x+PATCH] for b,y,x in boxes])
        output = coarse_patches.float() + self.correction(image_patches, coarse_patches)
        return output, targets, boxes

    def forward(self, images):
        # Common native-resolution inference: parent context tiling remains 512;
        # this inner tiling refines local details inside each context tile.
        self.base.eval()
        with torch.no_grad():
            coarse = self.base(images)
        h, w = images.shape[-2:]
        padding = (0, max(0, PATCH-w), 0, max(0, PATCH-h))
        padded = F.pad(images, padding, mode='replicate')
        coarse_pad = F.pad(coarse, padding, mode='replicate')
        boxes = [(b,y,x) for b in range(images.shape[0])
                 for y in starts(padded.shape[-2]) for x in starts(padded.shape[-1])]
        summed = torch.zeros_like(coarse_pad, dtype=torch.float32)
        count = torch.zeros_like(coarse_pad[:, :1], dtype=torch.float32)
        for first in range(0, len(boxes), CHUNK):
            part = boxes[first:first+CHUNK]
            image_patches = torch.stack([padded[b,:,y:y+PATCH,x:x+PATCH] for b,y,x in part])
            coarse_patches = torch.stack([coarse_pad[b,:,y:y+PATCH,x:x+PATCH] for b,y,x in part])
            delta = self.correction(image_patches, coarse_patches)
            for patch, (b,y,x) in zip(delta, part):
                summed[b,:,y:y+PATCH,x:x+PATCH] += patch
                count[b,:,y:y+PATCH,x:x+PATCH] += 1
        if (count == 0).any():
            raise RuntimeError('Uncovered inference pixels')
        return coarse.float() + (summed/count)[..., :h, :w]


def initialize_parent(model, saved):
    if saved['architecture'] != PARENT_ARCHITECTURE:
        raise ValueError('Expected trained Base parent')
    model.base.load_state_dict(saved['model'], strict=True)
    fingerprint = tensor_fingerprint(saved['model'])
    if tensor_fingerprint(model.base.state_dict()) != fingerprint:
        raise ValueError('Frozen parent transfer differs')
    assert not any(p.requires_grad for p in model.base.parameters())
    assert all(torch.count_nonzero(p) == 0 for p in model.refiner.out.parameters())
    return {'frozen_parent_fingerprint': fingerprint, 'all_parent_tensors_identical': True,
            'zero_initial_correction': True, 'parent_eval_during_training': True,
            'extra_trainable_parameters': sum(p.numel() for p in model.refiner.parameters())}


def build_optimizer(model, lr):
    parameters = list(model.refiner.parameters())
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)


def load_patch_refiner(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = PatchRefiner(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
