"""Learn contour and boundary distance alongside phase identity.

Inspired by Psi-Net's multi-task principle; not a reproduction or a claim that
human material identification has been fully captured. Auxiliary targets use
training masks only. Inference consumes the same three image channels as before.
"""
import numpy as np
import torch
import torch.nn.functional as F
from pretrained_model import NormalizedUNet, build_optimizer as base_optimizer
from expansion import multiclass_loss
from image_processing import cv2
ARCHITECTURE = 'normalized_unet_resnet34_shape_aux'


class ShapeAuxNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = NormalizedUNet()
        channels = self.base.net.segmentation_head[0].in_channels
        self.geometry = torch.nn.Conv2d(channels, 2, 1)

    @property
    def net(self):
        return self.base.net

    def forward(self, x, auxiliary=False):
        features = self.net.encoder((x - self.base.input_mean) / self.base.input_std)
        decoded = self.net.decoder(features)
        logits = self.net.segmentation_head(decoded)
        return (logits, self.geometry(decoded)) if auxiliary else logits


def build_optimizer(model):
    optimizer = base_optimizer(model)
    optimizer.add_param_group({'params': model.geometry.parameters(), 'lr': 1e-3})
    return optimizer


def geometry_targets(target, cap=16):
    """Distance to any phase boundary, preserving thin structures as labels.

    The cap-wide crop margin is ignored: an unseen boundary outside the crop
    could change its distance target. Image/crop borders are not phase borders.
    """
    boundaries, distances = [], []
    for labels in target.detach().cpu().numpy():
        boundary = np.zeros(labels.shape, dtype=np.uint8)
        diff = labels[1:] != labels[:-1]
        boundary[1:] |= diff
        boundary[:-1] |= diff
        diff = labels[:, 1:] != labels[:, :-1]
        boundary[:, 1:] |= diff
        boundary[:, :-1] |= diff
        if boundary.any():
            distance = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            distance = np.minimum(distance, cap) / cap
        else:
            distance = np.ones(labels.shape, np.float32)
        boundaries.append(boundary)
        distances.append(distance)
    valid = torch.zeros_like(target, dtype=torch.bool)
    if min(target.shape[-2:]) > 2 * cap:
        valid[:, cap:-cap, cap:-cap] = True
    return (torch.as_tensor(np.stack(boundaries), device=target.device).float(),
            torch.as_tensor(np.stack(distances), device=target.device).float(), valid)


def training_loss(output, target, method):
    logits, geometry = output
    base = multiclass_loss(logits, target)
    if method == 'shape_control':
        return base
    if method != 'shape_aux':
        raise ValueError(method)
    boundary, distance, valid = geometry_targets(target)
    edge_loss = F.binary_cross_entropy_with_logits(geometry[:, 0].float(), boundary, reduction='none')
    distance_loss = F.smooth_l1_loss(geometry[:, 1].float().sigmoid(), distance, reduction='none', beta=.1)
    edges, shapes = [], []
    for i in range(target.shape[0]):
        present = []
        for value in [0, 1]:
            select = valid[i] & (boundary[i] == value)
            if select.any():
                present.append(edge_loss[i][select].mean())
        if present:
            edges.append(torch.stack(present).mean())
        # Each phase gets equal weight, so the large Al matrix does not dominate.
        present = []
        for phase in range(4):
            select = valid[i] & (target[i] == phase)
            if select.any():
                present.append(distance_loss[i][select].mean())
        if present:
            shapes.append(torch.stack(present).mean())
    return base + (.1 * torch.stack(edges).mean() if edges else 0) + (.1 * torch.stack(shapes).mean() if shapes else 0)


def load_shape(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCHITECTURE:
        raise ValueError('Unexpected shape architecture')
    model = ShapeAuxNet()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
