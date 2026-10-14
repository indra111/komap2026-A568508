"""Candidate head that avoids the old native64 -> 128 -> native64 resampling.

This changes the native receptive field as well as removing interpolation.
It is an experimental architecture, not a claim that interpolation caused the
observed errors. The teacher, tile coverage and four-class residual stay fixed.
"""
import torch
from cached_refiner import CachedRefiner
from patch_refiner import PATCH

ARCHITECTURE = 'cached_four_class_native_grid_residual_bf16'


class NativeGridRefiner(CachedRefiner):
    def correction(self, images, probabilities):
        if images.ndim != 4 or images.shape[1:] != (3, PATCH, PATCH):
            raise ValueError('Expected aligned native64 RGB/probability patches')
        if probabilities.shape != (images.shape[0], 4, PATCH, PATCH):
            raise ValueError('Probability coordinates differ')
        features = torch.cat([(images-self.input_mean)/self.input_std, probabilities.float()], 1)
        with torch.autocast(images.device.type, dtype=torch.bfloat16, enabled=images.device.type == 'cuda'):
            delta = self.refiner(features)
        return delta.float()


def load_native_refiner(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCHITECTURE:
        raise ValueError('Wrong native-grid residual architecture')
    model = NativeGridRefiner()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
