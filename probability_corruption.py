"""Train-only probability perturbation; images and target labels are unchanged.

This is our multiclass probability augmentation hypothesis, not an implementation
of CascadePSP or SegRefiner. No validation/test data is needed by this module.
"""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from cached_refiner_data import CachedSampler

POLICY = {'clean': .5, 'shift': .25, 'coarsen': .25, 'max_shift_pixels': 2, 'coarsen_factor': 2}


def shift_probability(probability, dy, dx):
    if abs(dy) > 2 or abs(dx) > 2:
        raise ValueError('Shift outside predefined policy')
    h, w = probability.shape[-2:]
    pad = F.pad(probability, (2, 2, 2, 2), mode='replicate')
    return pad[..., 2+dy:2+dy+h, 2+dx:2+dx+w].clone()


def coarsen_probability(probability):
    h, w = probability.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError('Expected even native crop')
    return F.interpolate(F.avg_pool2d(probability, 2, 2), (h, w), mode='bilinear', align_corners=False)


class CorruptedCachedSampler(CachedSampler):
    def __init__(self, pairs, seed, method):
        super().__init__(pairs, seed)
        if method not in ('clean', 'spatial_noise'):
            raise ValueError(method)
        self.method = method
        # A separate stream leaves image selection/cropping/D4 identical to control.
        self.noise_rng = np.random.default_rng(seed+991)
        self.noise_counts = {'clean': 0, 'shift': 0, 'coarsen': 0}

    def batch(self, count):
        image, probability, target = super().batch(count)
        if self.method == 'clean':
            self.noise_counts['clean'] += count
            return image, probability, target
        noisy = probability.clone()
        for index in range(count):
            value = self.noise_rng.random()
            if value < .5:
                kind = 'clean'
            elif value < .75:
                kind = 'shift'
                offsets = [(y, x) for y in range(-2, 3) for x in range(-2, 3) if (y, x) != (0, 0)]
                dy, dx = offsets[int(self.noise_rng.integers(len(offsets)))]
                noisy[index:index+1] = shift_probability(probability[index:index+1], dy, dx)
            else:
                kind = 'coarsen'
                noisy[index:index+1] = coarsen_probability(probability[index:index+1])
            self.noise_counts[kind] += 1
        if not torch.isfinite(noisy).all() or (noisy < 0).any():
            raise ValueError('Probability augmentation must preserve a finite simplex')
        if not torch.allclose(noisy.sum(1), torch.ones_like(noisy[:, 0]), atol=2e-6, rtol=1e-5):
            raise ValueError('Probability mass changed')
        return image, noisy, target

    def state_dict(self):
        return {**super().state_dict(), 'noise_method': self.method,
                'noise_rng': copy.deepcopy(self.noise_rng.bit_generator.state), 'noise_counts': dict(self.noise_counts)}

    def load_state_dict(self, state):
        if state['noise_method'] != self.method:
            raise ValueError('Cannot change probability augmentation while resuming')
        super().load_state_dict(state)
        self.noise_rng.bit_generator.state = state['noise_rng']
        self.noise_counts = dict(state['noise_counts'])
