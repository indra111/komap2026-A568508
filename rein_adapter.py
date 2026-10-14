# SPDX-License-Identifier: GPL-3.0-only
"""Rein-core low-rank feature adapter, adapted from w1oves/Rein (2026-10-07).

Upstream: dc063429c4dadc0da9c6252b3db22fc55a9882ab, reins.py.
Changes: no MMSeg registry/query-link branch; batch-first interface; validate
dimensions. Core token attention and residual equations are preserved.
See third_party/rein-reference/LICENSE.txt and NOTICE.md.
"""
import math
import torch
from torch import nn


class LowRankRein(nn.Module):
    def __init__(self, layers=12, channels=768, patch_size=14, tokens=100, rank=16):
        super().__init__()
        if min(layers, channels, patch_size, rank) < 1 or tokens < 2:
            raise ValueError('Invalid adapter dimensions')
        self.layers, self.channels = layers, channels
        self.learnable_tokens_a = nn.Parameter(torch.empty(layers, tokens, rank))
        self.learnable_tokens_b = nn.Parameter(torch.empty(layers, rank, channels))
        self.scale = nn.Parameter(torch.tensor(.001))
        self.mlp_token2feat = nn.Linear(channels, channels)
        self.mlp_delta_f = nn.Linear(channels, channels)
        extent = math.sqrt(6. / (3 * patch_size * patch_size + (channels * rank) ** .5))
        nn.init.uniform_(self.learnable_tokens_a, -extent, extent)
        nn.init.uniform_(self.learnable_tokens_b, -extent, extent)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

    def forward(self, features, layer):
        if features.ndim != 3 or features.shape[-1] != self.channels or features.shape[1] < 2:
            raise ValueError('Expected B x (CLS+patches) x channels')
        if not 0 <= layer < self.layers:
            raise ValueError('Invalid layer index')
        # Preserve the original implementation's contraction order and CLS path.
        cls, patches = torch.tensor_split(features.permute(1, 0, 2), [1], dim=0)
        tokens = self.learnable_tokens_a[layer] @ self.learnable_tokens_b[layer]
        attention = torch.einsum('nbc,mc->nbm', patches, tokens)
        attention = (attention * self.channels ** -.5).softmax(dim=-1)
        # Token zero is a learned null alternative and does not add a feature.
        delta = torch.einsum('nbm,mc->nbc', attention[:, :, 1:], self.mlp_token2feat(tokens[1:]))
        delta = self.mlp_delta_f(delta + patches)
        return torch.cat([cls, patches + delta * self.scale], dim=0).permute(1, 0, 2)
