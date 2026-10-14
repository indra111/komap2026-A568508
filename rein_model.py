# SPDX-License-Identifier: GPL-3.0-only
"""Frozen DINOv2 Base controls and a Rein-core variant with identical DPT heads."""
import torch
from large_dino_model import BaseDinoDPT
from rein_adapter import LowRankRein

ARCHITECTURES = ('dpt_dinov2_base_frozen_detail_bf16', 'dpt_dinov2_base_rein_detail_bf16')


class ReinDinoDPT(BaseDinoDPT):
    def __init__(self, architecture=ARCHITECTURES[1]):
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        super().__init__()
        self.architecture = architecture
        encoder = self.net.encoder.model
        assert len(encoder.blocks) == 12 and encoder.embed_dim == 768
        assert encoder.num_prefix_tokens == 1
        self.net.encoder.requires_grad_(False)
        self._rein_hooks = []
        if architecture == ARCHITECTURES[1]:
            self.reins = LowRankRein()
            for layer, block in enumerate(encoder.blocks):
                self._rein_hooks.append(block.register_forward_hook(
                    lambda module, inputs, output, index=layer, adapter=self.reins: adapter(output, index)))
        self.net.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.net.encoder.eval()
        return self


def build_rein_optimizer(model):
    groups = [{'params': model.net.decoder.parameters(), 'lr': 1e-3},
              {'params': model.net.segmentation_head.parameters(), 'lr': 1e-3},
              {'params': model.detail.parameters(), 'lr': 1e-3}]
    if hasattr(model, 'reins'):
        groups.append({'params': model.reins.parameters(), 'lr': 1e-4})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    included = [p for group in optimizer.param_groups for p in group['params']]
    assert len(included) == len({id(p) for p in included})
    assert {id(p) for p in included} == {id(p) for p in model.parameters() if p.requires_grad}
    return optimizer


def load_rein_dino(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = ReinDinoDPT(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
