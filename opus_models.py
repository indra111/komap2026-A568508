"""Model families for scale-normalized training and a loader covering them.

Families reuse the original model classes, public initializations, optimizers
and the CE + all-class Dice loss of the original member trainers.
"""
import json
from pathlib import Path

import torch
from expansion import multiclass_loss

ROOT = Path(__file__).resolve().parent


def _loss(m, x, y):
    return multiclass_loss(m(x), y)


def build_family(family):
    if family == 'large':
        from dino_model import initialize_dino, build_dino_optimizer
        from dinov2_large_model import LargeDinoDPT
        model = LargeDinoDPT()
        w = json.loads((ROOT / 'data/pretrained/dinov2-large/manifest.json').read_text())
        initialize_dino(model, ROOT / w['path'], w['sha256'])
        return model, build_dino_optimizer, _loss
    if family == 'base':
        from dino_model import initialize_dino, build_dino_optimizer
        from large_dino_model import BaseDinoDPT
        model = BaseDinoDPT()
        w = json.loads((ROOT / 'data/pretrained/dinov2-base/manifest.json').read_text())
        initialize_dino(model, ROOT / w['path'], w['sha256'])
        return model, build_dino_optimizer, _loss
    if family == 'hrnet':
        from hrnet_ocr_model import HRNetOCR, initialize_hrnet, build_hrnet_optimizer, hrnet_training_loss
        model = HRNetOCR()
        w = json.loads((ROOT / 'data/pretrained/hrnet-w48/manifest.json').read_text())
        initialize_hrnet(model, ROOT / w['path'], w['sha256'])
        return model, build_hrnet_optimizer, hrnet_training_loss
    if family == 'dinov3':
        from dinov3_large_model import DinoV3LargeDPT, build_dinov3_large_optimizer, PUBLIC_FILE
        return DinoV3LargeDPT(ROOT / PUBLIC_FILE), build_dinov3_large_optimizer, _loss
    if family == 'convnext':
        from dinov3_convnext_model import DinoV3ConvNeXtFPN, build_optimizer, PUBLIC_FILE
        return DinoV3ConvNeXtFPN(ROOT / PUBLIC_FILE), build_optimizer, _loss
    raise ValueError(family)


def load_member(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    arch = saved['architecture']
    if arch == 'dpt_dinov3_large_detail_full_bf16_v1':
        from dinov3_large_model import DinoV3LargeDPT
        model = DinoV3LargeDPT()
    elif arch == 'dinov3_convnext_large_fpn_detail_v1':
        from dinov3_convnext_model import DinoV3ConvNeXtFPN
        model = DinoV3ConvNeXtFPN()
    else:
        from target80_evaluation import load_any
        return load_any(path, device)
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved


def llrd_optimizer(model, decay, encoder_lr=2e-5, head_lr=1e-3):
    """Layer-wise LR decay for timm ViT encoders inside smp DPT (DINOv2/DINOv3 Large).
    Block i of n gets encoder_lr * decay**(n-1-i); embeddings get the deepest decay."""
    vit = model.net.encoder.model
    n = len(vit.blocks)
    groups, seen = [], set()
    for i, block in enumerate(vit.blocks):
        params = [p for p in block.parameters() if p.requires_grad]
        groups.append({'params': params, 'lr': encoder_lr * decay ** (n - 1 - i)}); seen |= {id(p) for p in params}
    rest_encoder = [p for p in model.net.encoder.parameters() if p.requires_grad and id(p) not in seen]
    groups.append({'params': rest_encoder, 'lr': encoder_lr * decay ** n}); seen |= {id(p) for p in rest_encoder}
    head = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    groups.append({'params': head, 'lr': head_lr})
    assert sum(len(g['params']) for g in groups) == len([p for p in model.parameters() if p.requires_grad])
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def lovasz_softmax(probs, target, ignore=-100):
    """Lovasz-softmax (Berman et al., CVPR 2018) over present classes, batch-flattened.
    probs: (B,C,H,W) softmax; target: (B,H,W) with `ignore` pixels excluded."""
    C = probs.shape[1]
    p = probs.permute(0, 2, 3, 1).reshape(-1, C)
    t = target.reshape(-1)
    keep = t != ignore
    p, t = p[keep], t[keep]
    losses = []
    for c in range(C):
        fg = (t == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - p[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        gts = fg_sorted.sum()
        intersection = gts - fg_sorted.cumsum(0)
        union = gts + (1 - fg_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
        losses.append(torch.dot(errors_sorted, jaccard))
    return torch.stack(losses).mean() if losses else probs.sum() * 0.
