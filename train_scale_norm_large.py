"""DINOv2 Large/native-detail trained on physical-scale-normalized images.

Matched to runs/target100-dinov2-large-detail-v1 (same init, optimizer,
schedule, seed, crop RNG stream); only the low-resolution images are enlarged
(scale_norm.py). Validation is scored on the original masks/resolution.
Disk is scarce, so only light best/last weights are written (no full resume).
"""
import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from core import file_hash, load_manifest, write_json
from expansion import multiclass_loss
from dino_model import initialize_dino, build_dino_optimizer
from opus_models import build_family
import copy
from run import atomic_checkpoint, now
from scale_norm import FACTOR, SMALL_WIDTH, CopyPasteSampler, ScaleNormSampler, evaluate_normalized

ROOT = Path(__file__).resolve().parent
SOURCES = ['scale_norm.py', 'train_scale_norm_large.py', 'opus_models.py', 'dinov2_large_model.py', 'large_dino_model.py',
           'hrnet_ocr_model.py', 'dinov3_large_model.py', 'dinov3_convnext_model.py', 'dino_model.py', 'expansion.py', 'context_model.py', 'core.py', 'target80_methods.py']


def train(a):
    run = Path(a.run)
    if (run / 'config.json').exists():
        raise FileExistsError('Use a new run directory')
    run.mkdir(parents=True)
    spec, rows = load_manifest(a.manifest)
    config = {'family': a.family, 'ema': a.ema, 'llrd': a.llrd, 'llrd_lr': a.llrd_lr if a.llrd else None, 'epochs': a.epochs, 'steps': a.steps, 'batch': a.batch, 'patch': a.patch,
              'seed': a.seed, 'encoder_lr': 2e-5, 'decoder_lr': 1e-3, 'augmentation': f'dihedral+copy_paste{a.copy_paste}' if a.copy_paste else 'dihedral_only', 'lowres_boost': a.lowres_boost, 'init': a.init, 'lovasz': a.lovasz, 'head_lr': a.head_lr, 'lr_scale': a.lr_scale,
              'scale_normalization': {'small_width_lt': SMALL_WIDTH, 'factor': FACTOR, 'base': a.base_scale},
              'control_run': {'large': 'runs/target100-dinov2-large-detail-v1', 'base': 'runs/target80-dpt-dinov2-base-detail-bf16-v1',
                              'hrnet': 'runs/target100-hrnet-ocr-v1'}.get(a.family),
              'manifest_sha256': file_hash(a.manifest), 'source_sha256': {n: file_hash(ROOT / n) for n in SOURCES}}
    write_json(run / 'config.json', config)
    (run / 'source').mkdir()
    for n in SOURCES:
        shutil.copy2(ROOT / n, run / 'source' / n)
    torch.set_num_threads(4)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device)
    model, make_optimizer, loss_fn = build_family(a.family)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=True)['model'], strict=True)
    if a.lovasz:
        from expansion import multiclass_loss
        from opus_models import lovasz_softmax
        if a.family == 'hrnet':
            import torch.nn.functional as F
            def loss_fn(m, x, y):
                logits, auxiliary = m.forward_with_aux(x)
                logits = logits.float()
                return (multiclass_loss(logits, y) + .4 * F.cross_entropy(auxiliary.float(), y)
                        + a.lovasz * lovasz_softmax(logits.softmax(1), y))
        else:
            def loss_fn(m, x, y):
                logits = m(x).float()
                return multiclass_loss(logits, y) + a.lovasz * lovasz_softmax(logits.softmax(1), y)
    model.to(device)
    if a.llrd:
        from opus_models import llrd_optimizer
        optimizer = llrd_optimizer(model, a.llrd, encoder_lr=a.llrd_lr, head_lr=a.head_lr)
    else:
        optimizer = make_optimizer(model)
    if a.lr_scale != 1.:
        for g in optimizer.param_groups:
            g['lr'] *= a.lr_scale
    # Optional EMA of weights (evaluated and saved alongside the raw weights).
    ema = copy.deepcopy(model).eval() if a.ema else None
    if ema is not None:
        for q in ema.parameters():
            q.requires_grad_(False)
    total = a.epochs * a.steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: .05 + .95 * (1 + math.cos(math.pi * min(s, total) / total)) / 2)
    sampler = (CopyPasteSampler if a.copy_paste else ScaleNormSampler)(rows, spec, a.patch, a.seed)
    if a.copy_paste:
        sampler.prob = a.copy_paste
    if a.lowres_boost > 1:
        sampler.boost_lowres(a.lowres_boost)
    sampler.base = a.base_scale
    best = -1.
    for epoch in range(1, a.epochs + 1):
        begin = time.monotonic()
        model.train()
        losses = []
        for step in range(1, a.steps + 1):
            x, y = [t.to(device) for t in sampler.batch(a.batch)]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                loss = loss_fn(model, x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if ema is not None:
                with torch.no_grad():
                    for q, w in zip(ema.parameters(), model.parameters()):
                        q.mul_(a.ema).add_(w.detach(), alpha=1 - a.ema)
                    for q, w in zip(ema.buffers(), model.buffers()):
                        q.copy_(w)
        metrics = evaluate_normalized(model, rows, spec, device, patch=a.patch, overlap=a.patch // 4, base=a.base_scale)
        score = metrics['mean_image_miou']; source = model; ema_score = None
        if ema is not None:
            ema_metrics = evaluate_normalized(ema, rows, spec, device, patch=a.patch, overlap=a.patch // 4, base=a.base_scale)
            ema_score = ema_metrics['mean_image_miou']
            if ema_score > score:
                score, metrics, source = ema_score, ema_metrics, ema
        light = {'model': source.state_dict(), 'architecture': model.architecture, 'classes': spec['classes'],
                 'config': config, 'epoch': epoch, 'validation': metrics, 'weights': 'ema' if source is ema else 'raw'}
        if score > best:
            best = score
            atomic_checkpoint(run / 'best.pth', light)
            write_json(run / 'best-validation.json', metrics)
        if epoch == a.epochs:
            atomic_checkpoint(run / 'last.pth', light)
        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'miou': score, 'ema_miou': ema_score, 'best_miou': best,
                  'seconds': time.monotonic() - begin, 'updatedAt': now()}
        with (run / 'history.jsonl').open('a') as out:
            out.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    write_json(run / 'status.json', {'state': 'completed', 'best_miou': best, 'updatedAt': now()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='data/manifest.json')
    p.add_argument('--run', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--patch', type=int, default=512)
    p.add_argument('--seed', type=int, default=20261007)
    p.add_argument('--device', default='cuda')
    p.add_argument('--base-scale', type=float, default=1.)
    p.add_argument('--family', choices=('large', 'base', 'hrnet', 'dinov3', 'convnext'), default='large')
    p.add_argument('--ema', type=float, default=0., help='EMA decay per step, e.g. 0.999; 0 disables')
    p.add_argument('--llrd', type=float, default=0., help='layer-wise LR decay for ViT families, e.g. 0.9; 0 disables')
    p.add_argument('--llrd-lr', type=float, default=6e-5, help='top-block encoder LR when --llrd is set')
    p.add_argument('--copy-paste', type=float, default=0., help='per-crop probability of Al3Ni/eutectic Si particle copy-paste; 0 disables')
    p.add_argument('--lowres-boost', type=int, default=1, help='draw width<800 training images k times as often; 1 disables')
    p.add_argument('--init', default=None, help='start from this best.pth (fine-tuning)')
    p.add_argument('--lovasz', type=float, default=0., help='weight of an added Lovasz-softmax term; 0 disables')
    p.add_argument('--head-lr', type=float, default=1e-3, help='decoder/head LR when --llrd is set')
    p.add_argument('--lr-scale', type=float, default=1., help='multiply every optimizer LR (e.g. 0.1 for fine-tuning)')
    train(p.parse_args())
