"""Retrain the focused residual refiner on the current teacher (PURPOSE plan 4).

Same model/loss/sampler/noise as train_focused_refiner_warm.py (Stage45), but the
teacher is a weighted mix of member caches written by eval_teacher_mix.py
(train split: in-sample teacher predictions, as in the original cache).
Warm-starts from the current champion head. Per-epoch validation uses the
fast single-view argmax metric; the final head is scored through the full
D4 -> CRF/RankIoU pipeline with eval_teacher_mix.py mix --head.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from core import decode_mask, load_manifest, read_image, write_json
from image_processing import enhance
from focused_refiner_model import create_model, load_focused_refiner, ARCHITECTURES
from focused_cached_sampler import FocusedCachedSampler
from cached_refiner_eval import evaluate
from train_focused_refiner_warm import training_objective
from run import atomic_checkpoint, now

ROOT = Path(__file__).resolve().parent


def pairs_for(split, work, weights):
    spec, rows = load_manifest(ROOT / 'data/manifest.json')
    out = []
    for row in rows:
        if row['split'] != split:
            continue
        p = sum(w * np.load(Path(work) / name / (row['id'] + '.npy')) for name, w in weights.items())
        p = (p / p.sum(0, keepdims=True)).astype(np.float32)
        out.append((row['id'], enhance(read_image(row['image']), 'he50'), p, decode_mask(row['mask'], spec)))
    return spec, rows, out


def main(a):
    run = Path(a.run); run.mkdir(parents=True, exist_ok=False)
    train_w, val_w = json.loads(a.train_weights), json.loads(a.val_weights)
    config = vars(a) | {'architecture': ARCHITECTURES['upsampled'], 'loss': 'training_objective(method) from Stage45',
                        'sampler': 'FocusedCachedSampler spatial_noise', 'teacher_train': 'in-sample member caches',
                        'validation_used_for_training': False, 'createdAt': now()}
    write_json(run / 'config.json', config)
    spec, rows, training = pairs_for('train', a.work, train_w)
    _, _, validation = pairs_for('val', a.val_work, val_w)
    torch.set_num_threads(4); torch.manual_seed(a.seed)
    device = torch.device('cuda')
    model, saved = load_focused_refiner(ROOT / a.init, 'cuda')     # also configures the strict runtime
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * a.steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: .05 + .95 * (1 + math.cos(math.pi * min(s, total) / total)) / 2)
    sampler = FocusedCachedSampler(training, {r['id']: r['split'] for r in rows}, a.seed, a.method, 'spatial_noise')
    base, _ = evaluate(model, validation, spec, device)
    best = base['mean_image_miou']
    print(json.dumps({'epoch': 0, 'miou_single': best}), flush=True)
    atomic_checkpoint(run / 'best.pth', {'model': model.state_dict(), 'architecture': saved['architecture'],
                                         'classes': saved['classes'], 'config': saved['config'], 'epoch': 0, 'validation': base})
    for epoch in range(1, a.epochs + 1):
        begin = time.monotonic(); losses = []; model.train()
        for _ in range(a.steps):
            x, p, y = [t.to(device) for t in sampler.batch(a.batch)]
            optimizer.zero_grad(set_to_none=True)
            loss = training_objective(model.training_logits(x, p), y, a.method)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        metrics, _ = evaluate(model, validation, spec, device)
        score = metrics['mean_image_miou']
        if score > best:
            best = score
            # Keep the original config so load_focused_refiner accepts the head.
            atomic_checkpoint(run / 'best.pth', {'model': model.state_dict(), 'architecture': saved['architecture'],
                                                 'classes': saved['classes'], 'config': saved['config'],
                                                 'epoch': epoch, 'validation': metrics})
        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'miou_single': score, 'best_single': best,
                  'seconds': time.monotonic() - begin, 'updatedAt': now()}
        with (run / 'history.jsonl').open('a') as out:
            out.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    write_json(run / 'status.json', {'state': 'completed', 'best_single': best, 'updatedAt': now()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--work', default='/dev/shm/komap-opus-eval/train-members')
    p.add_argument('--val-work', default='/dev/shm/komap-opus-eval/members')
    p.add_argument('--train-weights', required=True); p.add_argument('--val-weights', required=True)
    p.add_argument('--init', default='runs/target100-focused-warm-uniform-v1/best.pth')
    p.add_argument('--method', default='uniform'); p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--batch', type=int, default=16); p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--steps', type=int, default=300); p.add_argument('--seed', type=int, default=20261014)
    main(p.parse_args())
