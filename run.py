"""Audit, train and predict. Outputs are local; nothing is submitted automatically."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from core import (PatchSampler, audit, decode_mask, file_hash, load_manifest, make_model,
                  predict, read_image, save_mask, score_image, segmentation_loss, write_json)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_checkpoint(path, value):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    temp.replace(path)


def evaluate(model, rows, spec, device, patch, overlap):
    results = []
    alternatives = {policy: [] for policy in ('skip', 'one', 'zero')}
    for row in rows:
        if row['split'] != 'val':
            continue
        predicted = predict(model, read_image(row['image']), device, patch, overlap)
        target = decode_mask(row['mask'], spec)
        score = score_image(predicted, target, spec['absent_class_policy'])
        for policy in alternatives:
            alternatives[policy].append(score_image(predicted, target, policy)['miou'])
        results.append({'id': row['id'], **score})
    if not results:
        raise ValueError('No validation images')
    class_means = []
    for c in range(4):
        values = [r['iou_per_class'][c] for r in results if r['iou_per_class'][c] is not None]
        class_means.append(float(np.mean(values)) if values else None)
    return {'mean_image_miou': float(np.mean([r['miou'] for r in results])),
            'class_mean_iou': class_means, 'images': results,
            'metric_rule_status': spec['metric_rule_status'], 'absent_class_policy': spec['absent_class_policy'],
            'alternative_absent_policy_miou': {p: float(np.mean(v)) for p, v in alternatives.items()}}


def train(args):
    run = Path(args.run).resolve()
    if args.patch < 32 or args.patch % 32 or not 0 <= args.overlap < args.patch:
        raise ValueError('Invalid patch/overlap')
    if min(args.epochs, args.steps, args.batch) < 1 or args.lr <= 0:
        raise ValueError('Epochs, steps, batch and learning rate must be positive')
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (run / 'config.json').exists() and not args.resume:
            raise FileExistsError('Run already exists. Choose a new run or explicitly resume.')
        if args.resume and not (run / 'last.pth').exists():
            raise FileNotFoundError('Resume requires last.pth')
        spec, rows = load_manifest(args.manifest)
        if spec.get('provenance') == 'synthetic_fixture' and not args.allow_synthetic:
            raise ValueError('Fixture data requires --allow-synthetic; never use it as competition results')
        fingerprint = audit(args.manifest)
        config = {'model': 'unet_resnet34_random_init', 'patch': args.patch, 'overlap': args.overlap,
                  'batch': args.batch, 'steps': args.steps, 'lr': args.lr, 'seed': args.seed,
                  'manifest_sha256': fingerprint['manifest_sha256'], 'provenance': spec.get('provenance'),
                  'absent_class_policy': spec['absent_class_policy'], 'metric_rule_status': spec['metric_rule_status'],
                  'pretrained_weights': None, 'loss': 'cross_entropy_plus_four_class_dice'}
        config['source_sha256'] = {name: file_hash(Path(__file__).parent / name) for name in ('core.py', 'run.py')}
        if args.resume:
            if json.loads((run / 'config.json').read_text()) != config:
                raise ValueError('Resume config differs from original run')
            if json.loads((run / 'data-audit.json').read_text()) != fingerprint:
                raise ValueError('Dataset changed since run started')
        else:
            write_json(run / 'config.json', config)
            write_json(run / 'data-audit.json', fingerprint)
        try:
            torch.set_num_threads(4)
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            device = torch.device(args.device)
            if device.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable')
            model = make_model().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
            sampler = PatchSampler(rows, spec, args.patch, args.seed)
            start, best = 0, -1.
            if args.resume:
                saved = torch.load(run / 'last.pth', map_location='cpu', weights_only=True)
                model.load_state_dict(saved['model'])
                optimizer.load_state_dict(saved['optimizer'])
                scaler.load_state_dict(saved['scaler'])
                sampler.rng.bit_generator.state = saved['sampler_rng']
                torch.set_rng_state(saved['torch_rng'])
                if device.type == 'cuda':
                    torch.cuda.set_rng_state_all(saved['cuda_rng'])
                start, best = saved['epoch'], saved['best_score']
            if start >= args.epochs:
                raise ValueError('Requested epoch count has already been reached')
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for epoch in range(start, args.epochs):
                begin = time.monotonic()
                model.train()
                losses = []
                for step in range(args.steps):
                    x, y = sampler.batch(args.batch)
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        loss = segmentation_loss(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite training loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach()))
                    write_json(run / 'status.json', {'state': 'training', 'epoch': epoch + 1,
                               'epochs': args.epochs, 'step': step + 1, 'steps': args.steps,
                               'updatedAt': now(), 'provenance': config['provenance']})
                write_json(run / 'status.json', {'state': 'validating', 'epoch': epoch + 1, 'updatedAt': now()})
                metrics = evaluate(model, rows, spec, device, args.patch, args.overlap)
                score = metrics['mean_image_miou']
                payload = {'model': model.state_dict(), 'architecture': 'unet_resnet34',
                           'classes': spec['classes'], 'mask_encoding': spec['mask_encoding'],
                           'config': config, 'epoch': epoch + 1, 'validation': metrics}
                if score > best:
                    best = score
                    atomic_checkpoint(run / 'best.pth', payload)
                    write_json(run / 'best-validation.json', metrics)
                atomic_checkpoint(run / 'last.pth', {**payload, 'optimizer': optimizer.state_dict(),
                                  'scaler': scaler.state_dict(), 'sampler_rng': sampler.rng.bit_generator.state,
                                  'torch_rng': torch.get_rng_state(),
                                  'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                                  'best_score': best})
                record = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'miou': score,
                          'best_miou': best, 'seconds': time.monotonic() - begin, 'updatedAt': now(),
                          'peak_vram_mib': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else None}
                with (run / 'history.jsonl').open('a') as out:
                    out.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
            write_json(run / 'status.json', {'state': 'completed', 'epoch': args.epochs, 'best_miou': best,
                       'provenance': config['provenance'], 'metric_rule_status': spec['metric_rule_status'],
                       'best_weights_sha256': file_hash(run / 'best.pth'), 'updatedAt': now()})
        except BaseException as exc:
            write_json(run / 'status.json', {'state': 'failed', 'error': str(exc), 'updatedAt': now()})
            raise


def infer(args):
    spec, rows = load_manifest(args.manifest)
    selected = [r for r in rows if r['split'] == args.split]
    if not selected:
        raise ValueError('No images in requested split')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    saved = torch.load(args.weights, map_location='cpu', weights_only=True)
    if saved['classes'] != spec['classes'] or saved['mask_encoding'] != spec['mask_encoding']:
        raise ValueError('Class/color mapping differs from trained weights')
    if saved['config'].get('provenance') == 'synthetic_fixture' and not args.allow_synthetic:
        raise ValueError('Fixture weights are not competition weights')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = make_model().to(device)
    model.load_state_dict(saved['model'])
    records = []
    for row in selected:
        result = predict(model, read_image(row['image']), device,
                         saved['config']['patch'], saved['config']['overlap'])
        filename = row['id'] + '.png'
        save_mask(output / filename, result, spec)
        records.append({'id': row['id'], 'file': filename, 'shape': list(result.shape),
                        'sha256': file_hash(output / filename)})
    write_json(output / 'prediction-manifest.json', {'split': args.split,
               'weights_sha256': file_hash(args.weights), 'source_manifest_sha256': file_hash(args.manifest),
               'provenance': saved['config'].get('provenance'), 'createdAt': now(), 'images': records})
    print(json.dumps({'predicted': len(records), 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--manifest', required=True)
    a.add_argument('--output', required=True)
    t = sub.add_parser('train')
    t.add_argument('--manifest', required=True)
    t.add_argument('--run', required=True)
    for name, default in [('epochs', 30), ('steps', 100), ('batch', 2), ('patch', 512), ('overlap', 128), ('seed', 20261007)]:
        t.add_argument('--' + name, type=int, default=default)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--device', default='cuda')
    t.add_argument('--resume', action='store_true')
    t.add_argument('--allow-synthetic', action='store_true')
    p = sub.add_parser('predict')
    p.add_argument('--manifest', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='test')
    p.add_argument('--device', default='cuda')
    p.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    if args.command == 'audit':
        write_json(args.output, audit(args.manifest))
    elif args.command == 'train':
        train(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()
