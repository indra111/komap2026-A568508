"""Matched spatial-noise residual training with optional multiclass blob loss."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from core import file_hash, write_json
from expansion import multiclass_loss
from component_blob_loss import multiclass_blob_loss
from cached_refiner import CachedRefiner, ARCHITECTURE
from cached_refiner_data import load_training
from probability_corruption import CorruptedCachedSampler, POLICY
from cached_refiner_eval import load_validation, evaluate
from predict_target80 import checked_recipe
from run import atomic_checkpoint, now

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT
SOURCE_FILES = ['component_blob_loss.py', 'train_component_refiner.py', 'probability_corruption.py', 'train_corrupted_refiner.py', 'cached_refiner.py', 'cached_refiner_data.py', 'cached_refiner_eval.py', 'train_cached_refiner.py',
                'patch_refiner.py', 'core.py', 'run.py', 'expansion.py', 'context_model.py', 'image_processing.py',
                'train_prediction_cache.py', 'pretrained_model.py', 'large_dino_model.py', 'dino_model.py',
                'native_detail.py', 'target80_methods.py', 'train_pretrained.py']


METHODS = {'control': 0., 'blob025': .25}


def training_objective(logits, target, method):
    weight = METHODS[method]
    global_loss = multiclass_loss(logits, target)
    # Exact matched control: avoid even evaluating the auxiliary loss at zero.
    if weight == 0.:
        return global_loss
    return global_loss + weight*multiclass_blob_loss(logits, target)


def restore(saved, model, optimizer, scheduler, sampler):
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    sampler.load_state_dict(saved['sampler']); torch.set_rng_state(saved['torch_rng'])
    if saved['cuda_rng']:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def train(a):
    run = (ROOT/a.run).resolve()
    assert run.is_relative_to(ROOT.resolve())
    run.mkdir(parents=True, exist_ok=True)
    with (run/'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.method not in METHODS:
            raise ValueError('Unknown fixed loss method')
        if min(a.epochs, a.steps, a.batch) < 1:
            raise ValueError('Invalid training budget')
        teacher = checked_recipe(ROOT/a.teacher)
        spec, training, train_hashes = load_training(ROOT, a.cache, a.cache_audit)
        val_spec, validation, val_hashes = load_validation(ROOT, ROOT/a.validation, a.teacher)
        assert spec['classes'] == val_spec['classes'] == teacher['classes']
        assert {i[0] for i in training}.isdisjoint(i[0] for i in validation)
        cache_manifest = json.loads((ROOT/a.cache/'manifest.json').read_text())
        assert cache_manifest['config']['teacher_recipe_sha256'] == file_hash(ROOT/a.teacher)
        source_sha = {name: file_hash(CODE_ROOT/name) for name in SOURCE_FILES}
        config = {'provenance': spec['provenance'], 'architecture': ARCHITECTURE,
                  'teacher_recipe': a.teacher, 'teacher_recipe_sha256': file_hash(ROOT/a.teacher),
                  'training_cache': a.cache, 'training_cache_manifest_sha256': file_hash(ROOT/a.cache/'manifest.json'),
                  'evaluation_cache': a.validation, 'evaluation_cache_manifest_sha256': file_hash(ROOT/a.validation/'manifest.json'),
                  'original_manifest_sha256': file_hash(ROOT/'data/manifest.json'),
                  'training_inputs': train_hashes, 'validation_inputs': val_hashes,
                  'epochs': a.epochs, 'steps': a.steps, 'batch': a.batch, 'seed': a.seed, 'lr': a.lr,
                  'sampling': 'independent uniform image and native64 patch per batch element; aligned D4',
                  'native_patch': 64, 'zoom': 2, 'inference_stride': 32,
                  'loss': 'CE plus original four-class batch Dice plus optional multiclass blob DC+BCE',
                  'method': a.method, 'blob_coefficient': METHODS[a.method], 'blob_connectivity': 8,
                  'blob_class_policy': 'all four phases including matrix; present-class auxiliary mean',
                  'training_device_type': torch.device(a.device).type, 'source_sha256': source_sha,
                  'precision': 'BF16 conv; log probabilities and residual/loss FP32; no gradient scaler',
                  'teacher_frozen': True, 'OOF_predictions': False,
                  'probability_augmentation': 'spatial_noise', 'augmentation_policy': POLICY,
                  'validation_probability_augmentation': False,
                  'validation_used_for_training': False, 'resume_supported': True}
        resume = None
        if a.resume:
            resume = torch.load(ROOT/a.resume, map_location='cpu', weights_only=True)
            if resume['config'] != config:
                raise ValueError('Resume config/data/source differ')
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text()) != config:
                raise ValueError('Destination config differs')
            if (run/'status.json').exists() and json.loads((run/'status.json').read_text())['state'] == 'completed':
                raise ValueError('Completed experiments are immutable')
            if (run/'history.jsonl').exists():
                history = [json.loads(s) for s in (run/'history.jsonl').read_text().splitlines()]
                if history and history[-1]['epoch'] > resume['epoch']:
                    raise ValueError('Older saves require a new destination')
        elif (run/'config.json').exists():
            raise FileExistsError('Use a new immutable run name')
        write_json(run/'config.json', config)
        for name in SOURCE_FILES:
            target = run/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CODE_ROOT/name, target)
        torch.set_num_threads(4); torch.manual_seed(a.seed)
        device = torch.device(a.device)
        model = CachedRefiner().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .05+.95*(1+math.cos(math.pi*min(step,a.epochs*a.steps)/(a.epochs*a.steps)))/2)
        sampler = CorruptedCachedSampler(training, a.seed, 'spatial_noise')
        first, best, best_epoch = 1, -1., None
        if resume:
            restore(resume, model, optimizer, scheduler, sampler)
            first, best, best_epoch = resume['epoch']+1, resume['best_miou'], resume['best_epoch']
            if not (run/'best.pth').exists():
                if resume['validation']['mean_image_miou'] != best:
                    raise ValueError('New branches must start from a best full savepoint')
                atomic_checkpoint(run/'best.pth', {k: resume[k] for k in ('model', 'architecture', 'classes', 'config', 'epoch', 'validation')})
                write_json(run/'best-validation.json', resume['validation'])
            write_json(run/'resume-origin.json', {'file': a.resume, 'sha256': file_hash(ROOT/a.resume), 'epoch': resume['epoch'], 'updatedAt': now()})
            del resume
        if first > a.epochs:
            raise ValueError('Training budget already reached')
        (run/'savepoints').mkdir(exist_ok=True)
        try:
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            for epoch in range(first, a.epochs+1):
                begin = time.monotonic(); losses = []; model.train()
                for step in range(1, a.steps+1):
                    x, p, y = [t.to(device) for t in sampler.batch(a.batch)]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.training_logits(x, p)
                    loss = training_objective(logits, y, a.method)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
                    if step % 25 == 0:
                        write_json(run/'status.json', {'state': 'training', 'epoch': epoch, 'epochs': a.epochs,
                                   'step': step, 'steps': a.steps, 'best_miou': best, 'patches': sampler.draws, 'updatedAt': now()})
                write_json(run/'status.json', {'state': 'validating', 'epoch': epoch, 'best_miou': best, 'updatedAt': now()})
                metrics, _ = evaluate(model, validation, spec, device)
                improved = metrics['mean_image_miou'] > best
                if improved:
                    best, best_epoch = metrics['mean_image_miou'], epoch
                light = {'model': model.state_dict(), 'architecture': ARCHITECTURE, 'classes': spec['classes'],
                         'config': config, 'epoch': epoch, 'validation': metrics}
                full = {**light, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'scaler': {}, 'sampler': sampler.state_dict(), 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                        'best_miou': best, 'best_epoch': best_epoch}
                if improved:
                    save = run/'savepoints'/f'epoch-{epoch:03d}-miou-{best:.8f}.pth'
                    if save.exists():
                        raise FileExistsError('Never overwrite immutable savepoints')
                    atomic_checkpoint(save, full); atomic_checkpoint(run/'best.pth', light)
                    write_json(run/'best-validation.json', metrics)
                    with (run/'savepoints.jsonl').open('a') as stream:
                        stream.write(json.dumps({'epoch': epoch, 'miou': best, 'file': str(save.relative_to(run)),
                            'sha256': file_hash(save), 'includes_optimizer_scheduler_rng_sampler': True, 'updatedAt': now()})+'\n')
                atomic_checkpoint(run/'last.pth', full)
                history = {'epoch': epoch, 'loss': float(np.mean(losses)), 'miou': metrics['mean_image_miou'],
                           'best_miou': best, 'new_savepoint': improved, 'patches': sampler.draws,
                           'unique_train_images_seen': len(sampler.seen), 'noise_counts': dict(sampler.noise_counts),
                           'seconds': time.monotonic()-begin,
                           'peak_vram_mib': torch.cuda.max_memory_allocated()/2**20 if device.type == 'cuda' else 0,
                           'updatedAt': now()}
                with (run/'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(history)+'\n')
                print(json.dumps(history), flush=True)
                if a.stop_after_epoch and epoch >= a.stop_after_epoch and epoch < a.epochs:
                    write_json(run/'status.json', {'state': 'stopped_at_checkpoint', 'epoch': epoch, 'best_miou': best, 'updatedAt': now()})
                    return
            assert all(file_hash(CODE_ROOT/name) == sha for name, sha in source_sha.items())
            write_json(run/'status.json', {'state': 'completed', 'epochs': a.epochs, 'best_epoch': best_epoch,
                       'best_miou': best, 'weights_sha256': file_hash(run/'best.pth'), 'updatedAt': now()})
        except BaseException as exc:
            write_json(run/'status.json', {'state': 'failed', 'error': str(exc), 'updatedAt': now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'teacher', 'cache', 'cache-audit', 'validation'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--method', choices=list(METHODS), required=True)
    parser.add_argument('--epochs', type=int, default=12); parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16); parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=20261008); parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume'); parser.add_argument('--stop-after-epoch', type=int)
    train(parser.parse_args())
