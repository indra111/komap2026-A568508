"""DINOv2/DPT adaptation with immutable best and exact epoch-boundary resume."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from core import audit, file_hash, load_manifest, write_json
from expansion import ExpandedSampler, multiclass_loss
from context_model import evaluate_context
from pretrained_model import tensor_fingerprint
from dino_model import DinoDPT, initialize_dino, build_dino_optimizer
from run import atomic_checkpoint, now

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ['core.py', 'run.py', 'expansion.py', 'context_model.py', 'pretrained_model.py', 'train_pretrained.py','dino_model.py','train_dino.py','image_processing.py']


def restore_training(saved, model, optimizer, scheduler, scaler, sampler):
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    scaler.load_state_dict(saved['scaler'])
    state = saved['sampler']
    sampler.rng.bit_generator.state = state['rng']
    sampler.aug_rng.bit_generator.state = state['aug_rng']
    sampler.draws, sampler.seen = state['draws'], set(state['seen'])
    sampler.external_order = list(state['external_order'])
    torch.set_rng_state(saved['torch_rng'])
    if torch.cuda.is_available() and saved['cuda_rng']:
        torch.cuda.set_rng_state_all(saved['cuda_rng'])


def register_champion(run, savepoint, digest, metrics, config):
    if config['provenance'] != 'official_komap_2026':
        return
    review = ROOT / 'review/target80-iterations'
    review.mkdir(parents=True, exist_ok=True)
    with (review / '.champion.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = review / 'champion.json'
        old = json.loads(path.read_text()) if path.exists() else None
        score = metrics['mean_image_miou']
        if old and score <= old['mean_image_miou']:
            return
        value = {'id': run.name, 'mean_image_miou': score,
                 'savepoint': str(savepoint.relative_to(ROOT)), 'sha256': digest,
                 'architecture': config['model'], 'seed': config['seed'], 'initialization': config['initialization'],
                 'status': 'validation_record; independent reproduction pending', 'updatedAt': now()}
        if old:
            value['previous_record'] = {k: old[k] for k in ('id', 'mean_image_miou', 'savepoint', 'sha256') if k in old}
        write_json(path, value)
        with (review / 'champion-history.jsonl').open('a') as out:
            out.write(json.dumps(value) + '\n')


def train(a):
    run = Path(a.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if min(a.epochs, a.steps, a.batch) < 1 or a.patch % 32:
            raise ValueError('Invalid training settings')
        spec, rows = load_manifest(a.manifest)
        fingerprint = audit(a.manifest)
        config = {'provenance': spec['provenance'], 'model': a.architecture,
                  'initialization': a.initialization, 'patch': a.patch, 'overlap': a.patch // 4,
                  'epochs': a.epochs, 'steps': a.steps, 'batch': a.batch, 'seed': a.seed,
                  'encoder_lr': 2e-5, 'decoder_lr': 1e-3, 'scheduler': 'cosine, minimum factor 0.05',
                  'loss': 'cross_entropy_plus_all_class_dice', 'augmentation': 'dihedral_only',
                  'normalization': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]},
                  'manifest_sha256': file_hash(a.manifest), 'official_split_preserved': True,
                  'resume_supported': True, 'source_sha256': {n: file_hash(ROOT/n) for n in SOURCE_FILES}}
        resume = None
        if a.resume:
            resume = torch.load(a.resume, map_location='cpu', weights_only=True)
            if resume['config'] != config:
                raise ValueError('Resume config/data/code differ; supply the original arguments and source')
            existing = json.loads((run / 'config.json').read_text()) if (run/'config.json').exists() else None
            if existing is not None and existing != config:
                raise ValueError('Destination run config differs')
            if (run/'status.json').exists() and json.loads((run/'status.json').read_text())['state'] == 'completed':
                raise ValueError('Completed experiments are immutable; restore into a new run directory')
            if (run/'history.jsonl').exists():
                history = [json.loads(x) for x in (run/'history.jsonl').read_text().splitlines()]
                if history and history[-1]['epoch'] > resume['epoch']:
                    raise ValueError('Restoring an older savepoint requires a new run directory')
        elif (run/'config.json').exists():
            raise FileExistsError('Use a new run name; experiments are immutable')
        write_json(run/'config.json', config)
        write_json(run/'data-audit.json', fingerprint)
        (run/'source').mkdir(exist_ok=True)
        for n in SOURCE_FILES:
            shutil.copy2(ROOT/n, run/'source'/n)
        shutil.copy2(a.manifest, run/'source/manifest.json')
        try:
            torch.set_num_threads(4)
            random.seed(a.seed)
            np.random.seed(a.seed)
            torch.manual_seed(a.seed)
            device = torch.device(a.device)
            model = DinoDPT(a.architecture)
            decoder_hash = tensor_fingerprint(model.net.decoder.state_dict())
            head_hash = tensor_fingerprint(model.net.segmentation_head.state_dict())
            transfer = {'initialization': a.initialization, 'decoder_initial_sha256': decoder_hash,
                        'head_initial_sha256': head_hash}
            if a.initialization != 'scratch' and resume is None:
                weights = json.loads((ROOT/'data/pretrained/dinov2-small/manifest.json').read_text())
                transfer.update(initialize_dino(model, ROOT/weights['path'], weights['sha256']))
                transfer['source'] = weights
            if tensor_fingerprint(model.net.decoder.state_dict()) != decoder_hash or tensor_fingerprint(model.net.segmentation_head.state_dict()) != head_hash:
                raise RuntimeError('Transfer changed decoder or phase head')
            if resume:
                transfer = resume['initialization_audit']
                write_json(run/'resume-origin.json', {'file': str(Path(a.resume).resolve()),
                           'sha256': file_hash(a.resume), 'epoch': resume['epoch'], 'updatedAt': now()})
            write_json(run/'initialization-audit.json', transfer)
            model.to(device)
            optimizer = build_dino_optimizer(model)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .05 + .95 * (1 + math.cos(math.pi*min(step,a.epochs*a.steps)/(a.epochs*a.steps))) / 2)
            scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
            sampler = ExpandedSampler(rows, spec, a.patch, a.seed)
            best, best_epoch, first_epoch = -1., None, 1
            if resume:
                restore_training(resume, model, optimizer, scheduler, scaler, sampler)
                best, best_epoch, first_epoch = resume['best_miou'], resume['best_epoch'], resume['epoch'] + 1
                # Make the inherited best accessible in a new branch even without future improvement.
                if not (run/'best.pth').exists():
                    if resume['validation']['mean_image_miou'] != best:
                        raise ValueError('New branch must restore a best savepoint, not a worse last checkpoint')
                    atomic_checkpoint(run/'best.pth', {k:resume[k] for k in ['model','architecture','classes','config','epoch','validation','initialization_audit']})
                    write_json(run/'best-validation.json', resume['validation'])
                del resume
            if first_epoch > a.epochs:
                raise ValueError('This savepoint has already reached the configured training budget')
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            (run/'savepoints').mkdir(exist_ok=True)
            for epoch in range(first_epoch, a.epochs + 1):
                begin = time.monotonic()
                model.train()
                losses = []
                for step in range(1, a.steps + 1):
                    x, y = [t.to(device) for t in sampler.batch(a.batch)]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                        loss = multiclass_loss(model(x), y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss')
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    losses.append(float(loss.detach()))
                    if step % 25 == 0:
                        write_json(run/'status.json', {'state':'training','epoch':epoch,'epochs':a.epochs,
                                   'step':step,'steps':a.steps,'best_miou':best,'patch_views':sampler.draws,'updatedAt':now()})
                write_json(run/'status.json', {'state':'validating','epoch':epoch,'best_miou':best,'updatedAt':now()})
                metrics = evaluate_context(model, rows, spec, device, patch=a.patch, overlap=a.patch//4)
                improved = metrics['mean_image_miou'] > best
                if improved:
                    best, best_epoch = metrics['mean_image_miou'], epoch
                light = {'model':model.state_dict(), 'architecture':config['model'], 'classes':spec['classes'],
                         'config':config,'epoch':epoch,'validation':metrics,'initialization_audit':transfer}
                full = {**light,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'scaler':scaler.state_dict(),'sampler':sampler.state_dict(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                        'best_miou':best,'best_epoch':best_epoch}
                if improved:
                    savepoint = run/'savepoints'/f'epoch-{epoch:03d}-miou-{best:.8f}.pth'
                    if savepoint.exists():
                        raise FileExistsError('Never overwrite an immutable savepoint')
                    atomic_checkpoint(savepoint, full)
                    digest = file_hash(savepoint)
                    atomic_checkpoint(run/'best.pth', light)
                    write_json(run/'best-validation.json', metrics)
                    record = {'epoch':epoch,'miou':best,'file':str(savepoint.relative_to(run)),
                              'sha256':digest,'includes_optimizer_scheduler_rng_sampler':True,'updatedAt':now()}
                    with (run/'savepoints.jsonl').open('a') as out:
                        out.write(json.dumps(record)+'\n')
                    register_champion(run, savepoint, digest, metrics, config)
                atomic_checkpoint(run/'last.pth', full)
                record = {'epoch':epoch,'loss':float(np.mean(losses)),'miou':metrics['mean_image_miou'],
                          'best_miou':best,'new_savepoint':improved,'lr':[g['lr'] for g in optimizer.param_groups],
                          'patch_views':sampler.draws,'unique_source_crops_seen':len(sampler.seen),
                          'seconds':time.monotonic()-begin,'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0,
                          'updatedAt':now()}
                with (run/'history.jsonl').open('a') as out:
                    out.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
                if a.stop_after_epoch and epoch >= a.stop_after_epoch and epoch < a.epochs:
                    write_json(run/'status.json',{'state':'stopped_at_checkpoint','epoch':epoch,'best_miou':best,'updatedAt':now()})
                    return
            write_json(run/'status.json',{'state':'completed','epochs':a.epochs,'best_epoch':best_epoch,
                       'best_miou':best,'weights_sha256':file_hash(run/'best.pth'),'patch_views':sampler.draws,
                       'savepoints':len(list((run/'savepoints').glob('*.pth'))),'updatedAt':now()})
        except BaseException as exc:
            write_json(run/'status.json',{'state':'failed','error':str(exc),'updatedAt':now()})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default='data/manifest.json')
    p.add_argument('--run',required=True)
    p.add_argument('--initialization',choices=['scratch','dinov2'],default='dinov2')
    p.add_argument('--architecture',choices=['dpt_dinov2_small','dpt_dinov2_small_detail'],required=True)
    p.add_argument('--epochs',type=int,default=12)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--patch',type=int,default=512)
    p.add_argument('--seed',type=int,default=20261007)
    p.add_argument('--device',default='cuda')
    p.add_argument('--resume')
    p.add_argument('--stop-after-epoch',type=int)
    train(p.parse_args())
