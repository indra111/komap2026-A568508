"""Restartable train-only arithmetic teacher predictions, with immutable provenance.

This does not read target masks or produce out-of-fold predictions. Teacher
models have already seen train70 labels; the cache supports a supervised model
extension, never an independent estimate of teacher accuracy.
"""
import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from core import file_hash, load_manifest, read_image, write_json
from image_processing import enhance
from predict_target80 import checked_recipe, predict_member
from target80_evaluation import load_any
from run import now

ROOT = Path(__file__).resolve().parent
CODE_ROOT = ROOT
SUPPORT = ('train_prediction_cache.py', 'predict_target80.py', 'target80_evaluation.py', 'tools/build_train_cache.py')


def inside(path):
    value = (ROOT / path).resolve()
    if not value.is_relative_to(ROOT.resolve()):
        raise ValueError('Path escapes project')
    return value


def valid_probabilities(value, shape):
    if value.dtype != np.float32 or value.shape != (4, *shape) or not np.isfinite(value).all():
        raise ValueError('Expected finite native-resolution four-class FP32 probabilities')
    if value.min() < -1e-7 or value.max() > 1+1e-7 or not np.allclose(value.sum(0), 1., atol=2e-6, rtol=1e-5):
        raise ValueError('Invalid probability simplex')


def save_array(path, array):
    if path.exists():
        raise FileExistsError('Uncommitted or immutable prediction exists: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npy.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def build(recipe_path, output, device='cuda', resume=False):
    output = inside(output)
    if output.exists() and not resume:
        raise FileExistsError('Use explicit resume or a new cache directory')
    if resume and not output.is_dir():
        raise FileNotFoundError('No cache to resume')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe_path = inside(recipe_path)
        recipe = checked_recipe(recipe_path)
        spec, all_rows = load_manifest(ROOT / 'data/manifest.json')
        if spec['provenance'] == 'official_komap_2026':
            assert [sum(r['split'] == split for r in all_rows) for split in ('train','val','test')] == [70,20,10]
        assert recipe['classes'] == spec['classes'] and spec['absent_class_policy'] == 'skip'
        rows = sorted((r for r in all_rows if r['split'] == 'train'), key=lambda r:r['id'])
        if not rows:
            raise ValueError('No training images')
        inputs = [{'id':r['id'], 'file':str(inside(r['image']).relative_to(ROOT)),
                   'sha256':file_hash(r['image'])} for r in rows]
        source_hashes = {name:file_hash(CODE_ROOT/name) for name in SUPPORT}
        config = {'teacher_recipe':str(recipe_path.relative_to(ROOT)), 'teacher_recipe_sha256':file_hash(recipe_path),
                  'model_files':recipe['model_files'], 'input_manifest':'data/manifest.json',
                  'input_manifest_sha256':file_hash(ROOT/'data/manifest.json'), 'classes':spec['classes'],
                  'split':'train', 'inputs':inputs, 'mask_read':False, 'OOF_predictions':False,
                  'precision':'FP32; sequential weighted arithmetic exactly as predict_target80',
                  'support_sha256':source_hashes, 'inference_source_sha256':recipe['inference_source_sha256']}
        config_path = output/'config.json'
        if resume:
            if json.loads(config_path.read_text()) != config:
                raise ValueError('Cache source/config/input changed')
            if (output/'status.json').exists() and json.loads((output/'status.json').read_text())['state']=='completed':
                raise ValueError('Completed caches are immutable')
        else:
            write_json(config_path,config)
            for name in SUPPORT:
                p=output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes((CODE_ROOT/name).read_bytes())
            for name,sha in recipe['inference_source_sha256'].items():
                original=inside(name);assert file_hash(original)==sha
                p=output/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(original.read_bytes())
        index_path = output/'index.json'
        index = json.loads(index_path.read_text()) if index_path.exists() else {'members':{},'combined':{}}
        device=torch.device(device)
        try:
            for number, member in enumerate(recipe['model_files']):
                key=str(number)
                done=index['members'].setdefault(key,{})
                model=saved=None
                for row,source in zip(rows,inputs):
                    write_json(output/'status.json',{'state':'predicting_train','member':number,
                        'image':row['id'],'completed_in_member':len(done),'training_images':len(rows),'updatedAt':now()})
                    path=output/'members'/key/(row['id']+'.npy')
                    if row['id'] in done:
                        if done[row['id']]['file'] != str(path.relative_to(output)) or file_hash(path) != done[row['id']]['sha256']:
                            raise ValueError('Cached member prediction changed')
                        p=np.load(path,allow_pickle=False);valid_probabilities(p,tuple(done[row['id']]['shape']));del p
                        continue
                    if path.exists():
                        raise FileExistsError('Uncommitted prediction requires explicit recovery: '+str(path))
                    if model is None:
                        model,saved=load_any(ROOT/member['file'],device)
                        assert saved['classes']==spec['classes']
                        model.eval()
                    if file_hash(row['image']) != source['sha256']:
                        raise ValueError('Training image changed during generation')
                    image=read_image(row['image'])
                    shape=image.shape[:2]
                    with torch.inference_mode():
                        p=predict_member(model,enhance(image,member['processing']),device,member,
                                         member.get('patch',saved['config']['patch']),
                                         member.get('overlap',saved['config']['overlap']))
                    valid_probabilities(p,shape)
                    if file_hash(row['image']) != source['sha256']:
                        raise ValueError('Training image changed during read')
                    save_array(path,p)
                    done[row['id']]={'file':str(path.relative_to(output)),'sha256':file_hash(path),'shape':list(shape)}
                    write_json(index_path,index)
                    del p,image
                if model is not None:
                    del model,saved
                    torch.cuda.empty_cache()
            for row in rows:
                path=output/'probabilities'/(row['id']+'.npy')
                if row['id'] in index['combined']:
                    item=index['combined'][row['id']]
                    if item['file'] != str(path.relative_to(output)) or file_hash(path) != item['sha256']:
                        raise ValueError('Combined prediction changed')
                    continue
                total=None
                for number,member in enumerate(recipe['model_files']):
                    item=index['members'][str(number)][row['id']]
                    source=output/item['file'];assert file_hash(source)==item['sha256']
                    p=np.load(source,allow_pickle=False);valid_probabilities(p,tuple(item['shape']))
                    if total is None:total=np.zeros_like(p)
                    total += member['weight']*p
                valid_probabilities(total,tuple(item['shape']))
                save_array(path,total)
                index['combined'][row['id']]={'file':str(path.relative_to(output)),'sha256':file_hash(path),'shape':item['shape']}
                write_json(index_path,index)
            assert {r['id'] for r in rows}==set(index['combined'])
            for source in inputs:
                assert file_hash(ROOT/source['file'])==source['sha256']
            checked_recipe(recipe_path)
            assert file_hash(recipe_path)==config['teacher_recipe_sha256']
            assert all(file_hash(CODE_ROOT/name)==sha for name,sha in source_hashes.items())
            result={'createdAt':now(),'config':config,'config_sha256':file_hash(config_path),
                    'items':[{'id':r['id'],'split':'train','image':source,
                              'probability':index['combined'][r['id']]} for r,source in zip(rows,inputs)],
                    'training_images':len(rows),'member_predictions':len(rows)*len(recipe['model_files']),
                    'mask_read':False,'OOF_predictions':False,'validation_or_test_inference':False}
            write_json(output/'manifest.json',result)
            write_json(output/'status.json',{'state':'completed','training_images':len(rows),
                'manifest_sha256':file_hash(output/'manifest.json'),'updatedAt':now()})
            return result
        except BaseException as exc:
            write_json(output/'status.json',{'state':'failed','error':f'{type(exc).__name__}: {exc}','updatedAt':now()})
            raise
