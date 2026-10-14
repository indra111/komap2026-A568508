"""Validated candidate recipes for continued experiments, sharing a fixed metric."""
import fcntl
import json
from pathlib import Path
import sys
import numpy as np
import torch
from core import load_manifest,file_hash,read_image,write_json
from context_model import evaluate_context
from pretrained_model import load_saved
from advanced_model import load_advanced,ARCHITECTURES
from image_processing import enhance
from target80_methods import predict_d4
from run import now
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import review_pretrained as scoring
REVIEW=ROOT/'review/target80-iterations'
scoring.REVIEW=REVIEW


def load_any(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    architecture=saved['architecture'];del saved
    if architecture=='frozen_dinov2_base_patch_refiner_bf16':
        from patch_refiner import load_patch_refiner
        return load_patch_refiner(path,device)
    if architecture in ('dpt_dinov2_base_frozen_detail_bf16', 'dpt_dinov2_base_rein_detail_bf16'):
        from rein_model import load_rein_dino
        return load_rein_dino(path,device)
    if architecture=='dpt_dinov2_large_detail_bf16':
        from dinov2_large_model import load_large_dino
        return load_large_dino(path,device)
    if architecture=='hrnet_w48_ocr_detail_bf16':
        from hrnet_ocr_model import load_hrnet
        return load_hrnet(path,device)
    if architecture=='dpt_dinov3_base_detail_bf16':
        from dinov3_model import load_dinov3
        return load_dinov3(path,device)
    if architecture=='dpt_dinov2_small_pair_bf16':
        from dino_pair_model import load_pair_dino
        return load_pair_dino(path,device)
    if architecture=='dpt_dinov2_small_texture_bf16':
        from dino_texture_model import load_texture_dino
        return load_texture_dino(path,device)
    if architecture=='dpt_dinov2_base_detail_bf16':
        from large_dino_model import load_base_dino
        return load_base_dino(path,device)
    if architecture=='dpt_dinov2_small_detail_bf16':
        from stable_dino_model import load_stable_dino
        return load_stable_dino(path,device)
    if architecture=='unet_convnext_base22k':
        from large_model import load_large
        return load_large(path,device)
    if architecture=='unet_convnext_tiny_pixel_contrast':
        from pixel_contrast import load_contrast
        return load_contrast(path,device)
    if architecture in ['dpt_dinov2_small','dpt_dinov2_small_detail']:
        from dino_model import load_dino
        return load_dino(path,device)
    if architecture=='normalized_unet_resnet34_local_global':
        from pretrained_context import load_pretrained_context
        return load_pretrained_context(path,device)
    if architecture=='normalized_unet_resnet34_shape_aux':
        from shape_model import load_shape
        return load_shape(path,device)
    if architecture=='normalized_unet_resnet34_native_detail':
        from native_detail import load_detail
        return load_detail(path,device)
    return load_advanced(path,device) if architecture in ARCHITECTURES else load_saved(path,device)


def register(record):
    path=REVIEW/'records'/(record['id']+'.json')
    if path.exists():raise FileExistsError(path)
    write_json(path,record)
    spec,_=load_manifest(ROOT/'data/manifest.json')
    recipe=REVIEW/'candidates'/(record['id']+'.json')
    extra_sources = ['scaled_inference.py'] if any(f.get('scale',1) != 1 for f in record['model_files']) else []
    if any(f.get('architecture') == 'frozen_dinov2_base_patch_refiner_bf16' for f in record['model_files']):
        extra_sources.append('patch_refiner.py')
    if any('phase_offset' in f for f in record['model_files']):
        extra_sources.append('phase_inference.py')
    if any(f.get('architecture') in ('dpt_dinov2_base_frozen_detail_bf16', 'dpt_dinov2_base_rein_detail_bf16') for f in record['model_files']):
        extra_sources.extend(['rein_model.py', 'rein_adapter.py',
            'third_party/rein-reference/LICENSE.txt', 'third_party/rein-reference/NOTICE.md',
            'third_party/rein-reference/reins-upstream.py', 'third_party/rein-reference/source-manifest.json',
            'third_party/rein-reference/DINOv2-LICENSE.txt',
            'data/pretrained/dinov2-base/README.md', 'data/pretrained/dinov2-base/manifest.json'])
    if any(f.get('architecture') == 'dpt_dinov2_large_detail_bf16' for f in record['model_files']):
        extra_sources.extend(['dinov2_large_model.py', 'data/pretrained/dinov2-large/LICENSE',
                              'data/pretrained/dinov2-large/NOTICE.md'])
    if any(f.get('architecture') == 'hrnet_w48_ocr_detail_bf16' for f in record['model_files']):
        extra_sources.extend(['hrnet_ocr_model.py', 'data/pretrained/hrnet-w48/LICENSE',
                              'data/pretrained/hrnet-w48/NOTICE.md'])
    if any(f.get('architecture') in ('dpt_dinov2_small_texture_bf16', 'dpt_dinov2_small_pair_bf16') for f in record['model_files']):
        extra_sources.append('dino_texture_model.py')
    if any(f.get('architecture') == 'dpt_dinov2_small_pair_bf16' for f in record['model_files']):
        extra_sources.append('dino_pair_model.py')
    if any(f.get('architecture') == 'dpt_dinov3_base_detail_bf16' for f in record['model_files']):
        extra_sources.extend(['dinov3_model.py', 'data/pretrained/dinov3-base/LICENSE.md',
                              'data/pretrained/dinov3-base/NOTICE.md'])
    write_json(recipe,{'candidate':record['id'],'record':record,'model_files':record['model_files'],'classes':spec['classes'],
        'original_manifest_sha256':file_hash(ROOT/'data/manifest.json'),'status':'reproduced_validation_not_official',
        'inference_source_sha256':{n:file_hash(ROOT/n) for n in ['target80_methods.py','advanced_model.py','native_detail.py','shape_model.py','pretrained_context.py','dino_model.py','stable_dino_model.py','large_dino_model.py','pixel_contrast.py','large_model.py','pretrained_model.py','image_processing.py','context_model.py','core.py'] + extra_sources}})
    with (REVIEW/'.champion.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        old=json.loads((REVIEW/'champion.json').read_text())
        if record['metrics']['mean_image_miou']>=old['mean_image_miou']:
            value={'id':record['id'],'mean_image_miou':record['metrics']['mean_image_miou'],
                'savepoint':str(recipe.relative_to(ROOT)),'sha256':file_hash(recipe),'status':'reproduced_validation_not_official','updatedAt':now()}
            write_json(REVIEW/'champion.json',value)
            with (REVIEW/'champion-history.jsonl').open('a') as out:out.write(json.dumps(value)+'\n')
    print(json.dumps({'candidate':record['id'],'miou':record['metrics']['mean_image_miou']}),flush=True)
    return record


def evaluate_run(name,device):
    folder=ROOT/'runs'/name
    status=json.loads((folder/'status.json').read_text());assert status['state']=='completed'
    model,saved=load_any(folder/'best.pth',device)
    assert file_hash(folder/'best.pth')==status['weights_sha256']
    processing='he50'
    manifest=ROOT/'data/processed/he50/manifest.json'
    assert saved['config']['manifest_sha256']==file_hash(manifest)
    spec,rows=load_manifest(manifest);assert saved['classes']==spec['classes']
    patch=saved['config']['patch'];overlap=saved['config']['overlap']
    metrics=evaluate_context(model,rows,spec,device,patch=patch,overlap=overlap,
        output=REVIEW/'predictions'/name,probability_output=REVIEW/'probabilities'/name)
    old={r['id']:r for r in saved['validation']['images']}
    delta=max(abs(r['miou']-old[r['id']]['miou']) for r in metrics['images']);assert delta<=1e-6
    saves=[json.loads(x) for x in (folder/'savepoints.jsonl').read_text().splitlines()]
    previous=-1
    for s in saves:
        assert s['miou']>previous and file_hash(folder/s['file'])==s['sha256'];previous=s['miou']
    full=torch.load(folder/saves[-1]['file'],map_location='cpu',weights_only=True)
    assert all(k in full for k in ['optimizer','scheduler','scaler','sampler','torch_rng','cuda_rng']);del full
    member={'id':name,'file':str((folder/'best.pth').relative_to(ROOT)),'sha256':file_hash(folder/'best.pth'),
        'architecture':saved['architecture'],'processing':processing,'tta':'identity','weight':1.,'patch':patch,'overlap':overlap,
        'resume_savepoint':str((folder/saves[-1]['file']).relative_to(ROOT)),'resume_sha256':saves[-1]['sha256']}
    record=register({'id':name,'kind':'single','model_files':[member],'metrics':metrics,'config':saved['config'],
                     'reproduction_max_delta':delta,'savepoints':saves})
    del model,saved;torch.cuda.empty_cache()
    return record


def d4(name,member,device):
    spec,rows=load_manifest(ROOT/'data/manifest.json')
    assert file_hash(ROOT/member['file'])==member['sha256']
    model,_=load_any(ROOT/member['file'],device)
    cache=REVIEW/'probabilities'/name;cache.mkdir(parents=True,exist_ok=False)
    def produce(row):
        p=predict_d4(model,enhance(read_image(row['image']),member['processing']),device,
            patch=member.get('patch',512),overlap=member.get('overlap',128))
        np.save(cache/(row['id']+'.npy'),p,allow_pickle=False)
        return p
    metrics,predictions=scoring.score_probabilities(name,rows,spec,produce)
    record=register({'id':name,'kind':'d4','metrics':metrics,'predictions':predictions,
                     'model_files':[{**member,'tta':'d4','weight':1.}]})
    del model;torch.cuda.empty_cache()
    return record


def ensemble(name,members):
    for member in members:
        record=json.loads((REVIEW/'records'/(member+'.json')).read_text())
        if record.get('kind')=='spatial_prior' or record.get('postprocessing'):
            raise ValueError('Nonlinear spatial candidates cannot be flattened into arithmetic ensembles')
    spec,rows=load_manifest(ROOT/'data/manifest.json')
    cache=REVIEW/'probabilities'/name;cache.mkdir(parents=True,exist_ok=False)
    def produce(row):
        p=sum(np.load(REVIEW/'probabilities'/m/(row['id']+'.npy'),allow_pickle=False) for m in members)/len(members)
        np.save(cache/(row['id']+'.npy'),p,allow_pickle=False)
        return p
    metrics,predictions=scoring.score_probabilities(name,rows,spec,produce)
    files=[]
    for m in members:
        for f in json.loads((REVIEW/'records'/(m+'.json')).read_text())['model_files']:
            files.append({**f,'weight':f['weight']/len(members)})
    return register({'id':name,'kind':'ensemble','members':members,'metrics':metrics,'predictions':predictions,'model_files':files})
