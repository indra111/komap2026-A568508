"""Reproduce a reviewed candidate offline and export native four-color masks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from core import load_manifest,file_hash,read_image,save_mask,decode_mask,score_image,write_json
from context_model import predict_probabilities,summarize_scores
from target80_methods import predict_d4
from target80_evaluation import load_any
from image_processing import enhance
from run import now
ROOT=Path(__file__).resolve().parent
REVIEW=ROOT/'review/target80-iterations'
PHASE_INFERENCE_VERSION = 1


def checked_recipe(path):
    recipe=json.loads(Path(path).read_text())
    if recipe.get('kind')=='spatial_prior' or recipe.get('cannot_flatten_into_arithmetic_ensemble'):
        raise ValueError('This nonlinear candidate requires its dedicated inference entrypoint')
    if not recipe['status'].startswith('reproduced'):
        raise ValueError('Candidate has not been reproduced')
    if recipe['original_manifest_sha256']!=file_hash(ROOT/'data/manifest.json'):
        raise ValueError('Original manifest changed')
    for name,digest in recipe.get('inference_source_sha256',{}).items():
        if file_hash(ROOT/name)!=digest:raise ValueError('Inference source changed: '+name)
    members=recipe['model_files']
    if not members or any(f.get('weight',0)<=0 for f in members) or abs(sum(f['weight'] for f in members)-1)>1e-8:
        raise ValueError('Invalid ensemble weights')
    for f in members:
        path=(ROOT/f['file']).resolve()
        if not path.is_relative_to(ROOT) or file_hash(path)!=f['sha256']:
            raise ValueError('Weights path or hash mismatch')
        if f.get('tta','identity') not in ('identity','d4'):raise ValueError('Unknown inference transform')
        if 'scale' in f:
            from scaled_inference import validate_scale
            validate_scale(f['scale'])
            if f['scale'] != 1 and 'scaled_inference.py' not in recipe.get('inference_source_sha256', {}):
                raise ValueError('Scaled inference source is not frozen in this recipe')
        if 'phase_offset' in f:
            from phase_inference import validate_offset
            validate_offset(f['phase_offset'])
            if f.get('scale', 1) != 1:
                raise ValueError('Combined scale and phase inference is not supported')
            if 'phase_inference.py' not in recipe.get('inference_source_sha256', {}):
                raise ValueError('Phase inference source is not frozen in this recipe')
    return recipe


def predict_member(model, image, device, member, patch, overlap):
    if 'phase_offset' in member:
        if member.get('scale', 1) != 1:
            raise ValueError('Combined scale and phase inference is not supported')
        from phase_inference import predict_phase
        return predict_phase(model, image, device, offset=member['phase_offset'],
                             tta=member.get('tta', 'identity'), patch=patch, overlap=overlap)
    if member.get('scale', 1) != 1:
        from scaled_inference import predict_scaled
        return predict_scaled(model, image, device, scale=member['scale'],
                              tta=member.get('tta', 'identity'), patch=patch, overlap=overlap)
    predictor = predict_d4 if member.get('tta') == 'd4' else predict_probabilities
    return predictor(model, image, device, patch=patch, overlap=overlap)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',default='target80-d4-mixed4')
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args();torch.set_num_threads(4)
    recipe=checked_recipe(REVIEW/'candidates'/(a.candidate+'.json'))
    spec,rows=load_manifest(ROOT/'data/manifest.json');assert recipe['classes']==spec['classes']
    rows=[r for r in rows if r['split']==a.split]
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    device=torch.device(a.device);begin=time.monotonic()
    total={}
    for f in recipe['model_files']:
        model,saved=load_any(ROOT/f['file'],device);assert saved['classes']==spec['classes']
        patch=f.get('patch',saved['config']['patch']);overlap=f.get('overlap',saved['config']['overlap'])
        for row in rows:
            image=enhance(read_image(row['image']),f['processing'])
            probs=predict_member(model,image,device,f,patch,overlap)
            if row['id'] not in total:total[row['id']]=np.zeros_like(probs)
            total[row['id']]+=f['weight']*probs
        del model,saved;torch.cuda.empty_cache()
    predictions=[];scores=[];alternative={k:[] for k in ['skip','one','zero']}
    for row in rows:
        probs=total[row['id']];assert np.isfinite(probs).all()
        np.testing.assert_allclose(probs.sum(0),1,atol=1e-5)
        pred=probs.argmax(0).astype(np.uint8);path=output/(row['id']+'.png')
        assert pred.shape==read_image(row['image']).shape[:2]
        save_mask(path,pred,spec);np.testing.assert_array_equal(decode_mask(path,spec),pred)
        predictions.append({'id':row['id'],'file':path.name,'sha256':file_hash(path),'shape':list(pred.shape)})
        if a.split=='val':
            truth=decode_mask(row['mask'],spec)
            scores.append({'id':row['id'],**score_image(pred,truth,spec['absent_class_policy'])})
            for rule in alternative:alternative[rule].append(score_image(pred,truth,rule)['miou'])
    result={'candidate':a.candidate,'split':a.split,'predictions':predictions,'model_files':recipe['model_files'],
        'seconds':time.monotonic()-begin,'not_submitted':True,'createdAt':now()}
    if scores:
        metrics=summarize_scores(scores,alternative,spec)
        reference={r['id']:r for r in recipe['record']['metrics']['images']}
        delta=max(abs(r['miou']-reference[r['id']]['miou']) for r in scores)
        if delta>1e-6:raise ValueError('Independent recipe reproduction differs: '+str(delta))
        result.update(metrics=metrics,reproduction_max_delta=delta)
    write_json(output/'prediction-manifest.json',result)
    print(json.dumps({'candidate':a.candidate,'split':a.split,'images':len(rows),'seconds':result['seconds'],
                      'miou':result.get('metrics',{}).get('mean_image_miou')}))


if __name__=='__main__':main()
