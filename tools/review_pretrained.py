"""Reproduce all saved models, check savepoint histories, fixed TTA/ensemble."""
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from core import decode_mask,file_hash,load_manifest,read_image,save_mask,score_image,write_json
from context_model import evaluate_context,predict_probabilities,summarize_scores
from pretrained_model import load_saved
from pretrained_inference import predict_tta4
from run import now

REVIEW=ROOT/'review/pretrained'


def score_probabilities(name,rows,spec,producer):
    output=REVIEW/'predictions'/name
    output.mkdir(parents=True,exist_ok=False)
    scores,alternatives,predictions=[],{p:[] for p in ['skip','one','zero']},[]
    for row in rows:
        if row['split']!='val':
            continue
        probs=producer(row)
        if not np.isfinite(probs).all() or (probs<0).any():
            raise ValueError('Invalid probabilities')
        np.testing.assert_allclose(probs.sum(0),1,atol=1e-5)
        pred=probs.argmax(0).astype(np.uint8)
        truth=decode_mask(row['mask'],spec)
        scores.append({'id':row['id'],**score_image(pred,truth,spec['absent_class_policy'])})
        for policy in alternatives:
            alternatives[policy].append(score_image(pred,truth,policy)['miou'])
        path=output/(row['id']+'.png')
        save_mask(path,pred,spec)
        np.testing.assert_array_equal(pred,decode_mask(path,spec))
        predictions.append({'id':row['id'],'file':str(path.relative_to(ROOT)),'sha256':file_hash(path)})
    return summarize_scores(scores,alternatives,spec),predictions


def main():
    torch.set_num_threads(4)
    device=torch.device('cuda')
    spec,rows=load_manifest(ROOT/'data/manifest.json')
    state=json.loads((REVIEW/'queue-status.json').read_text())
    if state['state']!='reproducing':
        raise RuntimeError('Training queue must finish all planned models first')
    old=json.loads((ROOT/'review/context/comparison.json').read_text())
    baseline=next(r for r in old['records'] if r['id']==old['best_observed'])
    baseline={**baseline,'label':'이전 최고 · 4개 결합','stage':'baseline'}
    records=[baseline]
    savepoint_checks=[]
    for name in state['completed']:
        folder=ROOT/'runs'/name
        status=json.loads((folder/'status.json').read_text())
        if status['state']!='completed':
            raise RuntimeError('Unfinished experiment: '+name)
        assert file_hash(folder/'best.pth')==status['weights_sha256']
        model,saved=load_saved(folder/'best.pth',device)
        assert saved['config']['manifest_sha256']==file_hash(ROOT/'data/manifest.json')
        assert saved['classes']==spec['classes']
        previous=-1.
        saves=[]
        for line in (folder/'savepoints.jsonl').read_text().splitlines():
            row=json.loads(line)
            assert row['miou']>previous
            assert file_hash(folder/row['file'])==row['sha256']
            assert row['includes_optimizer_scheduler_rng_sampler']
            previous=row['miou'];saves.append(row)
        assert previous==saved['validation']['mean_image_miou']
        savepoint_checks.append({'run':name,'savepoints':saves,'monotonic_best_and_hashes_verified':True})
        output=REVIEW/'predictions'/name
        start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        metrics=evaluate_context(model,rows,spec,device,output=output,probability_output=REVIEW/'probabilities'/name)
        prior={r['id']:r for r in saved['validation']['images']}
        delta=max(abs(r['miou']-prior[r['id']]['miou']) for r in metrics['images'])
        if delta>1e-6:
            raise RuntimeError('Saved model reproduction mismatch: '+name)
        history=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
        init=saved['config']['initialization']
        stage='pilot' if '-pilot-' in name else 'repeat' if '-repeat-' in name else 'long'
        label={'scratch':'처음부터 학습','imagenet':'ImageNet 사전학습','micronet':'MicroNet 사전학습'}[init]
        label+=' · '+{'pilot':'1,200스텝','long':'6,000스텝 A','repeat':'6,000스텝 B'}[stage]
        record={'id':name,'kind':'single','stage':stage,'label':label,'metrics':metrics,'config':saved['config'],
                'best_epoch':saved['epoch'],'history':history,'weights_sha256':status['weights_sha256'],
                'initialization_audit':saved['initialization_audit'],'savepoint_count':len(saves),
                'reproduction_max_delta':delta,'training_validation_seconds':sum(h['seconds'] for h in history),
                'training_peak_vram_mib':max(h['peak_vram_mib'] for h in history),
                'inference_peak_vram_mib':torch.cuda.max_memory_allocated()/2**20,
                'evaluation_and_save_seconds':time.monotonic()-start,
                'predictions':[{'id':r['id'],'file':str((output/(r['id']+'.png')).relative_to(ROOT)),
                                'sha256':file_hash(output/(r['id']+'.png'))} for r in metrics['images']]}
        write_json(REVIEW/'singles'/(name+'.json'),record);records.append(record)
        del model,saved
        torch.cuda.empty_cache()
        print(json.dumps({'reproduced':name,'miou':metrics['mean_image_miou'],'delta':delta}),flush=True)
    selected=state['selected']['selected_pretrained']
    members=[f'pretrain-{selected}-long-v1',f'pretrain-{selected}-repeat-v1']
    name='pretrain-ensemble-2'
    def ensemble(row):
        p=[np.load(REVIEW/'probabilities'/m/(row['id']+'.npy'),allow_pickle=False) for m in members]
        return (p[0]+p[1])/2
    metrics,predictions=score_probabilities(name,rows,spec,ensemble)
    records.append({'id':name,'kind':'ensemble','stage':'inference','label':'사전학습 2개 결합',
                    'members':members,'weights':[.5,.5],'metrics':metrics,'predictions':predictions})
    best_single=max((r for r in records if r['kind']=='single'),key=lambda r:r['metrics']['mean_image_miou'])
    model,_=load_saved(ROOT/'runs'/best_single['id']/'best.pth',device)
    name='pretrain-best-tta4'
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    metrics,predictions=score_probabilities(name,rows,spec,
        lambda row:predict_tta4(model,read_image(row['image']),device))
    records.append({'id':name,'kind':'tta','stage':'inference','label':'최고 단일 모델 · 4방향 보기',
                    'members':[best_single['id']],'transforms':['identity','flip_x','flip_y','flip_xy'],
                    'metrics':metrics,'predictions':predictions,'evaluation_and_save_seconds':time.monotonic()-start,
                    'inference_peak_vram_mib':torch.cuda.max_memory_allocated()/2**20})
    del model
    torch.cuda.empty_cache()
    best=max(records,key=lambda r:r['metrics']['mean_image_miou'])
    pairs=[]
    by_id={r['id']:r for r in records}
    for stage,seed in [('long',20261007),('repeat',20261008)]:
        a,b=f'pretrain-scratch-{stage}-v1',f'pretrain-{selected}-{stage}-v1'
        pairs.append({'seed':seed,'control':a,'pretrained':b,'control_miou':by_id[a]['metrics']['mean_image_miou'],
                      'pretrained_miou':by_id[b]['metrics']['mean_image_miou'],
                      'difference_pp':100*(by_id[b]['metrics']['mean_image_miou']-by_id[a]['metrics']['mean_image_miou'])})
    comparison={'date':now(),'records':records,'baseline':baseline['id'],'best_single':best_single['id'],
                'best_observed':best['id'],'selection':state['selected'],'paired_seeds':pairs,
                'classes':spec['classes'],'images':old['images'],'validation_images':20,'official_train_images':70,
                'target_miou':.8,'validation_reused_for_selection':True,'independent_final_test_score':None,
                'test_images_used_for_selection':0,'contest_external_pretraining_and_ensemble_permission':'unconfirmed'}
    write_json(REVIEW/'savepoint-checks.json',{'passed':True,'runs':savepoint_checks})
    write_json(REVIEW/'comparison.json',comparison)
    # A candidate recipe is also an immutable savepoint for ensemble/TTA behavior.
    recipe={'candidate':best['id'],'record':best,'classes':spec['classes'],
            'manifest_sha256':file_hash(ROOT/'data/manifest.json'),'model_files':[],
            'status':'reproduced_on_validation_not_submitted','updatedAt':now()}
    members=best.get('members',[best['id']])
    if best['id']==baseline['id']:
        recipe['baseline_recipe']=json.loads((REVIEW/'baseline-savepoint.json').read_text())
    else:
        for member in members:
            folder=ROOT/'runs'/member
            saves=[json.loads(line) for line in (folder/'savepoints.jsonl').read_text().splitlines()]
            recipe['model_files'].append({'id':member,'file':str((folder/'best.pth').relative_to(ROOT)),
                'sha256':file_hash(folder/'best.pth'),'resume_savepoint':str((folder/saves[-1]['file']).relative_to(ROOT)),
                'resume_sha256':saves[-1]['sha256']})
    write_json(REVIEW/'candidates'/f'{best["id"]}.json',recipe)
    write_json(REVIEW/'champion.json',{'id':best['id'],'mean_image_miou':best['metrics']['mean_image_miou'],
               'savepoint':str((REVIEW/'candidates'/f'{best["id"]}.json').relative_to(ROOT)),
               'sha256':file_hash(REVIEW/'candidates'/f'{best["id"]}.json'),
               'status':'reproduced_on_validation_not_submitted','updatedAt':now()})
    with (REVIEW/'champion-history.jsonl').open('a') as out:
        out.write((REVIEW/'champion.json').read_text().replace('\n','')+'\n')
    print(json.dumps({'best':best['id'],'miou':best['metrics']['mean_image_miou'],'paired':pairs}),flush=True)


if __name__=='__main__':
    main()
