"""Fresh raw-image inference for recipe-defined teacher ensembles + focused head.

Recipe JSON: {"members": [{"file", "weight", "processing", "scale_norm": bool,
"base_scale": float}], "head": path, "mode": "d4_crf", "class_factors": [4 floats, optional],
"psi_min_area": float, optional, "heads": [{"file", "weight"}] optional (replaces "head"), "crf": DenseCRF parameter dict optional
(replaces the micro_edges preset)}.
class_factors rescale the refined probability per class before CRF/RankIoU
(Al3Ni under-estimation correction, chosen by split-half validation). psi_min_area relabels
primary-Si components smaller than that area (in x1.5 scale-normalized px) as eutectic Si:
only 1.3% of train70 primary-Si particles are under 100 px vs 48% of eutectic Si. Teachers run in a
default-runtime subprocess; the strict-runtime head is applied afterwards.
Works for val (scored on official masks) and test (masks only written).
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from core import decode_mask, file_hash, load_manifest, read_image, save_mask, score_image, write_json

ROOT = Path(__file__).resolve().parent


def select_rows(split, images=None):
    """Manifest rows of one split, or every PNG/JPG/TIF in an arbitrary folder (images mode, unscored)."""
    if images:
        files = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        return [{'id': p.stem, 'image': str(p), 'split': 'images'} for p in files]
    _, rows = load_manifest(ROOT / 'data/manifest.json')
    return [r for r in rows if r['split'] == split]


def mask_name(row):
    stem = Path(row['image']).stem
    return (stem[:-len('_image')] if stem.endswith('_image') else stem) + '_mask.png'


def teacher_only(recipe, split, work, images=None):
    from image_processing import enhance
    from scale_norm import predict_normalized
    from opus_models import load_member as load_any
    from target80_methods import predict_d4
    rows = select_rows(split, images)
    device = torch.device('cuda'); total = {}
    for member in recipe['members']:
        model, _ = load_any(ROOT / member['file'], device)
        for row in rows:
            image = enhance(read_image(row['image']), member['processing'])
            with torch.inference_mode():
                p = (predict_normalized(model, image, device, d4=True, base=member.get('base_scale', 1.))
                     if member.get('scale_norm') else predict_d4(model, image, device))
            total[row['id']] = total.get(row['id'], 0) + member['weight'] * p
        print(json.dumps({'teacher_member_done': member['file']}), flush=True)
        del model; torch.cuda.empty_cache()
    for key, value in total.items():
        np.save(work / (key + '.npy'), value.astype(np.float32), allow_pickle=False)


def drop_small_primary_si(pred, min_area):
    from image_processing import cv2
    from scale_norm import factor_for
    f = factor_for(pred.shape, 1.5)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((pred == 1).astype(np.uint8), connectivity=8)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] * f * f < min_area)
    small = small[small > 0]
    out = pred.copy()
    out[np.isin(lab, small)] = 3
    return out


def main(a):
    recipe = json.loads(Path(a.recipe).read_text())
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    if a.teacher_only:
        return teacher_only(recipe, a.split, work, a.images)
    begin = time.monotonic()
    subprocess.run([sys.executable, '-u', __file__, '--recipe', a.recipe, '--split', a.split, '--work', str(work),
                    '--output', a.output, '--teacher-only', *(['--images', a.images] if a.images else [])], cwd=ROOT, check=True)
    from focused_refiner_model import load_focused_refiner
    from focused_refiner_evaluation import decode_probability
    from refiner_d4 import predict_refined_d4
    from image_processing import enhance
    spec, _ = load_manifest(ROOT / 'data/manifest.json')
    rows = select_rows(a.split, a.images)
    # 'heads': [{"file", "weight"}] averages the refined probabilities of several heads; 'head' is one path.
    heads = recipe.get('heads') or [{'file': recipe['head'], 'weight': 1.}]
    head_models = [(load_focused_refiner(ROOT / h['file'], 'cuda')[0], h['weight']) for h in heads]
    output = Path(a.output); output.mkdir(parents=True, exist_ok=False)
    scores, predictions = [], []
    for row in rows:
        original = read_image(row['image'])
        teacher = np.load(work / (row['id'] + '.npy'))
        with torch.inference_mode():
            q = sum(w * predict_refined_d4(m, enhance(original, 'he50'), teacher, torch.device('cuda'), 1.)
                    for m, w in head_models) / sum(w for _, w in head_models)
        q = q.astype(np.float32)
        if recipe.get('class_factors'):
            q = q * np.asarray(recipe['class_factors'], np.float32)[:, None, None]
            q = (q / q.sum(0, keepdims=True)).astype(np.float32)
        if recipe.get('crf'):   # explicit CRF parameters (re-tuned for the current teacher), then RankIoU
            from dense_crf import refine
            from rank_iou import rank_iou
            pred = rank_iou(refine(q, original, recipe['crf']), pruning_prob=recipe.get('rank_prune', .5))[0].astype(np.uint8)
        else:
            pred = decode_probability(q, original, recipe.get('mode', 'd4_crf'))
        if recipe.get('psi_min_area'):
            pred = drop_small_primary_si(pred, recipe['psi_min_area'])
        path = output / mask_name(row)
        save_mask(path, pred, spec)
        predictions.append({'id': row['id'], 'file': path.name, 'sha256': file_hash(path)})
        if a.split == 'val' and not a.images:
            scores.append({'id': row['id'], **score_image(pred, decode_mask(row['mask'], spec), 'skip')})
    result = {'recipe': recipe, 'recipe_sha256': file_hash(a.recipe), 'split': a.split,
              'predictions': predictions, 'seconds': time.monotonic() - begin}
    if scores:
        result['mean_image_miou'] = float(np.mean([s['miou'] for s in scores]))
        result['images'] = scores
    write_json(output / 'prediction-manifest.json', result)
    print(json.dumps({'split': a.split, 'miou': result.get('mean_image_miou'), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', required=True); p.add_argument('--output', required=True)
    p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--work', required=True); p.add_argument('--teacher-only', action='store_true')
    p.add_argument('--images', default=None, help='segment every image in this folder instead of a manifest split (no scoring)')
    a = p.parse_args(); torch.set_num_threads(4); main(a)
