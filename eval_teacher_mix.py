"""Cache per-member val D4 probabilities once, then score teacher weight mixes
through the champion focused head D4 -> CRF/RankIoU.

  cache --name NAME --file runs/.../best.pth --processing raw|he50 [--scale-norm --base-scale S]
  mix --mixes '{"label": {"NAME": w, ...}, ...}'
Screening only: a winning mix is confirmed with predict_opus_champion.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from core import decode_mask, load_manifest, read_image, score_image, write_json

ROOT = Path(__file__).resolve().parent
HEAD = ROOT / 'runs/target100-focused-warm-uniform-v1/best.pth'


def val_rows(split='val'):
    spec, rows = load_manifest(ROOT / 'data/manifest.json')
    return spec, [r for r in rows if r['split'] == split]


def cache(a):
    from image_processing import enhance
    from scale_norm import predict_normalized
    from opus_models import load_member as load_any
    from target80_methods import predict_d4
    spec, rows = val_rows(a.split)
    out = Path(a.work) / a.name; out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda'); model, _ = load_any(ROOT / a.file, device); scores = []
    for row in rows:
        image = enhance(read_image(row['image']), a.processing)
        with torch.inference_mode():
            p = (predict_normalized(model, image, device, d4=not a.no_d4, base=a.base_scale) if a.scale_norm
                 else predict_d4(model, image, device))
        np.save(out / (row['id'] + '.npy'), p.astype(np.float32), allow_pickle=False)
        if row.get('mask'):
            scores.append(score_image(p.argmax(0).astype(np.uint8), decode_mask(row['mask'], spec), 'skip')['miou'])
    write_json(out / 'member.json', {'file': a.file, 'processing': a.processing, 'scale_norm': a.scale_norm,
                                     'base_scale': a.base_scale, 'split': a.split, 'd4': not a.no_d4, 'd4_miou': float(np.mean(scores)) if scores else None})
    print(json.dumps({a.name: float(np.mean(scores)) if scores else None}), flush=True)


def purpose_metrics(pred, truth):
    """PURPOSE.md metrics: phase area fraction (%) and particle count / median area
    (connected components, 8-neighbour) for primary Si, Al3Ni, eutectic Si."""
    from image_processing import cv2
    out = {'fraction_pred': [], 'fraction_true': [], 'particles': {}}
    for c in range(4):
        out['fraction_pred'].append(float((pred == c).mean() * 100))
        out['fraction_true'].append(float((truth == c).mean() * 100))
    for c, name in ((1, 'primary_si'), (2, 'al3ni'), (3, 'eutectic_si')):
        stats = {}
        for key, m in (('pred', pred), ('true', truth)):
            n, _, st, _ = cv2.connectedComponentsWithStats((m == c).astype(np.uint8), connectivity=8)
            areas = st[1:, 4]
            stats[key] = {'count': int(n - 1), 'median_area': float(np.median(areas)) if len(areas) else 0.}
        out['particles'][name] = stats
    return out


def summarize_purpose(per_image):
    fp = np.array([m['fraction_pred'] for m in per_image]); ft = np.array([m['fraction_true'] for m in per_image])
    err = fp - ft
    summary = {'fraction_mae_pp': np.abs(err).mean(0).round(3).tolist(), 'fraction_bias_pp': err.mean(0).round(3).tolist(),
               'fraction_max_pp': np.abs(err).max(0).round(3).tolist()}
    for name in ('primary_si', 'al3ni', 'eutectic_si'):
        cp = np.array([m['particles'][name]['pred']['count'] for m in per_image], float)
        ct = np.array([m['particles'][name]['true']['count'] for m in per_image], float)
        ap = np.array([m['particles'][name]['pred']['median_area'] for m in per_image])
        at = np.array([m['particles'][name]['true']['median_area'] for m in per_image])
        ok = ct > 0
        summary[name] = {'count_rel_err_pct': float(np.mean(np.abs(cp[ok] - ct[ok]) / ct[ok]) * 100),
                         'median_area_rel_err_pct': float(np.mean(np.abs(ap[ok] - at[ok]) / np.maximum(at[ok], 1)) * 100)}
    return summary


def mix(a):
    from focused_refiner_model import load_focused_refiner
    from focused_refiner_evaluation import decode_probability
    from refiner_d4 import predict_refined_d4
    from image_processing import enhance
    spec, rows = val_rows(); work = Path(a.work)
    # --head: one path, or 'path:weight,path:weight' to average refined probabilities of several heads.
    heads = [(h.rsplit(':', 1)[0], float(h.rsplit(':', 1)[1])) if ':' in h else (h, 1.) for h in a.head.split(',')]
    head_models = [(load_focused_refiner(ROOT / h, 'cuda')[0], w) for h, w in heads]
    results = json.loads((work / 'mixes.json').read_text()) if (work / 'mixes.json').exists() else {}
    for label, weights in json.loads(a.mixes).items():
        assert abs(sum(weights.values()) - 1) < 1e-6, label
        scores, purpose = [], []
        for row in rows:
            teacher = sum(w * np.load(work / n / (row['id'] + '.npy')) for n, w in weights.items()).astype(np.float32)
            original = read_image(row['image'])
            with torch.inference_mode():
                q = sum(w * predict_refined_d4(m, enhance(original, 'he50'), teacher, torch.device('cuda'), 1.)
                        for m, w in head_models) / sum(w for _, w in head_models)
            pred = decode_probability(q.astype(np.float32), original, 'd4_crf')
            truth = decode_mask(row['mask'], spec)
            scores.append({'id': row['id'], 'miou': score_image(pred, truth, 'skip')['miou']})
            purpose.append(purpose_metrics(pred, truth))
        results[label] = {'weights': weights, 'mean_image_miou': float(np.mean([s['miou'] for s in scores])),
                          'purpose': summarize_purpose(purpose), 'images': scores}
        write_json(work / 'mixes.json', results)
        p = results[label]['purpose']
        print(json.dumps({label: results[label]['mean_image_miou'], 'fraction_mae_pp': p['fraction_mae_pp'],
                          'fraction_bias_pp': p['fraction_bias_pp']}), flush=True)


def calibrate(a):
    """PURPOSE plan 3: class-prior correction of the final refined probability.
    Factors are chosen on one half of val and checked on the other half (2-fold)."""
    from focused_refiner_model import load_focused_refiner
    from focused_refiner_evaluation import decode_probability
    from refiner_d4 import predict_refined_d4
    from image_processing import enhance
    spec, rows = val_rows(); work = Path(a.work)
    weights = json.loads(a.weights)
    qdir = work / ('q_' + a.label); qdir.mkdir(exist_ok=True)
    model = None
    for row in rows:
        if (qdir / (row['id'] + '.npy')).exists():
            continue
        if model is None:
            heads = [(h.rsplit(':', 1)[0], float(h.rsplit(':', 1)[1])) if ':' in h else (h, 1.) for h in a.head.split(',')]
            model = [(load_focused_refiner(ROOT / h, 'cuda')[0], w) for h, w in heads]
        teacher = sum(w * np.load(work / n / (row['id'] + '.npy')) for n, w in weights.items()).astype(np.float32)
        with torch.inference_mode():
            original = enhance(read_image(row['image']), 'he50')
            q = sum(w * predict_refined_d4(m, original, teacher, torch.device('cuda'), 1.) for m, w in model) / sum(w for _, w in model)
        np.save(qdir / (row['id'] + '.npy'), q.astype(np.float32), allow_pickle=False)
    band_mode = a.grid == 'band'
    if a.grid == 'al':   # boundary-thinning fix: shrink Al matrix, keep eutectic Si at 1
        grid = [(f0, f2, 1.) for f0 in (1., .9, .8, .7) for f2 in (1., 1.1, 1.2)]
    elif band_mode:      # local fix: thin phases boosted only inside the predicted boundary band
        grid = [(fb, w, 1.) for fb in (1., 1.15, 1.3, 1.5, 1.8) for w in (1, 2)]
    else:
        grid = [(1., f2, f3) for f2 in (1., 1.1, 1.2, 1.35) for f3 in (1., .9, .8)]
    table = {}
    from image_processing import cv2
    for f0, f2, f3 in grid:
        factor = np.array([f0, 1., f2, f3], np.float32)[:, None, None]
        per = []
        for row in rows:
            original = read_image(row['image']); q = np.load(qdir / (row['id'] + '.npy'))
            if band_mode:
                q = q * np.array([1., 1., 1.2, 1.], np.float32)[:, None, None]   # global Al3Ni x1.2 kept
                lab = q.argmax(0).astype(np.uint8)
                edge = np.zeros(lab.shape, np.uint8)
                edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]; edge[1:, :] |= lab[1:, :] != lab[:-1, :]
                band = cv2.dilate(edge, np.ones((2 * f2 + 1, 2 * f2 + 1), np.uint8)) > 0
                boost = np.ones_like(q); boost[1:, band] = f0     # primary Si, Al3Ni, eutectic Si
                q = q * boost
            else:
                q = q * factor
            q /= q.sum(0, keepdims=True)
            pred = decode_probability(q.astype(np.float32), original, 'd4_crf')
            truth = decode_mask(row['mask'], spec)
            per.append({'id': row['id'], 'small': original.shape[1] < 800,
                        'miou': score_image(pred, truth, 'skip')['miou'], **purpose_metrics(pred, truth)})
        key = f'band{f0}_w{f2}' if band_mode else (f'{f2}_{f3}' if a.grid != 'al' else f'al{f0}_{f2}')
        table[key] = per
        print(json.dumps({'al': f0, 'al3ni': f2, 'eutectic': f3, 'miou': float(np.mean([p['miou'] for p in per])),
                          'bias': (np.array([p['fraction_pred'] for p in per]) - np.array([p['fraction_true'] for p in per])).mean(0).round(3).tolist()}), flush=True)
    # Stratified halves by resolution group, fixed order.
    small = [r['id'] for r in rows if read_image(r['image']).shape[1] < 800]; large = [r['id'] for r in rows if r['id'] not in small]
    halves = [set(small[0::2] + large[0::2]), set(small[1::2] + large[1::2])]

    def stats(per, ids):
        sel = [p for p in per if p['id'] in ids]
        err = np.array([np.array(p['fraction_pred']) - np.array(p['fraction_true']) for p in sel])
        return {'miou': float(np.mean([p['miou'] for p in sel])), 'bias': err.mean(0).tolist(), 'mae': np.abs(err).mean(0).tolist()}

    def objective(s, base):
        # Purpose rule: reduce Al3Ni/eutectic bias without losing more than 0.05%p mIoU.
        if s['miou'] < base['miou'] - .0005:
            return -1e9
        if a.grid in ('al', 'band'):   # boundary fix is judged on mIoU first (thickness restores IoU)
            return s['miou'] - .001 * (abs(s['bias'][2]) + abs(s['bias'][3]))
        return -(abs(s['bias'][2]) + abs(s['bias'][3]))

    report = {'grid': {k: {'all': stats(v, {r['id'] for r in rows})} for k, v in table.items()}, 'folds': []}
    for pick, check in ((0, 1), (1, 0)):
        ref = 'band1.0_w1' if band_mode else ('1.0_1.0' if a.grid != 'al' else 'al1.0_1.0')
        base = stats(table[ref], halves[pick])
        best = max(table, key=lambda k: objective(stats(table[k], halves[pick]), base))
        report['folds'].append({'chosen_on': pick, 'factors': best, 'pick_half': stats(table[best], halves[pick]),
                                'check_half_before': stats(table[ref], halves[check]),
                                'check_half_after': stats(table[best], halves[check])})
    write_json(work / f'calibration-{a.label}{"" if a.grid == "class" else "-" + a.grid}.json', report)
    print(json.dumps(report['folds'], indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['cache', 'mix', 'calibrate'])
    p.add_argument('--work', default='/dev/shm/komap-opus-eval/members')
    p.add_argument('--name'); p.add_argument('--file'); p.add_argument('--processing', default='raw')
    p.add_argument('--scale-norm', action='store_true'); p.add_argument('--base-scale', type=float, default=1.)
    p.add_argument('--mixes'); p.add_argument('--weights'); p.add_argument('--label'); p.add_argument('--grid', default='class')
    p.add_argument('--split', default='val')
    p.add_argument('--no-d4', action='store_true', help='single-orientation cache (fast; for train-split teacher inputs)')
    p.add_argument('--head', default='runs/target100-focused-warm-uniform-v1/best.pth')
    a = p.parse_args(); torch.set_num_threads(4)
    {'cache': cache, 'mix': mix, 'calibrate': calibrate}[a.phase](a)
