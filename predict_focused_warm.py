"""Fresh raw-image inference for the Stage45 focused-warm candidate.

The original three-model teacher runs in a separate default-runtime process.
Its arrays must exactly equal the frozen validation cache before the strict
runtime head is applied. Saved Stage45 outputs are compared only afterwards.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from core import file_hash, load_manifest, read_image, decode_mask, save_mask, score_image, write_json
from run import now
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def checked_recipe(path):
    recipe = read(path)
    assert recipe['kind'] == 'focused_warm_refiner' and recipe['cannot_flatten_into_arithmetic_ensemble']
    assert recipe['branch'] == 'control' and recipe['mode'] == 'd4_crf' and recipe['strength'] == 1.
    for name, sha in recipe['inference_source_sha256'].items():
        source = (ROOT/name).resolve()
        assert source.is_relative_to(ROOT) and file_hash(source) == sha, name
    teacher_path = ROOT/recipe['teacher_recipe']['file']
    assert file_hash(teacher_path) == recipe['teacher_recipe']['sha256']
    assert file_hash(ROOT/recipe['record']['file']) == recipe['record']['sha256']
    return recipe, read(teacher_path)


def teacher_only(recipe_file, output, split, device):
    # Default runtime only: the strict focused-head runtime is never imported here.
    from predict_target80 import predict_member
    from target80_evaluation import load_any
    from image_processing import enhance
    assert not torch.are_deterministic_algorithms_enabled()
    recipe, teacher = checked_recipe(ROOT/recipe_file)
    spec, rows = load_manifest(ROOT/'data/manifest.json'); rows = [r for r in rows if r['split'] == split]
    output = Path(output); output.mkdir(exist_ok=False); device = torch.device(device); total = {}
    for index, member in enumerate(teacher['model_files']):
        model, saved = load_any(ROOT/member['file'], device)
        for i, row in enumerate(rows, 1):
            with torch.inference_mode():
                p = predict_member(model, enhance(read_image(row['image']), member['processing']), device, member,
                    member.get('patch', saved['config']['patch']), member.get('overlap', saved['config']['overlap']))
            if row['id'] not in total: total[row['id']] = np.zeros_like(p)
            total[row['id']] += member['weight']*p
            print(json.dumps({'phase': 'fresh_teacher', 'member': index, 'image': i, 'total': len(rows)}), flush=True)
        del model, saved; torch.cuda.empty_cache()
    items = []
    for row in rows:
        path = output/(row['id']+'.npy'); np.save(path, total[row['id']], allow_pickle=False)
        items.append({'id': row['id'], 'file': path.name, 'sha256': file_hash(path)})
    write_json(output/'manifest.json', {'split': split, 'items': items, 'fresh_model_inference': True,
        'recipe_sha256': file_hash(ROOT/recipe_file)})


def predict(recipe_file, output, split='val', device='cuda'):
    recipe, _ = checked_recipe(ROOT/recipe_file)
    spec, rows = load_manifest(ROOT/'data/manifest.json'); rows = [r for r in rows if r['split'] == split]
    output = ROOT/output; output.mkdir(parents=True, exist_ok=False); begin = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='komap-focused-fresh-', dir='/dev/shm') as temporary:
        generated = Path(temporary)/'teacher'
        subprocess.run([sys.executable, '-u', __file__, '--recipe', recipe_file, '--output', str(generated),
            '--split', split, '--device', device, '--teacher-only'], cwd=ROOT, check=True)
        fresh = read(generated/'manifest.json'); assert fresh['recipe_sha256'] == file_hash(ROOT/recipe_file)
        teachers = {r['id']: np.load(generated/r['file'], allow_pickle=False) for r in fresh['items']}
        exact_teacher = 0
        if split == 'val':
            cache = ROOT/recipe['validation_cache']['folder']
            assert file_hash(cache/'manifest.json') == recipe['validation_cache']['manifest_sha256']
            for item in read(cache/'manifest.json')['items']:
                np.testing.assert_array_equal(teachers[item['id']], np.load(cache/item['file'], allow_pickle=False))
                exact_teacher += 1
            assert exact_teacher == 20
        # Strict runtime starts only after the default-runtime teacher process has finished.
        from focused_refiner_model import load_focused_refiner
        from focused_refiner_evaluation import decode_probability
        from refiner_d4 import predict_refined_d4
        from image_processing import enhance
        from context_model import summarize_scores
        model, saved = load_focused_refiner(ROOT/recipe['head']['file'], device)
        assert file_hash(ROOT/recipe['head']['file']) == recipe['head']['sha256'] and saved['classes'] == spec['classes']
        record = read(ROOT/recipe['record']['file']); folder = ROOT/record['prediction_root']
        refs = {r['id']: r for r in record['predictions']}; arrays = {r['id']: r for r in record['probabilities']}
        scores, predictions, probability_exact = [], [], 0; alternatives = {r: [] for r in ('skip', 'one', 'zero')}
        for index, row in enumerate(rows, 1):
            original = read_image(row['image'])
            with torch.inference_mode():
                q = predict_refined_d4(model, enhance(original, 'he50'), teachers[row['id']], torch.device(device), 1.)
            pred = decode_probability(q, original, 'd4_crf')
            path = output/(row['id']+'.png'); save_mask(path, pred, spec)
            if split == 'val':
                saved_q = np.load(folder/arrays[row['id']]['file'], allow_pickle=False)
                np.testing.assert_array_equal(q.astype(np.float32, copy=False), saved_q); probability_exact += 1
                np.testing.assert_array_equal(pred, decode_mask(folder/refs[row['id']]['file'], spec))
                truth = decode_mask(row['mask'], spec); scores.append({'id': row['id'], **score_image(pred, truth, 'skip')})
                for rule in alternatives: alternatives[rule].append(score_image(pred, truth, rule)['miou'])
            predictions.append({'id': row['id'], 'file': path.name, 'sha256': file_hash(path), 'shape': list(pred.shape)})
            print(json.dumps({'phase': 'fresh_focused_head', 'image': index, 'total': len(rows)}), flush=True)
        result = {'candidate': recipe['candidate'], 'split': split, 'predictions': predictions,
            'recipe_sha256': file_hash(ROOT/recipe_file), 'seconds': time.monotonic()-begin,
            'fresh_teacher_and_refiner_inference': True, 'fresh_teacher_arrays_equal_cache': exact_teacher,
            'fresh_d4_probabilities_equal_saved': probability_exact, 'not_submitted': True, 'createdAt': now()}
        if scores:
            metrics = summarize_scores(scores, alternatives, spec)
            expected = {r['id']: r for r in record['metrics']['images']}
            assert all(r['confusion'] == expected[r['id']]['confusion'] for r in scores)
            assert metrics['mean_image_miou'] == record['metrics']['mean_image_miou']
            result.update(metrics=metrics, reproduction_max_delta=0.)
        write_json(output/'prediction-manifest.json', result)
        print(json.dumps({'candidate': recipe['candidate'], 'seconds': result['seconds'],
            'miou': result.get('metrics', {}).get('mean_image_miou')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='val'); parser.add_argument('--device', default='cuda')
    parser.add_argument('--teacher-only', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(4)
    (teacher_only if args.teacher_only else predict)(args.recipe, args.output, args.split, args.device)
