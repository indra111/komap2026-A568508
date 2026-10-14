"""Validation-only inputs and no-gradient native refiner evaluation."""
import json
from pathlib import Path
import numpy as np
import torch

from core import file_hash, load_manifest, read_image, decode_mask, save_mask, score_image
from context_model import summarize_scores
from image_processing import enhance
from train_prediction_cache import valid_probabilities


def load_validation(root, folder, teacher_file):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    assert folder.is_relative_to(root)
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['split'] == 'val' and manifest['for_training'] is False
    assert manifest['teacher_recipe_sha256'] == file_hash(root/teacher_file)
    spec, rows = load_manifest(root/'data/manifest.json')
    assert manifest['input_manifest_sha256'] == file_hash(root/'data/manifest.json')
    rows = {r['id']: r for r in rows if r['split'] == 'val'}
    assert len(manifest['items']) == len(rows) and {r['id'] for r in manifest['items']} == set(rows)
    if spec['provenance'] == 'official_komap_2026':
        assert len(rows) == 20
    pairs, hashes = [], []
    for item in manifest['items']:
        row = rows[item['id']]
        assert item['split'] == row['split'] == 'val'
        assert file_hash(row['image']) == item['image_sha256']
        path = (folder/item['file']).resolve()
        assert path.is_relative_to(folder) and file_hash(path) == item['sha256']
        p = np.load(path, allow_pickle=False)
        image = enhance(read_image(row['image']), 'he50')
        truth = decode_mask(row['mask'], spec)
        valid_probabilities(p, truth.shape)
        pairs.append((row['id'], image, p, truth))
        hashes.append({'id': row['id'], 'image_sha256': item['image_sha256'],
                       'mask_sha256': file_hash(row['mask']), 'probability_sha256': item['sha256']})
    return spec, pairs, hashes


@torch.inference_mode()
def predict_refined(model, image, probability, device, strength=1.):
    x = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().unsqueeze(0).to(device)/255
    p = torch.from_numpy(np.ascontiguousarray(probability)).unsqueeze(0).to(device)
    model.eval()
    result = model(x, p, strength).squeeze(0).float().cpu().numpy()
    valid_probabilities(result, image.shape[:2])
    return result


def evaluate(model, pairs, spec, device, strength=1., output=None):
    model.eval()
    scores, predictions = [], []
    alternatives = {r: [] for r in ('skip', 'one', 'zero')}
    if output is not None:
        output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for name, image, p, truth in pairs:
        q = predict_refined(model, image, p, device, strength)
        pred = q.argmax(0).astype(np.uint8)
        scores.append({'id': name, **score_image(pred, truth, 'skip')})
        for rule in alternatives:
            alternatives[rule].append(score_image(pred, truth, rule)['miou'])
        if output is not None:
            path = output/(name+'.png'); save_mask(path, pred, spec)
            np.testing.assert_array_equal(pred, decode_mask(path, spec))
            predictions.append({'id': name, 'file': path.name, 'sha256': file_hash(path), 'shape': list(pred.shape)})
    return summarize_scores(scores, alternatives, spec), predictions
