"""Aligned train-only patches from an audited full-image probability cache."""
import copy
import json
from pathlib import Path
import numpy as np
import torch

from core import file_hash, load_manifest, read_image, decode_mask
from image_processing import enhance
from patch_refiner import PATCH
from train_prediction_cache import valid_probabilities


def load_training(root, cache_folder, audit_file):
    root = Path(root).resolve()
    folder = (root/cache_folder).resolve()
    assert folder.is_relative_to(root)
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['config']['split'] == 'train'
    checked = json.loads((root/audit_file).read_text())
    assert checked['passed'] and checked['cache_manifest_sha256'] == file_hash(folder/'manifest.json')
    assert manifest['OOF_predictions'] is False and manifest['validation_or_test_inference'] is False
    spec, all_rows = load_manifest(root/'data/manifest.json')
    assert file_hash(root/'data/manifest.json') == manifest['config']['input_manifest_sha256']
    rows = {r['id']: r for r in all_rows if r['split'] == 'train'}
    assert len(manifest['items']) == len(rows) and {r['id'] for r in manifest['items']} == set(rows)
    if spec['provenance'] == 'official_komap_2026':
        assert len(rows) == checked['train_images_checked'] == 70
    pairs, inputs = [], []
    for item in manifest['items']:
        row = rows[item['id']]
        assert item['split'] == row['split'] == 'train'
        assert (root/item['image']['file']).resolve() == Path(row['image']).resolve()
        assert file_hash(row['image']) == item['image']['sha256']
        path = (folder/item['probability']['file']).resolve()
        assert path.is_relative_to(folder) and file_hash(path) == item['probability']['sha256']
        image = enhance(read_image(row['image']), 'he50')
        probability = np.load(path, allow_pickle=False)
        target = decode_mask(row['mask'], spec)
        assert image.shape[:2] == target.shape
        valid_probabilities(probability, target.shape)
        pairs.append((row['id'], image, probability, target))
        inputs.append({'id': row['id'], 'image_sha256': item['image']['sha256'],
                       'mask_sha256': file_hash(row['mask']), 'probability_sha256': item['probability']['sha256']})
    return spec, pairs, inputs


class CachedSampler:
    def __init__(self, pairs, seed):
        if not pairs:
            raise ValueError('No training pairs')
        self.pairs = pairs
        self.rng = np.random.default_rng(seed)
        self.draws = 0
        self.seen = set()

    def batch(self, count):
        images, probabilities, targets = [], [], []
        for _ in range(count):
            name, image, probability, target = self.pairs[int(self.rng.integers(len(self.pairs)))]
            h, w = target.shape
            ph, pw = max(0, PATCH-h), max(0, PATCH-w)
            if ph or pw:
                image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode='edge')
                probability = np.pad(probability, ((0, 0), (0, ph), (0, pw)), mode='edge')
                target = np.pad(target, ((0, ph), (0, pw)), mode='edge')
            y = int(self.rng.integers(target.shape[0]-PATCH+1))
            x = int(self.rng.integers(target.shape[1]-PATCH+1))
            image = image[y:y+PATCH, x:x+PATCH]
            probability = probability[:, y:y+PATCH, x:x+PATCH]
            target = target[y:y+PATCH, x:x+PATCH]
            turns, flip = int(self.rng.integers(4)), bool(self.rng.integers(2))
            image = np.rot90(image, turns, axes=(0, 1))
            probability = np.rot90(probability, turns, axes=(1, 2))
            target = np.rot90(target, turns, axes=(0, 1))
            if flip:
                image = np.flip(image, 1); probability = np.flip(probability, 2); target = np.flip(target, 1)
            images.append(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()/255)
            probabilities.append(torch.from_numpy(np.ascontiguousarray(probability)))
            targets.append(torch.from_numpy(np.ascontiguousarray(target)).long())
            self.draws += 1; self.seen.add(name)
        return torch.stack(images), torch.stack(probabilities), torch.stack(targets)

    def state_dict(self):
        return {'rng': copy.deepcopy(self.rng.bit_generator.state), 'draws': self.draws, 'seen': sorted(self.seen)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state['rng']
        self.draws = state['draws']; self.seen = set(state['seen'])
