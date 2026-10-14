"""Paired online augmentation and isolated external-label pretraining helpers."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from core import decode_mask, file_hash, image_tensor, read_image


def crop_pair(image, mask, size, y_fraction, x_fraction, output_size):
    """The exact same crop is applied to image/labels; labels use nearest sampling."""
    h, w = mask.shape
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode='edge')
        mask = np.pad(mask, ((0, ph), (0, pw)), mode='edge')
    y = min(int(y_fraction * (image.shape[0] - size + 1)), image.shape[0] - size)
    x = min(int(x_fraction * (image.shape[1] - size + 1)), image.shape[1] - size)
    image, mask = image[y:y+size, x:x+size], mask[y:y+size, x:x+size]
    if size != output_size:
        image = np.array(Image.fromarray(image).resize((output_size, output_size), Image.Resampling.BILINEAR))
        mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((output_size, output_size), Image.Resampling.NEAREST))
    return image, mask


class ExpandedSampler:
    def __init__(self, rows, spec, patch, seed, augmented=False, external=False):
        self.items = [r for r in rows if r['split'] == 'train']
        if not self.items:
            raise ValueError('No training images')
        self.spec, self.patch = spec, patch
        self.augmented, self.external = augmented, external
        # Separate streams preserve the same image choices in the matched control.
        self.rng = np.random.default_rng(seed)
        self.aug_rng = np.random.default_rng(seed + 731)
        self.draws = 0
        self.seen = set()
        self.external_order = []

    @lru_cache(maxsize=80)
    def pair(self, i):
        row = self.items[i]
        image = read_image(row['image'])
        if self.external:
            with Image.open(row['mask']) as im:
                mask = np.array(im)
            if mask.ndim != 2 or not np.isin(mask, range(len(self.spec['classes']))).all():
                raise ValueError('External normalized masks must contain declared IDs only')
        else:
            mask = decode_mask(row['mask'], self.spec)
        if image.shape[:2] != mask.shape:
            raise ValueError('Mismatched image/mask dimensions')
        return image, mask

    def batch(self, count):
        images, masks = [], []
        for _ in range(count):
            if self.external:
                # Cover every published crop before repeating one in pretraining.
                if not self.external_order:
                    self.external_order = self.rng.permutation(len(self.items)).tolist()
                i = self.external_order.pop()
            else:
                i = int(self.rng.integers(len(self.items)))
            self.seen.add(self.items[i]['id'])
            y, x = self.rng.random(2)
            turns, flip = int(self.rng.integers(4)), self.rng.random() < .5
            scale = self.aug_rng.uniform(.8, 1.2) if self.augmented else 1.
            image, mask = crop_pair(*self.pair(i), int(round(self.patch / scale)), y, x, self.patch)
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            if flip:
                image, mask = np.flip(image, 1), np.flip(mask, 1)
            if self.augmented and self.aug_rng.random() < .75:
                contrast, gain, gamma = self.aug_rng.uniform(.9, 1.1, 3)
                a = image.astype(np.float32) / 255.
                a = np.clip((a - .5) * contrast + .5, 0, 1)
                image = np.round(np.clip(a ** gamma * gain, 0, 1) * 255).astype(np.uint8)
            images.append(image_tensor(image))
            masks.append(torch.from_numpy(np.ascontiguousarray(mask)).long())
            self.draws += 1
        return torch.stack(images), torch.stack(masks)

    def state_dict(self):
        return {'rng': self.rng.bit_generator.state, 'aug_rng': self.aug_rng.bit_generator.state,
                'draws': self.draws, 'seen': sorted(self.seen), 'external_order': self.external_order}


def external_manifest(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get('provenance') != 'zenodo_17315241' or spec.get('license') != 'CC-BY-4.0':
        raise ValueError('Unreviewed external dataset')
    rows = [{**r, 'image': str(path.parent / r['image']), 'mask': str(path.parent / r['mask'])}
            for r in spec['items']]
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('External data is exclusively for pretraining, with no external score claim')
    for row in rows:
        for kind in ('image', 'mask'):
            if not Path(row[kind]).resolve().is_relative_to(path.parent):
                raise ValueError('External dataset path escapes the dataset folder')
            if file_hash(row[kind]) != row[kind + '_sha256']:
                raise ValueError('External file changed after verification')
    return spec, rows


def multiclass_loss(logits, target):
    # Pixels labelled -100 (copy-paste rims) are excluded from CE and Dice; identical otherwise.
    valid = (target >= 0).unsqueeze(1).float()
    probs = logits.float().softmax(1) * valid
    onehot = F.one_hot(target.clamp(min=0), logits.shape[1]).permute(0, 3, 1, 2).float() * valid
    intersection = (probs * onehot).sum((0, 2, 3))
    denominator = (probs + onehot).sum((0, 2, 3))
    return F.cross_entropy(logits.float(), target) + 1 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()


def transfer_encoder(model, checkpoint):
    """Never transfer an external material's decoder or class head into KoMaP."""
    if checkpoint['config']['provenance'] != 'zenodo_17315241':
        raise ValueError('Encoder initialization requires the reviewed external pretraining run')
    encoder = {k.removeprefix('encoder.'): v for k, v in checkpoint['model'].items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(encoder, strict=True)
