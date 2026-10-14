"""Four-phase segmentation helpers. No model downloads or external services."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

NUM_CLASSES = 4


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def make_model():
    return smp.Unet(encoder_name='resnet34', encoder_weights=None, in_channels=3, classes=NUM_CLASSES)


def read_image(path):
    with Image.open(path) as im:
        if im.mode not in ('L', 'RGB', 'RGBA'):
            raise ValueError(f'Unsupported input mode {im.mode}: {path}; define conversion explicitly')
        return np.array(im.convert('RGB'))


def decode_mask(path, spec):
    with Image.open(path) as im:
        a = np.array(im if spec['mask_encoding'] == 'index' else im.convert('RGB'))
    if spec['mask_encoding'] == 'index':
        if a.ndim != 2 or not np.isin(a, np.arange(NUM_CLASSES)).all():
            raise ValueError(f'Expected class IDs 0..3 only: {path}')
        return a.astype(np.int64)
    output = np.full(a.shape[:2], -1, dtype=np.int64)
    for cls in spec['classes']:
        output[np.all(a == cls['color'], axis=-1)] = cls['id']
    if (output < 0).any():
        unknown = np.unique(a[output < 0], axis=0)
        raise ValueError(f'Unknown mask colors, no nearest-color guessing: {path}: {unknown[:12].tolist()}')
    return output


def load_manifest(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    if spec.get('provenance') not in ('official_komap_2026', 'synthetic_fixture'):
        raise ValueError('Explicit official_komap_2026 or synthetic_fixture provenance required')
    if spec.get('mask_encoding') not in ('rgb', 'index'):
        raise ValueError('Explicit mask_encoding rgb or index is required')
    if sorted(c['id'] for c in spec['classes']) != list(range(NUM_CLASSES)):
        raise ValueError('Exactly four unique class IDs 0..3 required')
    colors = [tuple(c['color']) for c in sorted(spec['classes'], key=lambda x: x['id'])]
    if len(set(colors)) != 4 or any(len(c) != 3 or any(not isinstance(v, int) or v < 0 or v > 255 for v in c) for c in colors):
        raise ValueError('Four distinct RGB output colors required')
    if spec.get('absent_class_policy') not in ('skip', 'one', 'zero'):
        raise ValueError('Select an explicit absent_class_policy; do not assume an official rule')
    if spec.get('metric_rule_status') not in ('provisional', 'official_confirmed'):
        raise ValueError('metric_rule_status must be provisional or official_confirmed')
    if not spec.get('items'):
        raise ValueError('No dataset items provided')
    items, ids = [], set()
    for row in spec['items']:
        if row['id'] in ids or Path(row['id']).name != row['id'] or row['id'] in ('.', '..', ''):
            raise ValueError('Image IDs must be unique safe filenames')
        ids.add(row['id'])
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError('Unsupported split')
        item = dict(row)
        for key in ('image', 'mask'):
            if row.get(key):
                item[key] = str((path.parent / row[key]).resolve())
                if not Path(item[key]).is_file():
                    raise ValueError(f'Missing {key} for {row["id"]}')
        if row['split'] != 'test' and not item.get('mask'):
            raise ValueError('Training and validation need ground truth')
        items.append(item)
    return spec, items


def audit(path):
    spec, items = load_manifest(path)
    counts = {split: sum(r['split'] == split for r in items) for split in ('train', 'val', 'test')}
    expected = spec.get('expected_counts', {'train': 70, 'val': 20, 'test': 10})
    if spec['provenance'] == 'official_komap_2026' and expected != {'train': 70, 'val': 20, 'test': 10}:
        raise ValueError('Official dataset must preserve the 70/20/10 split')
    if counts != expected:
        raise ValueError(f'Split counts {counts} differ from declared {expected}')
    seen, groups, records = {}, {}, []
    pixels = {s: np.zeros(4, dtype=np.int64) for s in ('train', 'val')}
    for row in items:
        image = read_image(row['image'])
        # Decoded pixels detect identical images saved under different file encodings.
        digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest in seen:
            raise ValueError(f'Duplicate decoded image: {row["id"]}, {seen[digest]}')
        seen[digest] = row['id']
        group = row.get('group')
        if group and group in groups and groups[group] != row['split']:
            raise ValueError(f'Specimen/group leaks across splits: {group}')
        if group:
            groups[group] = row['split']
        record = {'id': row['id'], 'split': row['split'], 'shape': list(image.shape),
                  'image_sha256': file_hash(row['image']), 'pixel_sha256': digest}
        # Test labels are neither opened nor used, even if included in the archive.
        if row['split'] in ('train', 'val'):
            mask = decode_mask(row['mask'], spec)
            if mask.shape != image.shape[:2]:
                raise ValueError(f'Image/mask size mismatch: {row["id"]}')
            pixels[row['split']] += np.bincount(mask.ravel(), minlength=4)
            record['mask_sha256'] = file_hash(row['mask'])
        records.append(record)
    return {'counts': counts, 'pixels_by_split': {s: v.tolist() for s, v in pixels.items()},
            'items': records, 'manifest_sha256': file_hash(path),
            'metric_rule_status': spec['metric_rule_status'],
            'absent_class_policy': spec['absent_class_policy'],
            'group_metadata_complete': all(r.get('group') for r in items),
            'scope': 'Exact duplicate and supplied group checks; near-duplicate fields of view need visual review'}


def image_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float().div_(255.)


class PatchSampler:
    def __init__(self, items, spec, patch, seed):
        self.items = [r for r in items if r['split'] == 'train']
        self.spec, self.patch = spec, patch
        self.rng = np.random.default_rng(seed)

    @lru_cache(maxsize=4)
    def pair(self, index):
        row = self.items[index]
        return read_image(row['image']), decode_mask(row['mask'], self.spec)

    def batch(self, size):
        images, masks = [], []
        for _ in range(size):
            image, mask = self.pair(int(self.rng.integers(len(self.items))))
            h, w = mask.shape
            ph, pw = max(0, self.patch - h), max(0, self.patch - w)
            if ph or pw:
                image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode='edge')
                mask = np.pad(mask, ((0, ph), (0, pw)), mode='edge')
            y = int(self.rng.integers(image.shape[0] - self.patch + 1))
            x = int(self.rng.integers(image.shape[1] - self.patch + 1))
            image = image[y:y+self.patch, x:x+self.patch]
            mask = mask[y:y+self.patch, x:x+self.patch]
            turns = int(self.rng.integers(4))
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            if self.rng.random() < .5:
                image, mask = np.flip(image, 1), np.flip(mask, 1)
            images.append(image_tensor(image))
            masks.append(torch.from_numpy(np.ascontiguousarray(mask)).long())
        return torch.stack(images), torch.stack(masks)


def segmentation_loss(logits, target):
    ce = F.cross_entropy(logits.float(), target)
    probs = logits.float().softmax(1)
    onehot = F.one_hot(target, 4).permute(0, 3, 1, 2).float()
    intersection = (probs * onehot).sum((0, 2, 3))
    denominator = (probs + onehot).sum((0, 2, 3))
    dice = 1 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()
    return ce + dice  # All four classes participate, including matrix.


def score_image(prediction, target, absent_policy):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError('Prediction and target must have the same nonempty shape')
    if not np.isin(prediction, range(4)).all() or not np.isin(target, range(4)).all():
        raise ValueError('Metrics require IDs 0..3')
    cm = np.bincount(4 * target.ravel() + prediction.ravel(), minlength=16).reshape(4, 4)
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    values = []
    for c in range(4):
        if union[c]:
            values.append(float(inter[c] / union[c]))
        elif absent_policy == 'skip':
            values.append(None)
        elif absent_policy in ('one', 'zero'):
            values.append(1. if absent_policy == 'one' else 0.)
        else:
            raise ValueError('Unknown absent-class policy')
    present = [v for v in values if v is not None]
    return {'miou': float(np.mean(present)), 'iou_per_class': values, 'confusion': cm.tolist()}


def starts(length, patch, stride):
    result = list(range(0, max(0, length - patch) + 1, stride))
    last = max(0, length - patch)
    return result if result[-1] == last else result + [last]


@torch.inference_mode()
def predict(model, image, device, patch=512, overlap=128, amp=True):
    if patch < 32 or patch % 32 or not 0 <= overlap < patch:
        raise ValueError('Patch must be a positive multiple of 32; overlap must be smaller')
    model.eval()
    h, w = image.shape[:2]
    padded = np.pad(image, ((0, max(0, patch-h)), (0, max(0, patch-w)), (0, 0)), mode='edge')
    scores = np.zeros((4, *padded.shape[:2]), np.float32)
    weights = np.zeros(padded.shape[:2], np.float32)
    axis = np.linspace(-1., 1., patch, dtype=np.float32)
    window = np.maximum(np.outer(np.exp(-axis**2/.5), np.exp(-axis**2/.5)), .001)
    for y in starts(padded.shape[0], patch, patch-overlap):
        for x in starts(padded.shape[1], patch, patch-overlap):
            tensor = image_tensor(padded[y:y+patch, x:x+patch]).unsqueeze(0).to(device)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
                probs = model(tensor).float().softmax(1)[0].cpu().numpy()
            scores[:, y:y+patch, x:x+patch] += probs * window
            weights[y:y+patch, x:x+patch] += window
    if (weights <= 0).any():
        raise RuntimeError('Inference left uncovered pixels')
    scores /= weights[None]
    return scores[:, :h, :w].argmax(0).astype(np.uint8)


def save_mask(path, prediction, spec):
    palette = np.array([c['color'] for c in sorted(spec['classes'], key=lambda x: x['id'])], dtype=np.uint8)
    output = prediction.astype(np.uint8) if spec['mask_encoding'] == 'index' else palette[prediction]
    Image.fromarray(output).save(path)
