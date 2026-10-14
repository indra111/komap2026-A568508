"""Physical-scale normalization for the low-resolution KoMaP images.

The 675x480 images show the same microstructure ~1.85x smaller (train-mask
component areas and boundary density, 2026-10-10). Rule uses image size only:
width < 800 -> enlarge by FACTOR for training and inference; probabilities
are reduced back to the original grid before scoring, so metrics stay on the
official masks and resolution.
"""
from functools import lru_cache

import numpy as np
from context_model import predict_probabilities
from core import decode_mask, read_image, save_mask, score_image
from expansion import ExpandedSampler
from image_processing import cv2
from context_model import summarize_scores
from target80_methods import predict_d4

SMALL_WIDTH = 800
FACTOR = 1.85


def factor_for(shape, base=1.):
    return (FACTOR if shape[1] < SMALL_WIDTH else 1.) * base


def enlarge_pair(image, mask, base=1.):
    f = factor_for(image.shape, base)
    if f == 1.:
        return image, mask
    h, w = image.shape[:2]
    size = (round(w * f), round(h * f))
    big = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR_EXACT)
    # Smooth class boundaries: resize one-hot planes, then argmax (no staircase).
    planes = np.stack([cv2.resize((mask == c).astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
                       for c in range(4)])
    return big, planes.argmax(0).astype(mask.dtype)


class ScaleNormSampler(ExpandedSampler):
    base = 1.
    alias = None

    @lru_cache(maxsize=80)
    def _pair_base(self, i):
        return enlarge_pair(*ExpandedSampler.pair.__wrapped__(self, i), base=self.base)

    def pair(self, i):
        return self._pair_base(self.alias[i] if self.alias else i)

    def boost_lowres(self, k):
        """Draw low-resolution (width < SMALL_WIDTH) training images k times as often by appending
        k-1 aliases per image; the cache stays keyed on the original index."""
        from PIL import Image
        n = len(self.items)
        low = [i for i, r in enumerate(self.items) if Image.open(r['image']).size[0] < SMALL_WIDTH]
        extra = [i for i in low for _ in range(k - 1)]
        self.alias = list(range(n)) + extra
        self.items = self.items + [self.items[i] for i in extra]
        return len(low)


class CopyPasteSampler(ScaleNormSampler):
    """Scale-normalized crops plus particle copy-paste. A few Al3Ni / eutectic Si particles
    from another training crop are pasted at random positions where their 1-px ring lies
    wholly on the target's Al matrix. Source brightness is rescaled by the ratio of matrix means so
    the particle keeps its brightness relative to Al (the human cue Al3Ni 0.65 vs Si 0.37).
    Pasted rims (inner+outer 1 px) are labelled -100 and excluded from the loss."""
    prob = .3
    max_particles = 12

    def batch(self, count):
        import torch
        x, y = ScaleNormSampler.batch(self, count)
        k3 = np.ones((3, 3), np.uint8)
        for k in range(count):
            if self.aug_rng.random() >= self.prob:
                continue
            src_img, src_mask = self.pair(int(self.aug_rng.integers(len(self.items))))
            h, w = src_mask.shape; s = self.patch
            if h < s or w < s:
                continue
            oy, ox = int(self.aug_rng.integers(h - s + 1)), int(self.aug_rng.integers(w - s + 1))
            sm = src_mask[oy:oy + s, ox:ox + s]; si = src_img[oy:oy + s, ox:ox + s].astype(np.float32) / 255.
            img = x[k].permute(1, 2, 0).numpy().copy(); target = y[k].numpy().copy()
            matrix = target == 0
            if matrix.sum() < 1000 or (sm == 0).sum() < 1000:
                continue
            gain = img[matrix].mean(0) / np.maximum(si[sm == 0].mean(0), 1e-3)
            free = cv2.erode(matrix.astype(np.uint8), k3) > 0   # pixel and its 8 neighbours are matrix
            pasted = np.zeros((s, s), bool); rim = np.zeros((s, s), bool)
            comps = []
            for c in (2, 3):
                n, lab, stats, _ = cv2.connectedComponentsWithStats((sm == c).astype(np.uint8), connectivity=8)
                for i in range(1, n):
                    if 16 <= stats[i, cv2.CC_STAT_AREA] <= 4000:
                        bx, by, bw, bh = stats[i, :4]
                        comps.append((c, lab[by:by + bh, bx:bx + bw] == i, si[by:by + bh, bx:bx + bw]))
            if not comps:
                continue
            want = int(self.aug_rng.integers(3, self.max_particles + 1))
            for t in self.aug_rng.permutation(len(comps))[:want * 3]:
                c, part, pix = comps[t]
                bh, bw = part.shape
                if bh + 2 > s or bw + 2 > s:
                    continue
                py, px = int(self.aug_rng.integers(1, s - bh)), int(self.aug_rng.integers(1, s - bw))
                win = (slice(py - 1, py + bh + 1), slice(px - 1, px + bw + 1))
                local = np.pad(part, 1)
                ring = cv2.dilate(local.astype(np.uint8), k3) > 0
                if not free[win][ring].all() or pasted[win][ring].any():
                    continue
                img[win][local] = np.clip(np.pad(pix, ((1, 1), (1, 1), (0, 0)))[local] * gain, 0, 1)
                target[win][local] = c; pasted[win] |= local
                rim[win] |= ring & ~(cv2.erode(local.astype(np.uint8), k3) > 0)
                want -= 1
                if want == 0:
                    break
            if not pasted.any():
                continue
            target[rim] = -100
            x[k] = torch.from_numpy(img).permute(2, 0, 1)
            y[k] = torch.from_numpy(target)
        return x, y


def predict_normalized(model, image, device, *, d4=False, patch=512, overlap=128, base=1.):
    predictor = predict_d4 if d4 else predict_probabilities
    f = factor_for(image.shape, base)
    if f == 1.:
        return predictor(model, image, device, patch=patch, overlap=overlap)
    h, w = image.shape[:2]
    big = cv2.resize(image, (round(w * f), round(h * f)), interpolation=cv2.INTER_LINEAR_EXACT)
    p = predictor(model, big, device, patch=patch, overlap=overlap)
    p = np.stack([cv2.resize(c, (w, h), interpolation=cv2.INTER_AREA) for c in p]).astype(np.float32)
    return p / p.sum(0, keepdims=True)


def evaluate_normalized(model, rows, spec, device, *, d4=False, patch=512, overlap=128,
                        output=None, probability_output=None, split='val', base=1.):
    results, alternatives = [], {p: [] for p in ('skip', 'one', 'zero')}
    for row in rows:
        if row['split'] != split:
            continue
        probs = predict_normalized(model, read_image(row['image']), device, d4=d4, patch=patch, overlap=overlap, base=base)
        pred = probs.argmax(0).astype(np.uint8)
        if output:
            output.mkdir(parents=True, exist_ok=True)
            save_mask(output / (row['id'] + '.png'), pred, spec)
        if probability_output:
            probability_output.mkdir(parents=True, exist_ok=True)
            np.save(probability_output / (row['id'] + '.npy'), probs.astype(np.float16), allow_pickle=False)
        if split != 'val':
            continue
        target = decode_mask(row['mask'], spec)
        results.append({'id': row['id'], **score_image(pred, target, spec['absent_class_policy'])})
        for p in alternatives:
            alternatives[p].append(score_image(pred, target, p)['miou'])
    return summarize_scores(results, alternatives, spec) if results else None
