"""NumPy adaptation of RankSEG-RMA's unsmoothed multiclass IoU decoder.

Algorithm reference: Wang & Dai 2025, arXiv:2510.15362 and rankseg/rankseg
revision36e64890b206e0865aeab8fb4c249483d961839a (_rankseg_algo.py).
Adapted algorithm portions: Copyright (c)2025 Ben Dai, Zixun Wang, BSD-3-Clause.
See third_party/rankseg-reference/LICENSE. No model fitting or label input.
This approximation is not a guarantee of higher measured IoU.
"""
import numpy as np


def rank_iou(probs, pruning_prob=.5):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] != 4 or not np.isfinite(p).all():
        raise ValueError('Expected finite four-class C,H,W probabilities')
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(0), 1., atol=1e-5):
        raise ValueError('Invalid probability simplex')
    if not 0 <= pruning_prob <= 1:
        raise ValueError('Invalid pruning probability')
    shape = p.shape[1:]
    p = p.reshape(4, -1)
    total = p.sum(1)
    binary = np.zeros(p.shape, dtype=bool)
    active = p.max(1) > pruning_prob
    volumes, expected = [], []
    sizes = np.arange(1, p.shape[1] + 1, dtype=np.float64)
    for phase in range(4):
        if not active[phase]:
            volumes.append(0)
            expected.append(0.)
            continue
        order = np.argsort(-p[phase], kind='stable')
        cumul = p[phase, order].cumsum()
        values = cumul / (total[phase] + sizes - cumul)
        # Earliest maximum, including an empty mask with unsmoothed score0.
        best = int(np.argmax(np.concatenate(([0.], values))))
        binary[phase, order[:best]] = True
        volumes.append(best)
        expected.append(float(values[best - 1]) if best else 0.)
    counts = binary.sum(0)
    unique = binary & (counts == 1)[None, :]
    intersection = (p * unique).sum(1)
    cardinality = unique.sum(1).astype(np.float64)
    union = total + cardinality - intersection
    # Change in approximate IoU when adding a pixel to the uniquely assigned
    # region. Algebraic form avoids subtracting nearly equal O(1) scores.
    numerator = p * (union + intersection)[:, None] - intersection[:, None]
    denominator = union[:, None] * (union[:, None] + 1. - p)
    increments = np.divide(numerator, denominator, out=np.zeros_like(p), where=denominator > 0)
    eligible = binary.copy()
    unassigned = counts == 0
    allowed = active if active.any() else np.ones(4, dtype=bool)
    eligible[:, unassigned] = allowed[:, None]
    pred = np.where(eligible, increments, -np.inf).argmax(0).astype(np.uint8)
    return pred.reshape(shape), {'binary_volumes': volumes, 'approximate_binary_iou': expected,
        'active_classes': active.tolist(), 'overlap_pixels': int((counts > 1).sum()),
        'unassigned_pixels': int(unassigned.sum()),
        'changed_from_argmax_pixels': int((pred != p.argmax(0)).sum())}
