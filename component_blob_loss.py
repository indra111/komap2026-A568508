"""Two-dimensional, four-phase adaptation of multiclass blob DC+BCE loss.

Reference: https://arxiv.org/html/2604.24276v1, sections 2.1, 2.3, 2.4.
All four KoMaP phases participate, including matrix. Each connected region is
paired with pixels of OTHER classes; other regions of its own class are ignored.
We average regions within a present class, then present classes, then images.
Use alongside the existing four-class global loss, not as an evaluation metric.
No inverse-size pixel weights, learned geometry, or prediction postprocessing.
"""
import numpy as np
import torch
from image_processing import cv2

SMOOTH = 1e-5


def connected_regions(target):
    """Native crop labels, 8-connectivity; regions are not physical particles."""
    value = np.asarray(target)
    if value.ndim != 2 or not value.size or not np.issubdtype(value.dtype, np.integer):
        raise ValueError('Expected a nonempty 2D integer mask')
    if not np.isin(value, range(4)).all():
        raise ValueError('Expected exactly the allowed phase IDs 0..3')
    result = []
    for cls in range(4):
        foreground = (value == cls).astype(np.uint8)
        if not foreground.any():
            continue
        count, labels = cv2.connectedComponents(foreground, connectivity=8)
        sizes = np.bincount(labels.ravel(), minlength=count)
        assert (sizes[1:] > 0).all()
        result.append((cls, labels, sizes))
    return result


def multiclass_blob_loss(logits, target):
    """O(4*N) memory/work after connected components, independent of blob count.

    Component domains share the same other-class pixels. Summing that context
    once and combining it with scatter-summed foreground statistics is exactly
    equivalent to independently masking the whole image for every component.
    All arithmetic uses FP32; log(1-p) uses the other logits' logsumexp to stay
    finite even when softmax rounds a confident probability to exactly one.
    Absent classes are omitted only from this auxiliary term. The unchanged
    global CE+four-class Dice still penalizes their false-positive predictions.
    """
    if logits.ndim != 4 or logits.shape[1] != 4 or target.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError('Expected logits Bx4xHxW and target BxHxW')
    if target.dtype != torch.long or not target.numel():
        raise ValueError('Expected nonempty int64 target')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite logits')
    labels_cpu = target.detach().cpu().numpy()
    values = logits.float()
    normalizer = values.logsumexp(1)
    probabilities = values.softmax(1)
    images = []
    for index, mask in enumerate(labels_cpu):
        classes = []
        for cls, labels, sizes in connected_regions(mask):
            region = torch.from_numpy(labels.ravel().astype(np.int64)).to(values.device)
            counts = torch.from_numpy(sizes).to(device=values.device, dtype=torch.float32)
            p = probabilities[index, cls].flatten()
            pos_ce = (normalizer[index] - values[index, cls]).flatten()
            other_logits = values[index, [c for c in range(4) if c != cls]]
            neg_ce = (normalizer[index] - other_logits.logsumexp(0)).flatten()
            neg = region == 0
            sums = torch.zeros(len(sizes), dtype=torch.float32, device=values.device)
            probability_sum = sums.scatter_add(0, region, p)
            positive_ce_sum = sums.scatter_add(0, region, pos_ce)
            positive_sizes = counts[1:]
            context_size = counts[0]
            bce = (positive_ce_sum[1:] + neg_ce[neg].sum()) / (positive_sizes + context_size)
            dice = 1 - (2 * probability_sum[1:] + SMOOTH) / (
                positive_sizes + probability_sum[1:] + probability_sum[0] + SMOOTH)
            classes.append((bce + dice).mean())
        images.append(torch.stack(classes).mean())
    return torch.stack(images).mean()
