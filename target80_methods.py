"""Loss and inference hypotheses for target80; never consumes validation labels in fitting."""
import numpy as np
import torch
import torch.nn.functional as F
from expansion import multiclass_loss
from context_model import predict_probabilities


def lovasz_image_loss(logits, target):
    """Per-image Lovasz extension of Jaccard, all 4 classes including absent ones.

    Formula: Berman et al., CVPR2018, arXiv:1705.08790.
    Sorting is float32; CE supplies gradients for every pixel as well.
    """
    probs = logits.float().softmax(1)
    terms = []
    for p, y in zip(probs, target):
        for c in range(p.shape[0]):
            foreground = (y.flatten() == c).float()
            errors, order = (foreground - p[c].flatten()).abs().sort(descending=True)
            ordered = foreground[order]
            positives = ordered.sum()
            intersection = positives - ordered.cumsum(0)
            union = positives + (1 - ordered).cumsum(0)
            jaccard = 1 - intersection / union.clamp_min(1)
            gradient = torch.cat((jaccard[:1], jaccard[1:] - jaccard[:-1]))
            terms.append(torch.dot(errors, gradient))
    return torch.stack(terms).mean()


def pair23_loss(logits, target):
    """Conditional 2-vs-3 classification, image/class-balanced on train pixels.

    This is a task-specific hypothesis related to hierarchical losses, not a
    claim of a new scientific loss. Other classes retain the base 4-class loss.
    """
    terms = []
    logp = logits[:, 2:4].float().log_softmax(1)
    for sample, y in zip(logp, target):
        for c in (2, 3):
            keep = y == c
            if keep.any():
                terms.append(-sample[c - 2][keep].mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0.


def training_loss(logits, target, method):
    if method == 'control':
        return multiclass_loss(logits, target)
    if method == 'lovasz':
        return F.cross_entropy(logits.float(), target) + lovasz_image_loss(logits, target)
    if method == 'pair23':
        return multiclass_loss(logits, target) + .25 * pair23_loss(logits, target)
    if method == 'ohem':
        ce = F.cross_entropy(logits.float(), target, reduction='none').flatten(1)
        hard = ce.topk(max(1, ce.shape[1] // 4), dim=1).values.mean()
        return multiclass_loss(logits, target) + .25 * hard
    raise ValueError('Unknown loss method: ' + method)


def predict_d4(model, image, device, patch=512, overlap=128, amp=True):
    """All rotations and horizontal reflections, each undone before averaging."""
    total = None
    for turns in range(4):
        for flip in (False, True):
            x = np.rot90(image, turns)
            if flip:
                x = np.flip(x, 1)
            p = predict_probabilities(model, np.ascontiguousarray(x), device,
                                      patch=patch, overlap=overlap, amp=amp)
            if flip:
                p = np.flip(p, 2)
            p = np.rot90(p, -turns, axes=(1, 2))
            total = p.copy() if total is None else total + p
    return total / 8.
