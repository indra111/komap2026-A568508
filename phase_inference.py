"""Integer input shifts with exact return to original pixel coordinates.

This tests patch-grid sensitivity without resizing or fitting to masks. Edge
padding also changes border context and sliding-window placement; it is not a
guarantee of shift equivariance or a reproduction of adaptive ViT architectures.
"""
import numpy as np
from context_model import predict_probabilities
from target80_methods import predict_d4


def validate_offset(offset):
    if (not isinstance(offset, (tuple, list)) or len(offset) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 13 for v in offset)):
        raise ValueError('Offset must be two integers in [0, 13]')
    return tuple(offset)


def predict_phase(model, image, device, *, offset=(7, 7), tta='d4', patch=512, overlap=128, amp=True):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError('Expected nonempty uint8 RGB image')
    offset = validate_offset(offset)
    if tta not in ('identity', 'd4'):
        raise ValueError('Unknown inference transform')
    predictor = predict_d4 if tta == 'd4' else predict_probabilities
    dy, dx = offset
    if dy == dx == 0:
        return predictor(model, image, device, patch=patch, overlap=overlap, amp=amp)
    height, width = image.shape[:2]
    # Symmetric edge extension supplies integer translations, never wrapping
    # unrelated opposite borders or inventing interpolated image measurements.
    padded = np.pad(image, ((dy, dy), (dx, dx), (0, 0)), mode='edge')
    probability = predictor(model, padded, device, patch=patch, overlap=overlap, amp=amp)
    result = np.ascontiguousarray(probability[:, dy:dy+height, dx:dx+width])
    if result.shape != (4, height, width) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid shifted probabilities')
    if not np.allclose(result.sum(0), 1., atol=2e-5):
        raise ValueError('Invalid probability mass')
    return result


def predict_two_phase(model, image, device, *, tta='d4', patch=512, overlap=128, amp=True):
    """One fixed comparison: native grid and diagonal seven-pixel shift, 50:50."""
    original = predict_phase(model, image, device, offset=(0, 0), tta=tta,
                             patch=patch, overlap=overlap, amp=amp)
    shifted = predict_phase(model, image, device, offset=(7, 7), tta=tta,
                            patch=patch, overlap=overlap, amp=amp)
    return (original + shifted) * .5
