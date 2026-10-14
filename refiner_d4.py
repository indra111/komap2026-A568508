"""Uniform D4 probability averaging of the frozen native residual head.

Teacher probabilities and RGB undergo the same exact pixel permutation.
No resampling, fitting, or test-time parameter updates are performed.
"""
import numpy as np
from cached_refiner_eval import predict_refined

VIEWS = tuple((turns, flip) for turns in range(4) for flip in (False, True))


def predict_refined_d4(model, image, probability, device, strength):
    if image.ndim != 3 or image.shape[2] != 3 or probability.shape != (4, *image.shape[:2]):
        raise ValueError('Expected aligned RGB and four-class probabilities')
    if strength not in (.5, 1.):
        raise ValueError('Unreviewed residual strength')
    total = np.zeros_like(probability, dtype=np.float32)
    for turns, flip in VIEWS:
        rgb = np.rot90(image, turns, axes=(0, 1))
        prob = np.rot90(probability, turns, axes=(1, 2))
        if flip:
            rgb = np.flip(rgb, axis=1); prob = np.flip(prob, axis=2)
        result = predict_refined(model, np.ascontiguousarray(rgb), np.ascontiguousarray(prob), device, strength)
        if flip:
            result = np.flip(result, axis=2)
        result = np.rot90(result, -turns, axes=(1, 2))
        total += result
    total /= len(VIEWS)
    if not np.isfinite(total).all() or (total < 0).any():
        raise ValueError('Invalid D4 probabilities')
    np.testing.assert_allclose(total.sum(0), 1., atol=1e-5, rtol=1e-5)
    return total
