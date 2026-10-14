"""Fixed image magnification, D4 inference, and native-coordinate probabilities.

Interpolation changes the sampling seen by a patch14 encoder; it does not recover
new measured detail. No mask, image identifier, or per-image scale selection is used.
"""
import math

import numpy as np
from context_model import predict_probabilities
from image_processing import cv2
from target80_methods import predict_d4


def validate_scale(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError('Scale must be a finite number in [0.5, 4]')
    if not math.isfinite(value) or not .5 <= value <= 4:
        raise ValueError('Scale must be a finite number in [0.5, 4]')
    return float(value)


def predict_scaled(model, image, device, *, scale=1., tta='d4', patch=512, overlap=128):
    scale = validate_scale(scale)
    if tta not in ('identity', 'd4'):
        raise ValueError('Unknown inference transform')
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    height, width = image.shape[:2]
    if min(height, width) < 1:
        raise ValueError('Empty image')
    predictor = predict_d4 if tta == 'd4' else predict_probabilities
    if scale == 1:
        # Preserve historical predictions bit for bit.
        return predictor(model, image, device, patch=patch, overlap=overlap)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    # Exact integer interpolation avoids orientation-dependent uint8 rounding.
    enlarged = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR_EXACT)
    probabilities = predictor(model, enlarged, device, patch=patch, overlap=overlap)
    result = np.stack([cv2.resize(p, (width, height), interpolation=cv2.INTER_LINEAR)
                       for p in probabilities]).astype(np.float32)
    if result.shape != (4, height, width) or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid resized probabilities')
    mass = result.sum(0, keepdims=True)
    if (mass <= 0).any():
        raise ValueError('Zero probability mass')
    return result / mass
