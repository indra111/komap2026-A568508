"""Fixed four-flip inference; no fitting on validation labels."""
import numpy as np
from context_model import predict_probabilities


def predict_tta4(model, image, device, patch=512, overlap=128, amp=True):
    total = None
    for axes in [(), (1,), (0,), (0, 1)]:
        transformed = np.ascontiguousarray(np.flip(image, axes)) if axes else image
        p = predict_probabilities(model, transformed, device, patch=patch, overlap=overlap, amp=amp)
        if axes:
            p = np.flip(p, tuple(a + 1 for a in axes))
        total = p.copy() if total is None else total + p
    return total / 4.
