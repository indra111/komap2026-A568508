"""Fixed, label-free full-image enhancement; preserve every original pixel position."""
from pathlib import Path
import sys
import numpy as np

# OpenCV is isolated from the existing training environment.
DEPS = Path(__file__).resolve().parent / '.imageproc-deps'
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))
import cv2
cv2.setNumThreads(1)

METHODS = {
    'he50': {'algorithm': 'global_histogram_equalization', 'blend': .5},
    'clahe50': {'algorithm': 'clahe', 'blend': .5, 'clip_limit': 2., 'tile_grid': [8, 8]},
}


def enhance(image, method):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    if method == 'raw':
        return image.copy()
    config = METHODS[method]
    # Preserve grayscale exactly, avoiding unnecessary Lab round-trip quantization.
    gray = np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 0], image[..., 2])
    lab = None if gray else cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    luminance = image[..., 0] if gray else lab[..., 0]
    if luminance.min() == luminance.max():
        return image.copy()
    if method == 'he50':
        adjusted = cv2.equalizeHist(luminance)
    else:
        adjusted = cv2.createCLAHE(clipLimit=config['clip_limit'], tileGridSize=tuple(config['tile_grid'])).apply(luminance)
    mixed = np.rint((1-config['blend'])*luminance.astype(np.float32)+config['blend']*adjusted).clip(0,255).astype(np.uint8)
    if gray:
        return np.repeat(mixed[..., None], 3, axis=2)
    lab[..., 0] = mixed
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def predict_multiscale(model, image, device, scales=(.75, 1., 1.25), patch=512, overlap=128):
    """Average probabilities after resizing each view back to native geometry."""
    from context_model import predict_probabilities
    height, width = image.shape[:2]
    result = np.zeros((4, height, width), dtype=np.float32)
    for scale in scales:
        if scale <= 0:
            raise ValueError('Scale must be positive')
        shape = (max(1, round(width*scale)), max(1, round(height*scale)))
        view = image if scale == 1 else cv2.resize(image, shape, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        probability = predict_probabilities(model, view, device, patch=patch, overlap=overlap)
        for cls in range(4):
            result[cls] += cv2.resize(probability[cls], (width,height), interpolation=cv2.INTER_LINEAR) / len(scales)
    result /= result.sum(axis=0, keepdims=True)
    return result
