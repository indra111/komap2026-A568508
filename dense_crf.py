"""Fixed DenseCRF inference from model probabilities and original RGB, without labels."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
PRESETS = {
    'micro_edges': {'iterations': 5, 'spatial_sigma': 1., 'spatial_weight': .25,
                    'bilateral_sigma': 3., 'color_sigma': 13., 'bilateral_weight': 1.},
    'reference': {'iterations': 5, 'spatial_sigma': 3., 'spatial_weight': 3.,
                  'bilateral_sigma': 80., 'color_sigma': 13., 'bilateral_weight': 10.},
}


@lru_cache(maxsize=1)
def backend():
    manifest = json.loads((ROOT/'third_party/densecrf-reference/backend-manifest.json').read_text())
    for item in manifest['installed_files']:
        path = (ROOT/item['path']).resolve()
        if not path.is_relative_to(ROOT/'.crf-deps') or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('DenseCRF backend differs from verified build')
    sys.path.insert(0, str(ROOT/'.crf-deps'))
    import pydensecrf.densecrf as dcrf
    return dcrf


def refine(probabilities, image, parameters):
    p = np.asarray(probabilities)
    if p.ndim != 3 or p.shape[0] != 4 or min(p.shape[1:]) < 1:
        raise ValueError('Expected four-class C,H,W probabilities')
    if image.dtype != np.uint8 or image.shape != (*p.shape[1:], 3):
        raise ValueError('RGB uint8 and probability coordinates must match')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(0), 1., atol=1e-5, rtol=1e-5):
        raise ValueError('Invalid probability distribution')
    if set(parameters) != set(PRESETS['micro_edges']):
        raise ValueError('Unexpected DenseCRF parameters')
    if not isinstance(parameters['iterations'], int) or not 1 <= parameters['iterations'] <= 50:
        raise ValueError('Invalid iteration count')
    for key, value in parameters.items():
        if not np.isfinite(value) or value < 0 or (key.endswith('sigma') and value <= 0):
            raise ValueError('Invalid DenseCRF parameter: ' + key)
    dcrf = backend()
    h, w = p.shape[1:]
    d = dcrf.DenseCRF2D(w, h, 4)
    clipped = np.clip(p.astype(np.float32, copy=True), 1e-8, 1.)
    clipped /= clipped.sum(0, keepdims=True)
    unary = np.ascontiguousarray(-np.log(clipped).reshape(4, -1), dtype=np.float32)
    d.setUnaryEnergy(unary)
    if parameters['spatial_weight']:
        d.addPairwiseGaussian(sxy=parameters['spatial_sigma'], compat=parameters['spatial_weight'],
            kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    if parameters['bilateral_weight']:
        d.addPairwiseBilateral(sxy=parameters['bilateral_sigma'], srgb=parameters['color_sigma'],
            rgbim=np.ascontiguousarray(image), compat=parameters['bilateral_weight'],
            kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    q = np.array(d.inference(parameters['iterations']), dtype=np.float32).reshape(4, h, w)
    if not np.isfinite(q).all() or (q < 0).any() or not np.allclose(q.sum(0), 1., atol=1e-5, rtol=1e-5):
        raise FloatingPointError('Invalid DenseCRF result')
    return q
