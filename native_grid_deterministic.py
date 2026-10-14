"""Strict replay policy for the new native-grid architecture only."""
from dinov3_deterministic_v2 import configure_determinism, multiclass_loss
import torch
from native_grid_refiner import NativeGridRefiner

ARCHITECTURE = 'cached_four_class_native_grid_strict_residual_bf16'


def configure_native_determinism():
    return {'version': 'native_grid_strict_v1', 'runtime': configure_determinism(),
            'auxiliary': 'unchanged multiclass blob DC+BCE; deterministic torch scatter/reductions'}


def load_native_refiner(path, device):
    policy = configure_native_determinism()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCHITECTURE or saved['config']['deterministic_runtime'] != policy:
        raise ValueError('Native-grid strict checkpoint or runtime differs')
    model = NativeGridRefiner()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
