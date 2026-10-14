"""Versioned strict-runtime heads for matched focused-sampling experiments."""
import torch
from cached_refiner import CachedRefiner
from native_grid_refiner import NativeGridRefiner
from native_grid_deterministic import configure_native_determinism

ARCHITECTURES = {'upsampled':'focused_upsampled_strict_residual_bf16_v1',
                 'native':'focused_native_strict_residual_bf16_v1'}


def create_model(grid):
    if grid not in ARCHITECTURES: raise ValueError(grid)
    return CachedRefiner() if grid == 'upsampled' else NativeGridRefiner()


def configure_runtime():
    return {'version':'focused_sampling_strict_v1','base':configure_native_determinism()}


def load_focused_refiner(path, device):
    runtime=configure_runtime()
    saved=torch.load(path,map_location='cpu',weights_only=True)
    config=saved['config'];grid=config['grid']
    if saved['architecture']!=ARCHITECTURES[grid] or config['deterministic_runtime']!=runtime:
        raise ValueError('Focused architecture or runtime differs')
    model=create_model(grid);model.load_state_dict(saved['model'],strict=True)
    return model.to(torch.device(device)).eval(),saved
