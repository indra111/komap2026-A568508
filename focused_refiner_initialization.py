"""Audited weight-only initialization; optimizer/RNG start anew in every matched arm."""
from pathlib import Path
import torch
from core import file_hash
from focused_refiner_model import ARCHITECTURES
from pretrained_model import tensor_fingerprint

ALLOWED = {'upsampled':('cached_four_class_patch_residual_bf16',ARCHITECTURES['upsampled']),
           'native':('cached_four_class_native_grid_strict_residual_bf16',ARCHITECTURES['native'])}


def checked_initialization(root, filename, grid, teacher_sha256, classes):
    root=Path(root).resolve();path=(root/filename).resolve()
    if not path.is_relative_to(root):raise ValueError('Initialization outside project')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['architecture'] not in ALLOWED[grid]:raise ValueError('Initialization grid differs')
    if saved['classes']!=classes or saved['config']['teacher_recipe_sha256']!=teacher_sha256:
        raise ValueError('Initialization teacher or classes differ')
    if not all(torch.isfinite(v).all() for v in saved['model'].values() if v.is_floating_point()):
        raise ValueError('Nonfinite initial parameters')
    info={'file':str(path.relative_to(root)),'sha256':file_hash(path),'source_architecture':saved['architecture'],
          'source_epoch':saved['epoch'],'model_fingerprint':tensor_fingerprint(saved['model']),
          'teacher_recipe_sha256':teacher_sha256,'optimizer_scheduler_rng_reused':False}
    return saved,info


def install_initialization(model,saved,info):
    model.load_state_dict(saved['model'],strict=True)
    if tensor_fingerprint(model.state_dict())!=info['model_fingerprint']:
        raise ValueError('Initial model tensors differ')
