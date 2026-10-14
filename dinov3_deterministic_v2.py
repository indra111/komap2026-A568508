"""Explicit, checkpointed deterministic runtime for new Large experiments.

The loss retains original four-class CE + batch Dice. CE uses log-softmax,
gather and mean to avoid CUDA NLLLoss2d's nondeterministic reduction.
"""
import os
WORKSPACE = ':4096:8'
if os.environ.get('CUBLAS_WORKSPACE_CONFIG', WORKSPACE) != WORKSPACE:
    raise RuntimeError('Conflicting cuBLAS workspace policy')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = WORKSPACE
import torch
import torch.nn.functional as F


def configure_determinism():
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return runtime_policy()


def runtime_policy():
    return {'version': 'strict_dinov3_large_v2', 'torch': str(torch.__version__),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'loss': 'original_four_class_mean_CE_via_logsoftmax_gather_plus_batch_Dice'}


def multiclass_loss(logits, target):
    values = logits.float()
    ce = -values.log_softmax(1).gather(1, target.unsqueeze(1)).mean()
    probs = values.softmax(1)
    onehot = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probs * onehot).sum((0, 2, 3))
    denominator = (probs + onehot).sum((0, 2, 3))
    return ce + 1 - ((2 * intersection + 1e-5) / (denominator + 1e-5)).mean()


def load_dinov3_large_deterministic_v2(path, device):
    from dinov3_large_model import load_dinov3_large
    policy = configure_determinism()
    model, saved = load_dinov3_large(path, device)
    if saved['config'].get('deterministic_runtime') != policy:
        raise ValueError('Checkpoint deterministic runtime differs')
    return model, saved
