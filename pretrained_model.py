"""Four-phase U-Net with explicit preprocessing and checked encoder transfer."""
from pathlib import Path
import hashlib

import torch
from core import file_hash, make_model

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class NormalizedUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_model()  # Always create the same random decoder/head first.
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))

    def forward(self, x):
        return self.net((x - self.input_mean) / self.input_std)


def initialize_encoder(model, path, expected_sha256):
    actual = file_hash(path)
    if actual != expected_sha256:
        raise ValueError('Pretrained file SHA256 mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a plain tensor state_dict')
    expected = model.net.encoder.state_dict()
    # Torchvision classification head is deliberately excluded; never transfer phase labels.
    extra = set(state) - set(expected)
    missing = set(expected) - set(state)
    # Older official checkpoints predate BN's counter buffer. It is not a learned
    # weight; initialize only these scalar int64 counters to zero, explicitly.
    legacy_counters = {k for k in missing if k.endswith('.num_batches_tracked')
                       and expected[k].dtype == torch.int64 and expected[k].ndim == 0}
    if missing - legacy_counters or extra - {'fc.weight', 'fc.bias'}:
        raise ValueError(f'Encoder keys mismatch: missing={sorted(missing)}, unexpected={sorted(extra)}')
    subset = {k: torch.zeros_like(expected[k]) if k in legacy_counters else state[k] for k in expected}
    for k, value in subset.items():
        if value.shape != expected[k].shape or value.dtype != expected[k].dtype:
            raise ValueError('Encoder tensor shape/dtype mismatch: ' + k)
    model.net.encoder.load_state_dict(subset, strict=True)
    for k, value in model.net.encoder.state_dict().items():
        if not torch.equal(value.cpu(), subset[k]):
            raise ValueError('Encoder transfer failed: ' + k)
    return {'sha256': actual, 'tensors_loaded': len(subset)-len(legacy_counters),
            'legacy_bn_counters_initialized_to_zero': sorted(legacy_counters), 'skipped_classifier_keys': sorted(extra),
            'all_encoder_tensors_identical': True, 'decoder_and_phase_head_transferred': False}


def build_optimizer(model):
    return torch.optim.AdamW([
        {'params': model.net.encoder.parameters(), 'lr': 1e-4},
        {'params': list(model.net.decoder.parameters()) + list(model.net.segmentation_head.parameters()), 'lr': 1e-3},
    ], weight_decay=1e-4)


def tensor_fingerprint(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(str(v.dtype).encode())
        h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_saved(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != 'normalized_unet_resnet34':
        raise ValueError('Unexpected architecture')
    model = NormalizedUNet().to(device)
    model.load_state_dict(saved['model'], strict=True)
    return model, saved
