"""HRNet-W48 and object-contextual segmentation with native detail.

OCR mechanism follows Yuan et al. arXiv:1909.11065; independent compact adaptation.
Official Microsoft implementation is MIT; attribution in data/pretrained/hrnet-w48/NOTICE.md.
"""
import torch
import torch.nn.functional as F
import timm
from safetensors.torch import load_file
from core import file_hash
from pretrained_model import MEAN, STD

ARCHITECTURE = 'hrnet_w48_ocr_detail_bf16'


def conv_norm_relu(cin, cout, kernel=1):
    return torch.nn.Sequential(torch.nn.Conv2d(cin, cout, kernel, padding=kernel//2, bias=False),
                               torch.nn.BatchNorm2d(cout), torch.nn.ReLU())


def object_gather(features, logits):
    if features.shape[0] != logits.shape[0] or features.shape[-2:] != logits.shape[-2:]:
        raise ValueError('Features and auxiliary logits must share pixel coordinates')
    with torch.autocast(features.device.type, enabled=False):
        weights = logits.float().flatten(2).softmax(-1)
        return torch.bmm(features.float().flatten(2), weights.transpose(1, 2)).unsqueeze(-1)


class ObjectContext(torch.nn.Module):
    def __init__(self, channels=512, key_channels=256):
        super().__init__()
        self.key_channels = key_channels
        self.query = torch.nn.Sequential(conv_norm_relu(channels, key_channels), conv_norm_relu(key_channels, key_channels))
        self.key = torch.nn.Sequential(conv_norm_relu(channels, key_channels), conv_norm_relu(key_channels, key_channels))
        self.value = conv_norm_relu(channels, key_channels)
        self.expand = conv_norm_relu(key_channels, channels)
        self.merge = torch.nn.Sequential(conv_norm_relu(channels*2, channels), torch.nn.Dropout2d(.05))

    def forward(self, features, auxiliary):
        proxy = object_gather(features, auxiliary)
        q, k, v = self.query(features), self.key(proxy), self.value(proxy)
        with torch.autocast(features.device.type, enabled=False):
            affinity = torch.bmm(q.float().flatten(2).transpose(1, 2), k.float().flatten(2))
            affinity = (affinity * self.key_channels**-.5).softmax(-1)
            context = torch.bmm(v.float().flatten(2), affinity.transpose(1, 2))
            context = context.reshape(features.shape[0], self.key_channels, *features.shape[-2:])
        return self.merge(torch.cat([features, self.expand(context)], 1))


class HRNetOCR(torch.nn.Module):
    def __init__(self, architecture=ARCHITECTURE):
        super().__init__()
        if architecture != ARCHITECTURE:
            raise ValueError(architecture)
        self.architecture = architecture
        self.encoder = timm.create_model('hrnet_w48.ms_in1k', pretrained=False, features_only=True,
                                         feature_location='', out_indices=(1, 2, 3, 4))
        assert self.encoder.feature_info.channels() == [48, 96, 192, 384]
        assert self.encoder.feature_info.reduction() == [4, 8, 16, 32]
        self.decoder = torch.nn.ModuleDict({
            'project': conv_norm_relu(720, 512, 3),
            'auxiliary': torch.nn.Sequential(conv_norm_relu(720, 720), torch.nn.Conv2d(720, 4, 1)),
            'context': ObjectContext(),
            'classifier': torch.nn.Conv2d(512, 4, 1),
        })
        self.detail = torch.nn.Sequential(torch.nn.Conv2d(7, 32, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.GELU(), torch.nn.Conv2d(32, 4, 1))
        torch.nn.init.zeros_(self.detail[-1].weight)
        torch.nn.init.zeros_(self.detail[-1].bias)
        self.register_buffer('input_mean', torch.tensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('input_std', torch.tensor(STD).reshape(1, 3, 1, 1))

    def forward_with_aux(self, x):
        h, w = x.shape[-2:]
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
            normalized = (x - self.input_mean) / self.input_std
            padded = F.pad(normalized, (0, (-w)%32, 0, (-h)%32), mode='replicate')
            features = self.encoder(padded)
            combined = torch.cat([features[0]] + [F.interpolate(f, size=features[0].shape[-2:],
                mode='bilinear', align_corners=True) for f in features[1:]], 1)
            auxiliary = self.decoder['auxiliary'](combined)
            pixels = self.decoder['project'](combined)
            pixels = self.decoder['context'](pixels, auxiliary)
            logits = self.decoder['classifier'](pixels)
            logits = F.interpolate(logits.float(), size=padded.shape[-2:], mode='bilinear', align_corners=True)[..., :h, :w]
            auxiliary = F.interpolate(auxiliary.float(), size=padded.shape[-2:], mode='bilinear', align_corners=True)[..., :h, :w]
            logits = logits + self.detail(torch.cat([normalized, logits.softmax(1)], 1)).float()
        return logits, auxiliary

    def forward(self, x):
        return self.forward_with_aux(x)[0]


def initialize_hrnet(model, path, expected_sha256):
    if file_hash(path) != expected_sha256:
        raise ValueError('Pretrained SHA256 differs')
    source = load_file(str(path), device='cpu')
    expected = model.encoder.state_dict()
    missing = set(expected) - set(source)
    counters = {k for k in missing if k.endswith('.num_batches_tracked') and expected[k].ndim == 0 and expected[k].dtype == torch.int64}
    if missing - counters:
        raise ValueError('Missing feature tensors: ' + str(sorted(missing-counters)))
    extra = set(source) - set(expected)
    allowed = ('incre_modules.', 'downsamp_modules.', 'final_layer.', 'classifier.')
    if any(not k.startswith(allowed) for k in extra):
        raise ValueError('Unexpected unused tensors: ' + str(sorted(extra)))
    subset = {k: torch.zeros_like(v) if k in counters else source[k] for k, v in expected.items()}
    for k, v in subset.items():
        if v.shape != expected[k].shape or v.dtype != expected[k].dtype:
            raise ValueError('Feature shape/dtype differs: ' + k)
    model.encoder.load_state_dict(subset, strict=True)
    assert all(torch.equal(v, subset[k]) for k, v in model.encoder.state_dict().items())
    return {'sha256': expected_sha256, 'feature_tensors_loaded': len(subset)-len(counters),
            'all_feature_tensors_identical': True, 'legacy_bn_counters_zeroed': sorted(counters),
            'unused_classification_head_tensors': sorted(extra), 'offline_loading': True}


def build_hrnet_optimizer(model):
    groups = [{'params': model.encoder.parameters(), 'lr': 1e-4},
              {'params': model.decoder.parameters(), 'lr': 1e-3},
              {'params': model.detail.parameters(), 'lr': 1e-3}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    params = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(params) == len(set(params)) and set(params) == {id(p) for p in model.parameters()}
    return optimizer


def hrnet_training_loss(model, x, target):
    from expansion import multiclass_loss
    logits, auxiliary = model.forward_with_aux(x)
    return multiclass_loss(logits, target) + .4 * F.cross_entropy(auxiliary.float(), target)


def load_hrnet(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = HRNetOCR(saved['architecture'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
