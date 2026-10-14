"""Cross-source pixel contrast as a train-only auxiliary to the ConvNeXt parent.

The loss samples original-resolution labels, includes all four phases, and
never uses validation/test masks or an external memory bank. Inspired by
Wang et al., ICCV2021; this is a bounded adaptation, not their full method.
"""
import torch
import torch.nn.functional as F
from advanced_model import AdvancedUNet
from expansion import ExpandedSampler, multiclass_loss
from pretrained_model import build_optimizer as base_optimizer

ARCHITECTURE = 'unet_convnext_tiny_pixel_contrast'


class ContrastSampler(ExpandedSampler):
    def pair(self, i):
        self.source_indices.append(i)
        return super().pair(i)

    def batch(self, count):
        self.source_indices = []
        x, y = super().batch(count)
        assert len(self.source_indices) == count
        return x, y, torch.tensor(self.source_indices, dtype=torch.long)


class ContrastNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = AdvancedUNet('unet_convnext_tiny')
        channels = self.base.net.segmentation_head[0].in_channels
        self.projection = torch.nn.Sequential(torch.nn.Linear(channels, 64), torch.nn.GELU(),
                                              torch.nn.Linear(64, 64))

    @property
    def net(self):
        return self.base.net

    def forward(self, x, auxiliary=False):
        features = self.net.encoder((x - self.base.input_mean) / self.base.input_std)
        decoded = self.net.decoder(features)
        logits = self.net.segmentation_head(decoded)
        return (logits, decoded) if auxiliary else logits


def build_optimizer(model):
    optimizer = base_optimizer(model)
    optimizer.add_param_group({'params': model.projection.parameters(), 'lr': 1e-3})
    return optimizer


def contrast_loss(embeddings, labels, sources, temperature=.1):
    """Same-phase positives must come from a different original source image."""
    if len(embeddings) < 2:
        return embeddings.sum() * 0
    z = F.normalize(embeddings.float(), dim=1, eps=1e-6)
    similarities = z @ z.T / temperature
    same = labels[:, None] == labels[None, :]
    positive = same & (sources[:, None] != sources[None, :])
    negatives = ~same
    allowed = positive | negatives
    anchors = positive.any(1)
    if not anchors.any():
        return embeddings.sum() * 0
    similarities = similarities[anchors]
    positive, allowed = positive[anchors], allowed[anchors]
    denominator = torch.logsumexp(similarities.masked_fill(~allowed, -torch.inf), dim=1)
    numerator = (similarities * positive).sum(1) / positive.sum(1)
    losses = denominator - numerator
    # Weight phases equally even when a phase is absent from one crop.
    anchor_labels = labels[anchors]
    return torch.stack([losses[anchor_labels == c].mean() for c in anchor_labels.unique()]).mean()


def sampled_contrast(model, decoded, target, source_ids, samples_per_phase=64):
    if len(source_ids.unique()) < 2:
        return decoded.sum() * 0
    features, labels, sources = [], [], []
    for batch in range(target.shape[0]):
        pixels = decoded[batch].permute(1, 2, 0).reshape(-1, decoded.shape[1])
        truth = target[batch].reshape(-1)
        for phase in range(4):
            candidates = torch.nonzero(truth == phase, as_tuple=False).flatten()
            if len(candidates) == 0:
                continue
            selected = candidates[torch.randperm(len(candidates), device=target.device)[:samples_per_phase]]
            features.append(pixels[selected])
            labels.append(torch.full((len(selected),), phase, dtype=torch.long, device=target.device))
            sources.append(source_ids[batch].expand(len(selected)))
    if not features:
        return decoded.sum() * 0
    embeddings = model.projection(torch.cat(features))
    return contrast_loss(embeddings, torch.cat(labels), torch.cat(sources))


def training_loss(model, output, target, source_ids, method):
    logits, decoded = output
    base = multiclass_loss(logits, target)
    if method == 'contrast_control':
        return base
    if method != 'pixel_contrast':
        raise ValueError(method)
    return base + .05 * sampled_contrast(model, decoded, target, source_ids)


def load_contrast(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['architecture'] != ARCHITECTURE:
        raise ValueError('Unexpected contrast architecture')
    model = ContrastNet()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved
