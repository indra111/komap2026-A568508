"""Local detail + spatially aligned global context; no external weights or labels.

Inspired by OCELOT's context principle, not a reproduction of its tissue/cell model.
The original baseline implementation remains unchanged.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from core import (decode_mask, image_tensor, make_model, read_image, save_mask,
                  score_image, starts)
from expansion import ExpandedSampler


def global_view(image, mask=None, patch=512, size=256):
    """Letterbox the whole padded image, retaining aspect ratio and coordinate map."""
    h, w = image.shape[:2]
    ph, pw = max(h, patch), max(w, patch)
    padded = np.pad(image, ((0, ph-h), (0, pw-w), (0, 0)), mode='edge')
    scale = size / max(ph, pw)
    rh, rw = max(1, round(ph*scale)), max(1, round(pw*scale))
    top, left = (size-rh)//2, (size-rw)//2
    resized = np.array(Image.fromarray(padded).resize((rw, rh), Image.Resampling.BILINEAR))
    view = np.pad(resized, ((top, size-rh-top), (left, size-rw-left), (0, 0)), mode='edge')
    coarse = np.full((size, size), 255, dtype=np.uint8)
    if mask is not None:
        labels = np.full((ph, pw), 255, dtype=np.uint8)
        labels[:h, :w] = mask
        coarse[top:top+rh, left:left+rw] = np.array(
            Image.fromarray(labels).resize((rw, rh), Image.Resampling.NEAREST))
    geometry = (ph, pw, top, left, rh, rw, size)
    return padded, view, coarse, geometry


def crop_box(y, x, patch, geometry):
    """Normalized edge coordinates in the global letterbox, before augmentation."""
    ph, pw, top, left, rh, rw, size = geometry
    return np.array([(left+x*rw/pw)/size, (top+y*rh/ph)/size,
                     (left+(x+patch)*rw/pw)/size, (top+(y+patch)*rh/ph)/size], np.float32)


def transform_box(box, turns, flip):
    x1, y1, x2, y2 = map(float, box)
    for _ in range(turns % 4):
        x1, y1, x2, y2 = y1, 1-x2, y2, 1-x1
    if flip:
        x1, x2 = 1-x2, 1-x1
    return np.array([x1, y1, x2, y2], np.float32)


def transform_array(a, turns, flip):
    out = np.rot90(a, turns)
    return np.ascontiguousarray(np.flip(out, 1) if flip else out)


class ContextSampler(ExpandedSampler):
    def __init__(self, rows, spec, patch, seed, global_size=256):
        super().__init__(rows, spec, patch, seed, augmented=False, external=False)
        self.global_size = global_size
        self.globals = {}

    def batch(self, count):
        xs, ys, gs, boxes, coarse_labels = [], [], [], [], []
        for _ in range(count):
            # Exactly the existing control's random choices, order and transforms.
            i = int(self.rng.integers(len(self.items)))
            self.seen.add(self.items[i]['id'])
            fy, fx = self.rng.random(2)
            turns, flip = int(self.rng.integers(4)), self.rng.random() < .5
            image, mask = self.pair(i)
            if i not in self.globals:
                self.globals[i] = global_view(image, mask, self.patch, self.global_size)
            padded, view, coarse, geometry = self.globals[i]
            ph, pw = padded.shape[:2]
            y, x = min(int(fy*(ph-self.patch+1)), ph-self.patch), min(int(fx*(pw-self.patch+1)), pw-self.patch)
            labels = np.pad(mask, ((0, ph-mask.shape[0]), (0, pw-mask.shape[1])), mode='edge')
            xs.append(image_tensor(transform_array(padded[y:y+self.patch, x:x+self.patch], turns, flip)))
            ys.append(torch.from_numpy(transform_array(labels[y:y+self.patch, x:x+self.patch], turns, flip)).long())
            gs.append(image_tensor(transform_array(view, turns, flip)))
            boxes.append(torch.from_numpy(transform_box(crop_box(y, x, self.patch, geometry), turns, flip)))
            coarse_labels.append(torch.from_numpy(transform_array(coarse, turns, flip)).long())
            self.draws += 1
        return tuple(torch.stack(a) for a in (xs, ys, gs, boxes, coarse_labels))


def align_context(features, boxes, height, width):
    """Sample at local pixel centers with align_corners=False (edge-normalized ROI)."""
    boxes = boxes.float()
    u = (torch.arange(width, device=features.device, dtype=torch.float32)+.5)/width
    v = (torch.arange(height, device=features.device, dtype=torch.float32)+.5)/height
    x = boxes[:, 0, None] + (boxes[:, 2]-boxes[:, 0])[:, None]*u
    y = boxes[:, 1, None] + (boxes[:, 3]-boxes[:, 1])[:, None]*v
    grid = torch.stack((x[:, None, :].expand(-1,height,-1),
                        y[:, :, None].expand(-1,-1,width)), dim=-1)*2-1
    # Keep interpolation in float32 for stable sampling under AMP.
    with torch.autocast(device_type=features.device.type, enabled=False):
        return F.grid_sample(features.float(), grid, mode='bilinear',
                             padding_mode='border', align_corners=False)


class ContextUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Construct first: local initialization matches the baseline at the same seed.
        self.local = make_model()
        blocks, channels = [], 3
        for out in (32, 64, 128):
            blocks += [nn.Conv2d(channels, out, 3, stride=2, padding=1, bias=False),
                       nn.GroupNorm(8, out), nn.SiLU()]
            channels = out
        self.context_encoder = nn.Sequential(*blocks)
        self.fusion = nn.Conv2d(128, 512, 1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        self.coarse_head = nn.Conv2d(128, 4, 1)

    def encode_context(self, global_image):
        return self.context_encoder(global_image)

    def forward(self, local_image, global_image=None, boxes=None, context_features=None, auxiliary=False):
        if boxes is None:
            raise ValueError('Spatial crop coordinates are required')
        features = self.local.encoder(local_image)
        if context_features is None:
            context_features = self.encode_context(global_image)
        aligned = align_context(context_features, boxes, *features[-1].shape[-2:])
        features[-1] = features[-1] + self.fusion(aligned).to(features[-1].dtype)
        logits = self.local.segmentation_head(self.local.decoder(features))
        if auxiliary:
            coarse = F.interpolate(self.coarse_head(context_features).float(),
                                   size=global_image.shape[-2:], mode='bilinear', align_corners=False)
            return logits, coarse
        return logits


@torch.inference_mode()
def predict_probabilities(model, image, device, patch=512, overlap=128, global_size=256, amp=True):
    if patch < 32 or patch % 32 or not 0 <= overlap < patch:
        raise ValueError('Invalid patch or overlap')
    model.eval()
    h, w = image.shape[:2]
    padded, view, _, geometry = global_view(image, patch=patch, size=global_size)
    scores = np.zeros((4, *padded.shape[:2]), np.float32)
    weights = np.zeros(padded.shape[:2], np.float32)
    axis = np.linspace(-1., 1., patch, dtype=np.float32)
    window = np.maximum(np.outer(np.exp(-axis**2/.5), np.exp(-axis**2/.5)), .001)
    has_context = isinstance(model, ContextUNet)
    context = None
    with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda'):
        if has_context:
            context = model.encode_context(image_tensor(view).unsqueeze(0).to(device))
        for y in starts(padded.shape[0], patch, patch-overlap):
            for x in starts(padded.shape[1], patch, patch-overlap):
                tensor = image_tensor(padded[y:y+patch, x:x+patch]).unsqueeze(0).to(device)
                if has_context:
                    box = torch.from_numpy(crop_box(y,x,patch,geometry)).unsqueeze(0).to(device)
                    logits = model(tensor, boxes=box, context_features=context)
                else:
                    logits = model(tensor)
                probs = logits.float().softmax(1)[0].cpu().numpy()
                scores[:, y:y+patch, x:x+patch] += probs * window
                weights[y:y+patch, x:x+patch] += window
    if (weights <= 0).any():
        raise RuntimeError('Inference left uncovered pixels')
    return (scores / weights[None])[:, :h, :w]


def summarize_scores(results, alternatives, spec):
    if not results:
        raise ValueError('No validation images')
    means = []
    for c in range(4):
        v = [r['iou_per_class'][c] for r in results if r['iou_per_class'][c] is not None]
        means.append(float(np.mean(v)) if v else None)
    return {'mean_image_miou': float(np.mean([r['miou'] for r in results])),
            'class_mean_iou': means, 'images': results,
            'metric_rule_status': spec['metric_rule_status'], 'absent_class_policy': spec['absent_class_policy'],
            'alternative_absent_policy_miou': {p:float(np.mean(v)) for p,v in alternatives.items()}}


def evaluate_context(model, rows, spec, device, patch=512, overlap=128, output=None, probability_output=None):
    results, alternatives = [], {p:[] for p in ('skip','one','zero')}
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    if probability_output:
        Path(probability_output).mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row['split'] != 'val':
            continue
        probs = predict_probabilities(model, read_image(row['image']), device, patch, overlap)
        pred, target = probs.argmax(0).astype(np.uint8), decode_mask(row['mask'], spec)
        results.append({'id':row['id'], **score_image(pred,target,spec['absent_class_policy'])})
        for p in alternatives:
            alternatives[p].append(score_image(pred,target,p)['miou'])
        if output:
            out = Path(output)/(row['id']+'.png')
            save_mask(out,pred,spec)
            np.testing.assert_array_equal(pred,decode_mask(out,spec))
        if probability_output:
            np.save(Path(probability_output)/(row['id']+'.npy'),probs,allow_pickle=False)
    return summarize_scores(results, alternatives, spec)


def load_run_model(folder, device):
    saved = torch.load(Path(folder)/'best.pth',map_location='cpu',weights_only=True)
    architecture = saved.get('architecture')
    if architecture == 'unet_resnet34_local_global':
        model = ContextUNet()
    elif architecture == 'unet_resnet34':
        model = make_model()
    else:
        raise ValueError(f'Unknown architecture: {architecture}')
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval(), saved
