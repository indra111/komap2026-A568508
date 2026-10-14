"""Train-only crop geometry for separately testing core and small-region exposure."""
import copy
import hashlib
import json
import numpy as np
from image_processing import cv2
from spatial_error_diagnostics import boundary_distance

POLICIES = {'uniform': [('uniform', 1.)],
            'core50': [('uniform', .5), ('core_si', .5)],
            'small50': [('uniform', .5), ('tiny_eut', .25), ('small_al3ni', .25)]}
PATCH = 64


def small_mask(target, class_id, max_area):
    _, labels = cv2.connectedComponents((target == class_id).astype(np.uint8), connectivity=8)
    counts = np.bincount(labels.ravel())
    allowed = (counts > 0) & (counts <= max_area)
    allowed[0] = False
    return allowed[labels]


def masks(target):
    return {'core_si': (target == 3) & (boundary_distance(target) > 8),
            'tiny_eut': small_mask(target, 3, 16), 'small_al3ni': small_mask(target, 2, 64)}


class FocusIndex:
    def __init__(self, rows):
        if not rows or any(row['split'] != 'train' for row in rows):
            raise ValueError('Only nonempty training rows are allowed')
        self.names = [row['id'] for row in rows]
        if len(set(self.names)) != len(rows):
            raise ValueError('Duplicate image IDs')
        self.targets = [row['target'] for row in rows]
        if any(t.ndim != 2 or min(t.shape) < PATCH or t.min() < 0 or t.max() > 3 for t in self.targets):
            raise ValueError('Expected four-class native masks at least64px per axis')
        self.masks = [masks(t) for t in self.targets]
        self.positions = {kind: [np.flatnonzero(m[kind]).astype(np.int32) for m in self.masks]
                          for kind in ('core_si', 'tiny_eut', 'small_al3ni')}
        self.eligible = {kind: [i for i, positions in enumerate(items) if len(positions)]
                         for kind, items in self.positions.items()}
        digest = hashlib.sha256()
        for name, target in zip(self.names, self.targets):
            digest.update(json.dumps([name, list(target.shape)]).encode())
            digest.update(target.astype(np.uint8).tobytes())
        self.signature = digest.hexdigest()


class FocusPlanner:
    def __init__(self, index, seed, policy):
        if policy not in POLICIES:
            raise ValueError(policy)
        if any(kind != 'uniform' and not index.eligible[kind] for kind, _ in POLICIES[policy]):
            raise ValueError('Requested focus has no eligible training image')
        self.index, self.policy = index, policy
        self.rng = np.random.default_rng(seed)
        self.draws = 0
        self.counts = {kind: 0 for kind, _ in POLICIES[policy]}
        self.seen = set()

    def draw(self):
        value = self.rng.random()
        cumulative = 0.
        for kind, weight in POLICIES[self.policy]:
            cumulative += weight
            if value < cumulative:
                break
        if kind == 'uniform':
            index = int(self.rng.integers(len(self.index.names)))
            h, w = self.index.targets[index].shape
            y, x = int(self.rng.integers(h-PATCH+1)), int(self.rng.integers(w-PATCH+1))
            anchor = None
        else:
            eligible = self.index.eligible[kind]
            index = eligible[int(self.rng.integers(len(eligible)))]
            h, w = self.index.targets[index].shape
            candidates = self.index.positions[kind][index]
            ay, ax = divmod(int(candidates[int(self.rng.integers(len(candidates)))]), w)
            y = int(self.rng.integers(max(0, ay-PATCH+1), min(ay, h-PATCH)+1))
            x = int(self.rng.integers(max(0, ax-PATCH+1), min(ax, w-PATCH)+1))
            anchor = [ax, ay]
        self.draws += 1
        self.counts[kind] += 1
        self.seen.add(self.index.names[index])
        return {'id': self.index.names[index], 'index': index, 'kind': kind,
                'xywh': [x, y, PATCH, PATCH], 'anchor_xy': anchor}

    def state_dict(self):
        return {'rng': copy.deepcopy(self.rng.bit_generator.state), 'draws': self.draws,
                'counts': dict(self.counts), 'seen': sorted(self.seen),
                'policy': self.policy, 'index_signature': self.index.signature}

    def load_state_dict(self, state):
        if state['policy'] != self.policy or state['index_signature'] != self.index.signature:
            raise ValueError('Cannot resume with changed sampling policy or training data')
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.draws = state['draws']; self.counts = dict(state['counts']); self.seen = set(state['seen'])
