"""Aligned train-only focused batches with matched independent D4/noise streams."""
import copy
import numpy as np
import torch
from error_focus_sampling import FocusIndex, FocusPlanner
from probability_corruption import shift_probability, coarsen_probability


class FocusedCachedSampler:
    def __init__(self, pairs, split_by_id, seed, policy, noise='spatial_noise'):
        if noise not in ('clean', 'spatial_noise'):
            raise ValueError(noise)
        names = [p[0] for p in pairs]
        train_ids = {name for name, split in split_by_id.items() if split == 'train'}
        if set(names) != train_ids or len(names) != len(train_ids):
            raise ValueError('Exactly the allowed training images are required')
        for name, image, probability, target in pairs:
            if split_by_id.get(name) != 'train':
                raise ValueError('Validation/test image in training sampler')
            if image.shape != (*target.shape, 3) or image.dtype != np.uint8:
                raise ValueError('Image and target are not aligned')
            if probability.shape != (4, *target.shape) or probability.dtype != np.float32:
                raise ValueError('Probability and target are not aligned')
            if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(0),1.,atol=2e-5):
                raise ValueError('Expected finite probability simplex')
        self.pairs = pairs
        self.index = FocusIndex([{'id':name,'split':split_by_id[name],'target':target} for name,_,_,target in pairs])
        self.planner = FocusPlanner(self.index, seed, policy)
        self.augmentation_rng = np.random.default_rng(seed+337)
        self.noise_rng = np.random.default_rng(seed+991)
        self.noise = noise
        self.noise_counts = {'clean':0,'shift':0,'coarsen':0}
        self.last_metadata = []

    @property
    def draws(self): return self.planner.draws

    @property
    def seen(self): return self.planner.seen

    def batch(self, count):
        if count < 1: raise ValueError('Positive batch required')
        images, probabilities, targets = [], [], []
        self.last_metadata = []
        for _ in range(count):
            row = self.planner.draw(); x,y,w,h = row['xywh']
            name,image,probability,target = self.pairs[row['index']]
            image=image[y:y+h,x:x+w];probability=probability[:,y:y+h,x:x+w];target=target[y:y+h,x:x+w]
            turns=int(self.augmentation_rng.integers(4));flip=bool(self.augmentation_rng.integers(2))
            image=np.rot90(image,turns,axes=(0,1));probability=np.rot90(probability,turns,axes=(1,2));target=np.rot90(target,turns,axes=(0,1))
            if flip:
                image=np.flip(image,1);probability=np.flip(probability,2);target=np.flip(target,1)
            images.append(torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255)
            probabilities.append(torch.from_numpy(np.ascontiguousarray(probability)))
            targets.append(torch.from_numpy(np.ascontiguousarray(target)).long())
            self.last_metadata.append({**row,'turns':turns,'flip':flip})
        probability=torch.stack(probabilities);noisy=probability.clone()
        for i in range(count):
            value=self.noise_rng.random() if self.noise=='spatial_noise' else 0.
            if value < .5:kind='clean'
            elif value < .75:
                kind='shift';offsets=[(dy,dx) for dy in range(-2,3) for dx in range(-2,3) if (dy,dx)!=(0,0)]
                dy,dx=offsets[int(self.noise_rng.integers(len(offsets)))]
                noisy[i:i+1]=shift_probability(probability[i:i+1],dy,dx)
            else:
                kind='coarsen';noisy[i:i+1]=coarsen_probability(probability[i:i+1])
            self.noise_counts[kind]+=1;self.last_metadata[i]['noise']=kind
        if not torch.isfinite(noisy).all() or (noisy < 0).any() or not torch.allclose(noisy.sum(1),torch.ones_like(noisy[:,0]),atol=2e-6,rtol=1e-5):
            raise ValueError('Augmentation changed probability mass')
        return torch.stack(images),noisy,torch.stack(targets)

    def state_dict(self):
        return {'planner':self.planner.state_dict(),'draws':self.draws,'seen':sorted(self.seen),
            'augmentation_rng':copy.deepcopy(self.augmentation_rng.bit_generator.state),
            'noise_rng':copy.deepcopy(self.noise_rng.bit_generator.state),'noise_method':self.noise,
            'noise_counts':dict(self.noise_counts)}

    def load_state_dict(self,state):
        if state['noise_method']!=self.noise:
            raise ValueError('Cannot change noise policy while resuming')
        self.planner.load_state_dict(state['planner'])
        assert self.draws==state['draws'] and sorted(self.seen)==state['seen']
        self.augmentation_rng.bit_generator.state=copy.deepcopy(state['augmentation_rng'])
        self.noise_rng.bit_generator.state=copy.deepcopy(state['noise_rng'])
        self.noise_counts=dict(state['noise_counts'])
        assert sum(self.noise_counts.values())==self.draws
