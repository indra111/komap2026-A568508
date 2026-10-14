"""Selected fixed evaluation modes; avoid extra unplanned untrained-reference modes."""
from pathlib import Path
import numpy as np
import torch
from core import file_hash, save_mask, decode_mask, score_image
from context_model import summarize_scores
from cached_refiner_eval import predict_refined
from refiner_d4 import predict_refined_d4
from dense_crf import PRESETS, refine
from rank_iou import rank_iou
from train_prediction_cache import valid_probabilities

MODES = ('single', 'd4', 'single_crf', 'd4_crf')
CRF_PRESET = 'micro_edges'
PRUNING_PROB = .5
STRENGTH = 1.


def decode_probability(probability, original_rgb, mode):
    if mode not in MODES:
        raise ValueError('Unplanned evaluation mode')
    valid_probabilities(probability, original_rgb.shape[:2])
    if original_rgb.dtype != np.uint8 or original_rgb.shape != (*probability.shape[1:], 3):
        raise ValueError('Expected aligned original uint8 RGB')
    if mode.endswith('_crf'):
        refined = refine(probability, original_rgb, PRESETS[CRF_PRESET])
        return rank_iou(refined, pruning_prob=PRUNING_PROB)[0].astype(np.uint8)
    return probability.argmax(0).astype(np.uint8)


def evaluate_selected_branch(model, pairs, original_images, spec, device, output, modes, progress=None):
    """Truth is used for scoring only; postprocessing always uses original RGB."""
    modes=tuple(modes)
    if not modes or len(set(modes))!=len(modes) or not set(modes)<=set(MODES):
        raise ValueError("Unplanned or duplicate modes")
    probability_modes=tuple(m for m in ("single","d4") if any(("d4" if mode.startswith("d4") else "single")==m for mode in modes))
    output = Path(output)
    if output.exists():
        raise FileExistsError('Never overwrite an existing branch evaluation')
    identifiers = [row[0] for row in pairs]
    if len(set(identifiers)) != len(identifiers) or set(original_images) != set(identifiers):
        raise ValueError('Unaligned or duplicate validation IDs')
    if any(Path(name).name != name or name in ('.', '..') or '\\' in name for name in identifiers):
        raise ValueError('Unsafe image ID')
    assert spec['absent_class_policy'] == 'skip' and len(spec['classes']) == 4
    output.mkdir(parents=True, exist_ok=False)
    for name in (*modes, *(m+'_probability' for m in probability_modes)):
        (output/name).mkdir()
    scores = {mode: [] for mode in modes}
    predictions = {mode: [] for mode in modes}
    alternatives = {mode: {rule: [] for rule in ('skip', 'one', 'zero')} for mode in modes}
    arrays = {m:[] for m in probability_modes}
    model.eval()
    for index, (name, image, teacher, truth) in enumerate(pairs):
        original = original_images[name]
        assert original.shape == image.shape and truth.shape == image.shape[:2]
        with torch.inference_mode():
            probabilities = {m:(predict_refined_d4 if m=='d4' else predict_refined)(model,image,teacher,device,STRENGTH) for m in probability_modes}
        for mode, probability in probabilities.items():
            valid_probabilities(probability, truth.shape)
            path = output/(mode+'_probability')/(name+'.npy')
            np.save(path, probability.astype(np.float32, copy=False), allow_pickle=False)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), probability)
            arrays[mode].append({'id': name, 'file': str(path.relative_to(output)),
                                 'sha256': file_hash(path), 'shape': list(probability.shape)})
        for mode in modes:
            source = 'd4' if mode.startswith('d4') else 'single'
            prediction = decode_probability(probabilities[source], original, mode)
            path = output/mode/(name+'.png')
            save_mask(path, prediction, spec)
            np.testing.assert_array_equal(prediction, decode_mask(path, spec))
            scores[mode].append({'id': name, **score_image(prediction, truth, 'skip')})
            for rule in alternatives[mode]:
                alternatives[mode][rule].append(score_image(prediction, truth, rule)['miou'])
            predictions[mode].append({'id': name, 'file': str(path.relative_to(output)),
                                      'sha256': file_hash(path), 'shape': list(prediction.shape)})
        if progress:
            progress(index+1, len(pairs), name)
    return {mode: {'metrics': summarize_scores(scores[mode], alternatives[mode], spec),
                   'predictions': predictions[mode],
                   'probabilities': arrays['d4' if mode.startswith('d4') else 'single']}
            for mode in modes}
