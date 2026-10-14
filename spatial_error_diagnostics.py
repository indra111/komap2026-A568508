"""Descriptive spatial error attribution; no fitting or prediction modification."""
import numpy as np
from image_processing import cv2

RADII = (0, 1, 2, 4, 8, 16)
AREA_LIMITS = (16, 64, 256, 1024)


def boundary_distance(truth):
    """Euclidean pixels from either side of internal 4-neighbor label changes."""
    boundary = np.zeros(truth.shape, dtype=np.uint8)
    change = truth[:-1] != truth[1:]
    boundary[:-1] |= change
    boundary[1:] |= change
    change = truth[:, :-1] != truth[:, 1:]
    boundary[:, :-1] |= change
    boundary[:, 1:] |= change
    if not boundary.any():
        return np.full(truth.shape, np.inf, dtype=np.float32)
    return cv2.distanceTransform(1-boundary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def confusion(truth, prediction, selected):
    return np.bincount((4*truth[selected]+prediction[selected]).astype(np.int64), minlength=16).reshape(4, 4)


def distance_rows(truth, prediction):
    distance = boundary_distance(truth)
    result = []
    low = -1
    for high in (*RADII, None):
        selected = distance > low if high is None else ((distance > low) & (distance <= high))
        matrix = confusion(truth, prediction, selected)
        result.append({'lower_exclusive': low, 'upper_inclusive': high,
                       'pixels': int(matrix.sum()),
                       'errors': int(matrix.sum()-np.trace(matrix)),
                       'confusion': matrix.tolist()})
        low = high
    assert sum(row['pixels'] for row in result) == truth.size
    return result, distance


def component_rows(truth, prediction):
    """Recall of visible 8-connected GT regions, not physical grain counts."""
    result = []
    for class_id in range(4):
        mask = (truth == class_id).astype(np.uint8)
        count, labels = cv2.connectedComponents(mask, connectivity=8, ltype=cv2.CV_32S)
        sizes = np.bincount(labels[mask.astype(bool)], minlength=count)[1:]
        matrix = np.bincount((labels[mask.astype(bool)]*4+prediction[mask.astype(bool)]).astype(np.int64),
                             minlength=count*4).reshape(count, 4)[1:]
        assert np.array_equal(matrix.sum(1), sizes)
        low = 0
        for high in (*AREA_LIMITS, None):
            selected = sizes > low if high is None else ((sizes > low) & (sizes <= high))
            selected_matrix = matrix[selected]
            selected_sizes = sizes[selected]
            true_positives = selected_matrix[:, class_id]
            # A strict majority avoids arbitrarily breaking ties in class IDs.
            majority_other = ((selected_matrix[:, [c for c in range(4) if c != class_id]] > selected_sizes[:, None]/2).any(1)
                              if len(selected_sizes) else np.zeros(0, dtype=bool))
            result.append({'class_id': class_id, 'area_lower_exclusive': low, 'area_upper_inclusive': high,
                           'components': int(selected.sum()), 'truth_pixels': int(selected_sizes.sum()),
                           'correct_pixels': int(true_positives.sum()),
                           'missed_components': int((true_positives == 0).sum()),
                           'strict_majority_other_components': int(majority_other.sum()),
                           'prediction_counts': selected_matrix.sum(0).tolist()})
            low = high
    assert sum(row['truth_pixels'] for row in result) == truth.size
    assert sum(row['correct_pixels'] for row in result) == int((truth == prediction).sum())
    return result
