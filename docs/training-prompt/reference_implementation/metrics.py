"""Single-class detection metrics; all coordinates are model-input xyxy pixels."""
import numpy as np
from ultralytics.utils.metrics import ap_per_class


def match_boxes(predictions, targets, threshold):
    """Confidence-ordered one-to-one matching; returns prediction and GT masks."""
    predictions = np.asarray(predictions).reshape(-1, 6)
    targets = np.asarray(targets).reshape(-1, 4)
    correct = np.zeros(len(predictions), dtype=bool)
    found = np.zeros(len(targets), dtype=bool)
    for index in np.argsort(-predictions[:, 4], kind='stable'):
        box = predictions[index, :4]
        intersection = np.maximum(0, np.minimum(box[2:], targets[:, 2:]) - np.maximum(box[:2], targets[:, :2])).prod(1)
        area = np.maximum(0, box[2:] - box[:2]).prod()
        gt_area = np.maximum(0, targets[:, 2:] - targets[:, :2]).prod(1)
        overlap = intersection / np.maximum(area + gt_area - intersection, 1e-9)
        overlap[found] = -1
        if len(overlap):
            best = overlap.argmax()
            if overlap[best] >= threshold - 1e-8:
                correct[index] = True
                found[best] = True
    return correct, found


def summarize_detections(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError('Prediction and target frames must align')
    thresholds = np.linspace(.5, .95, 10)
    predictions = [np.asarray(p, dtype=float).reshape(-1, 6) for p in predictions]
    targets = [np.asarray(t, dtype=float).reshape(-1, 4) for t in targets]
    ground_truths = sum(len(t) for t in targets)
    correct = [np.stack([match_boxes(p, t, threshold)[0] for threshold in thresholds], axis=1) for p, t in zip(predictions, targets)]
    scores = np.concatenate([p[:, 4] for p in predictions]) if predictions else np.empty(0)
    true_positives = np.concatenate(correct) if correct else np.empty((0, 10), bool)
    if ground_truths and len(scores):
        ap = ap_per_class(true_positives, scores, np.zeros(len(scores)), np.zeros(ground_truths))[5]
        ap50, map5095 = float(ap[:, 0].mean()), float(ap.mean())
    else:
        ap50, map5095 = 0., 0.
    fixed = [p[p[:, 4] >= .25] for p in predictions]
    fixed_tp = sum(int(match_boxes(p, t, .5)[0].sum()) for p, t in zip(fixed, targets))
    fixed_count = sum(len(p) for p in fixed)
    candidate_tp = sum(int(x[:, 0].sum()) for x in correct)
    top20_tp = sum(int(match_boxes(p[np.argsort(-p[:, 4])[:20]], t, .5)[0].sum()) for p, t in zip(predictions, targets))
    return {
        'map50': ap50, 'map50_95': map5095,
        'precision_conf025': fixed_tp / fixed_count if fixed_count else 0.,
        'recall_conf025': fixed_tp / ground_truths if ground_truths else None,
        'candidate_recall_conf0001_max300': candidate_tp / ground_truths if ground_truths else None,
        'recall_top20_iou050': top20_tp / ground_truths if ground_truths else None,
        'gt_count': ground_truths, 'frames': len(targets),
        'predictions_conf025': fixed_count, 'true_positives_conf025': fixed_tp,
        'matching': 'confidence-ordered single-class one-to-one matching; ultralytics 101-point AP',
    }
