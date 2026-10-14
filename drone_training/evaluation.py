"""Original-coordinate COCO evaluation and validation-only threshold selection."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


PROVENANCE_SCHEMA = 'urap.predictions.provenance.v1'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _sample_digest(rows):
    identity = [[row.get('sample_id'), row.get('source_sha256'), row.get('source_frame_id')]
                for row in rows]
    return hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()


def _provenance_path(predictions_path):
    return Path(predictions_path).with_suffix('.meta.json')


def load_prediction_provenance(predictions_path):
    """Load and verify a prediction file's immutable provenance sidecar."""
    path = Path(predictions_path)
    metadata_path = _provenance_path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f'prediction provenance is missing: {metadata_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('schema') != PROVENANCE_SCHEMA:
        raise ValueError('unsupported prediction provenance schema')
    if metadata.get('predictions_sha256') != _sha256(path):
        raise ValueError('prediction SHA256 does not match provenance')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if int(metadata.get('sample_count', -1)) != len(rows):
        raise ValueError('prediction sample count does not match provenance')
    if metadata.get('sample_digest') != _sample_digest(rows):
        raise ValueError('prediction sample digest does not match provenance')
    splits = {row.get('split') for row in rows}
    if len(splits) != 1 or metadata.get('split') != next(iter(splits)):
        raise ValueError('prediction split does not match provenance')
    return metadata


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def inverse_letterbox(predictions, ratio, pad, original_size):
    result = np.asarray(predictions, dtype=float).reshape(-1, 6).copy()
    result[:, [0, 2]] = (result[:, [0, 2]] - float(pad[0])) / float(ratio)
    result[:, [1, 3]] = (result[:, [1, 3]] - float(pad[1])) / float(ratio)
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, original_size[1])
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, original_size[0])
    # NMS can retain boxes entirely inside the padded border. Clipping those to
    # the original image makes empty rectangles, which are not detections.
    if not np.isfinite(result).all(): raise ValueError('non-finite model prediction')
    return result[(result[:,2] > result[:,0]) & (result[:,3] > result[:,1])]


def _matches(predictions, boxes, threshold=.5):
    predictions = np.asarray(predictions, float).reshape(-1,6)
    boxes = np.asarray(boxes, float).reshape(-1,4)
    found = np.zeros(len(boxes), bool)
    tp = np.zeros(len(predictions), bool)
    for index in np.argsort(-predictions[:,4], kind='stable'):
        if not len(boxes): break
        box = predictions[index,:4]
        intersection = np.maximum(0, np.minimum(box[2:], boxes[:,2:]) - np.maximum(box[:2], boxes[:,:2])).prod(1)
        union = np.maximum(0, box[2:]-box[:2]).prod() + np.maximum(0, boxes[:,2:]-boxes[:,:2]).prod(1) - intersection
        ious = intersection/np.maximum(union, 1e-12)
        ious[found] = -1
        best = int(ious.argmax())
        if ious[best] >= threshold:
            found[best] = True; tp[index] = True
    return tp, found


def _operating(rows, threshold):
    tp = count = ground_truths = 0
    size_found = defaultdict(int); size_total = defaultdict(int)
    for row in rows:
        boxes = np.asarray(row['gt_boxes'], float).reshape(-1,4)
        pred = np.asarray(row['predictions'], float).reshape(-1,6)
        pred = pred[pred[:,4] >= threshold]
        matched, found = _matches(pred, boxes)
        tp += int(matched.sum()); count += len(pred); ground_truths += len(boxes)
        for box, hit in zip(boxes, found):
            longest = max(box[2]-box[0], box[3]-box[1])
            name = 'native_max_side_le16' if longest <= 16 else 'native_max_side_16_32' if longest <= 32 else 'native_max_side_gt32'
            size_total[name] += 1; size_found[name] += int(hit)
    precision = tp/count if count else 0.0
    recall = tp/ground_truths if ground_truths else None
    f1 = 2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0.0
    return dict(precision=precision, recall=recall, f1=f1, fp_per_frame=(count-tp)/len(rows) if rows else 0.0,
                true_positives=tp, false_positives=count-tp, ground_truths=ground_truths, frames=len(rows),
                threshold=float(threshold), size_recall={k:dict(gt=size_total[k],recalled=size_found[k],recall=size_found[k]/size_total[k]) for k in size_total})


def metrics_for_predictions(rows, threshold=.25, max_det=100):
    """Use official pycocotools bbox AP with explicit area/maxDet conventions."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    rows = list(rows)
    if not rows: raise ValueError('evaluation requires at least one frame')
    if max_det < 10: raise ValueError('COCO max_det must be at least 10')
    images, annotations, detections = [], [], []
    seen = set()
    for image_id, row in enumerate(rows, 1):
        if row['sample_id'] in seen: raise ValueError('duplicate evaluation sample_id')
        seen.add(row['sample_id'])
        images.append(dict(id=image_id,width=row['width'],height=row['height']))
        for box in row['gt_boxes']:
            x1,y1,x2,y2 = map(float,box)
            annotations.append(dict(id=len(annotations)+1,image_id=image_id,category_id=1,
                bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1),iscrowd=0))
        for pred in row['predictions']:
            x1,y1,x2,y2,score,category = map(float,pred)
            if category != 0: raise ValueError('only drone class 0 is supported')
            if not np.isfinite(pred).all() or x2<=x1 or y2<=y1: raise ValueError('invalid prediction')
            detections.append(dict(image_id=image_id,category_id=1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO(); gt.dataset=dict(images=images, annotations=annotations,categories=[dict(id=1,name='drone')],info={}); gt.createIndex()
        if detections: dt=gt.loadRes(detections)
        else:
            dt=COCO(); dt.dataset=dict(images=images,annotations=[],categories=gt.dataset['categories'],info={}); dt.createIndex()
        evaluator=COCOeval(gt,dt,'bbox'); evaluator.params.maxDets=[1,10,int(max_det)]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    stats = [None if value < 0 else float(value) for value in evaluator.stats]
    result = _operating(rows, threshold)
    result.update(map50_95=stats[0],map50=stats[1],map75=stats[2],coco_ap_small=stats[3],
                  coco_ap_medium=stats[4],coco_ap_large=stats[5],coco_ar_maxdet=stats[8],
                  max_det=max_det, ap_implementation='official pycocotools COCOeval bbox',
                  incomplete_annotation_frames=sum(r.get('boxes_complete_for_frame') is not True and r.get('role')!='negative_certified' for r in rows),
                  annotation_caveat='Metrics describe supplied boxes; unknown annotation completeness can bias AP/precision/recall. Native side-length bins are not COCO area bins.')
    return result


def evaluate(model, dataset, output_dir, batch_size=2, device='cpu', confidence_floor=.001,
             nms_iou=.7, max_det=100, threshold=.25, workers=0, provenance=None):
    import torch
    from torch.utils.data import DataLoader
    from ultralytics.utils.nms import non_max_suppression
    from .data import collate_samples
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,collate_fn=collate_samples)
    was_training=model.training; model.eval(); rows=[]; timings=[]
    started=time.monotonic()
    try:
        with torch.inference_mode():
            for batch in loader:
                if str(device).startswith('cuda'): torch.cuda.synchronize()
                tick=time.perf_counter()
                image=batch['img'].to(device)
                raw=model(image)
                raw=raw[0] if isinstance(raw,(tuple,list)) else raw
                predictions=non_max_suppression(raw,conf_thres=confidence_floor,iou_thres=nms_iou,max_det=max_det,nc=1)
                if str(device).startswith('cuda'): torch.cuda.synchronize()
                timings.append((time.perf_counter()-tick)/len(predictions))
                for pred,meta in zip(predictions,batch['metadata']):
                    restored=inverse_letterbox(pred.detach().cpu().numpy(),meta['ratio'],meta['pad'],meta['original_size'])
                    height,width=meta['original_size']
                    row={key:meta.get(key) for key in ('sample_id','source_sha256','source_frame_id','source_domain','sequence_id','split','role','boxes_complete_for_frame','motion_fallback','motion_reason')}
                    row.update(width=int(width),height=int(height),gt_boxes=meta['bbox_xyxy'],predictions=restored.tolist(),
                               source_video_path=f"videos/{meta['source_sha256']}.mp4")
                    rows.append(row)
    finally:
        model.train(was_training)
    if not rows: raise ValueError('evaluation dataset is empty')
    predictions_path=output/'predictions.jsonl'
    temporary=predictions_path.with_suffix('.jsonl.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row,allow_nan=False)+'\n')
    temporary.replace(predictions_path)
    splits = {row.get('split') for row in rows}
    if len(splits) != 1: raise ValueError('evaluation rows must have exactly one split')
    metadata = {**dict(provenance or {}), 'schema': PROVENANCE_SCHEMA,
                'predictions_sha256': _sha256(predictions_path),
                'sample_digest': _sample_digest(rows), 'sample_count': len(rows),
                'split': next(iter(splits))}
    _write(_provenance_path(predictions_path), metadata)
    result=metrics_for_predictions(rows,threshold,max_det)
    result.update(confidence_floor=confidence_floor,nms_iou=nms_iou,wall_seconds=time.monotonic()-started,
                  inference_nms_seconds_per_frame_p50=float(np.percentile(timings,50)),
                  inference_nms_seconds_per_frame_p95=float(np.percentile(timings,95)),
                  latency_scope='GPU-synchronized transfer+inference+NMS; wall_seconds additionally includes decode, motion, data and metric work',
                  motion_fallback_frames=sum(bool(r.get('motion_fallback')) for r in rows))
    grouped=defaultdict(list); videos=defaultdict(list)
    for row in rows:
        grouped[row.get('source_domain') or 'unknown'].append(row)
        videos[row['source_sha256']].append(row)
    result['per_domain']={key:metrics_for_predictions(group,threshold,max_det) for key,group in grouped.items()}
    result['per_video']={key:metrics_for_predictions(group,threshold,max_det) for key,group in videos.items()}
    _write(output/'metrics.json',result)
    return result


def calibrate(predictions_path, output, fp_per_frame=None):
    path=Path(predictions_path)
    provenance = load_prediction_provenance(path)
    required = ('arm','checkpoint_sha256','manifest_sha256','data_revision','code_sha256',
                'input_size','confidence_floor','nms_iou','max_det')
    missing = [name for name in required if not provenance.get(name)]
    if missing: raise ValueError(f'prediction provenance lacks formal fields: {missing}')
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(r.get('split') not in ('val','validation') for r in rows):
        raise ValueError('calibration accepts validation predictions only')
    thresholds=[.001,.005,.01,.02,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9,.95,.99,1.0]
    scores=[_operating(rows,threshold) for threshold in thresholds]
    eligible=scores if fp_per_frame is None else [r for r in scores if r['fp_per_frame']<=fp_per_frame]
    if not eligible: raise ValueError('no threshold satisfies the validation FP/frame limit')
    selected=max(eligible,key=lambda r:(r['f1'],r['recall'] or 0,r['threshold']))
    result=dict(threshold=selected['threshold'],selection_rule='maximum validation F1, ties recall then higher threshold',
                fp_per_frame_limit=fp_per_frame,validation_metrics=selected,candidates=scores,
                predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),split='validation',
                provenance=provenance)
    _write(output,result)
    return result


def compare_predictions(a_rows, b_rows, threshold_a=.25, threshold_b=.25, resamples=1000, seed=17):
    """Paired source-video bootstrap for operating metrics, with official AP point estimates.

    This resamples whole videos (never neighboring frames independently). Intervals
    condition on these trained checkpoints and do not estimate training-seed variance.
    """
    a_rows, b_rows = list(a_rows), list(b_rows)
    a = {row['sample_id']: row for row in a_rows}
    b = {row['sample_id']: row for row in b_rows}
    if not a or a.keys() != b.keys() or len(a) != len(a_rows) or len(b) != len(b_rows):
        raise ValueError('paired comparison requires identical frames and unique sample IDs')
    groups = defaultdict(list)
    for key in sorted(a):
        fields = ('source_sha256', 'source_frame_id', 'width', 'height', 'split', 'gt_boxes',
                  'boxes_complete_for_frame', 'role')
        if any(a[key].get(field) != b[key].get(field) for field in fields):
            raise ValueError(f'paired comparison ground truth or frame identity differs: {key}')
        if not a[key].get('source_sha256'): raise ValueError('source video SHA256 is required')
        groups[a[key]['source_sha256']].append(key)
    if resamples < 1: raise ValueError('resamples must be positive')
    point_a = metrics_for_predictions(a_rows, threshold_a)
    point_b = metrics_for_predictions(b_rows, threshold_b)
    names = ('recall', 'precision', 'f1', 'fp_per_frame')
    # Sufficient counts make resampling inexpensive even with hundreds of thousands of frames.
    def counts(rows, threshold):
        value = _operating(rows, threshold)
        return [value[k] for k in ('true_positives', 'false_positives', 'ground_truths', 'frames')]
    videos = sorted(groups)
    counts_a = np.asarray([counts([a[k] for k in groups[v]], threshold_a) for v in videos], float)
    counts_b = np.asarray([counts([b[k] for k in groups[v]], threshold_b) for v in videos], float)
    def summarize(value):
        tp, fp, gt, n = value
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/gt if gt else np.nan
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        return np.array([recall, precision, f1, fp/n])
    rng = np.random.default_rng(seed); differences = []
    for _ in range(resamples):
        indexes = rng.integers(0, len(videos), len(videos))
        differences.append(summarize(counts_b[indexes].sum(0))-summarize(counts_a[indexes].sum(0)))
    differences = np.asarray(differences)
    intervals = {}
    for index, name in enumerate(names):
        values = differences[:, index]; values = values[np.isfinite(values)]
        intervals[name] = dict(lower=float(np.percentile(values, 2.5)) if len(values) else None,
                               upper=float(np.percentile(values, 97.5)) if len(values) else None,
                               valid_resamples=len(values))
    delta = {name: (point_b[name]-point_a[name] if point_a[name] is not None and point_b[name] is not None else None)
             for name in (*names, 'map50', 'map50_95')}
    return dict(A=point_a, B=point_b, delta_B_minus_A=delta,
                paired_video_bootstrap=dict(unit='source_video_sha256', videos=len(videos), resamples=resamples,
                  seed=seed, confidence=.95, intervals=intervals,
                  caveat='Operating-metric intervals condition on these checkpoints. AP is a point estimate; training-seed variance requires independent matched runs.'))
