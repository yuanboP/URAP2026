import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from drone_training.evaluation import metrics_for_predictions, calibrate, inverse_letterbox, load_prediction_provenance


def write_predictions(path, data, *, arm='A', checkpoint_sha='1'*64, manifest_sha='2'*64):
    path.write_text(''.join(json.dumps(r)+'\n' for r in data))
    sample_digest = hashlib.sha256(json.dumps([
        [r['sample_id'], r['source_sha256'], r['source_frame_id']] for r in data
    ], separators=(',', ':')).encode()).hexdigest()
    metadata = {
        'schema': 'urap.predictions.provenance.v1',
        'predictions_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'sample_digest': sample_digest,
        'sample_count': len(data),
        'split': data[0]['split'],
        'arm': arm,
        'checkpoint_sha256': checkpoint_sha,
        'manifest_sha256': manifest_sha,
        'data_revision': 'pinned-revision',
        'code_sha256': '3'*64,
        'input_size': 1280,
        'confidence_floor': .001,
        'nms_iou': .7,
        'max_det': 100,
    }
    path.with_suffix('.meta.json').write_text(json.dumps(metadata))
    return metadata


def rows():
    return [
        dict(sample_id='p', source_sha256='a'*64, source_frame_id=1, source_domain='test-domain',
             sequence_id='p', width=100, height=80, split='validation', role='positive',
             boxes_complete_for_frame=True, gt_boxes=[[10, 10, 30, 30]],
             predictions=[[10, 10, 30, 30, .9, 0]]),
        dict(sample_id='n', source_sha256='b'*64, source_frame_id=1, source_domain='test-domain',
             sequence_id='n', width=100, height=80, split='validation', role='negative_certified',
             boxes_complete_for_frame=True, gt_boxes=[], predictions=[[50, 50, 60, 60, .8, 0]]),
    ]


def test_official_coco_and_fixed_threshold_include_empty_frames():
    result = metrics_for_predictions(rows(), threshold=.25)
    assert result['map50_95'] == pytest.approx(1.0)
    assert result['recall'] == 1
    assert result['precision'] == .5
    assert result['fp_per_frame'] == .5
    assert result['frames'] == 2
    assert 'pycocotools' in result['ap_implementation']


def test_empty_predictions_and_duplicate_matching():
    data = rows()
    data[0]['predictions'] += [[10, 10, 30, 30, .85, 0]]
    assert metrics_for_predictions(data)['true_positives'] == 1
    for row in data:
        row['predictions'] = []
    result = metrics_for_predictions(data)
    assert result['recall'] == 0 and result['map50'] == 0


def test_coordinates_restore_non_square_original():
    prediction = np.array([[20, 30, 60, 70, .9, 0]])
    restored = inverse_letterbox(prediction, ratio=2, pad=(0, 10), original_size=(80,100))
    assert np.allclose(restored[0,:4], [10,10,30,30])


def test_predictions_entirely_in_letterbox_padding_are_discarded():
    prediction = np.array([[2, 2, 8, 8, .5, 0], [20,30,60,70,.9,0]])
    restored = inverse_letterbox(prediction, ratio=2, pad=(0,10), original_size=(80,100))
    assert restored.shape == (1,6)
    assert np.allclose(restored[0,:4], [10,10,30,30])


def test_calibration_is_validation_only(tmp_path):
    path = tmp_path/'predictions.jsonl'
    metadata = write_predictions(path, rows())
    result = calibrate(path, tmp_path/'calibration.json')
    assert result['threshold'] > .8
    assert result['provenance'] == metadata
    data = rows(); data[0]['split'] = 'test'
    write_predictions(path, data)
    with pytest.raises(ValueError, match='split|validation'):
        calibrate(path, tmp_path/'bad.json')


def test_calibration_rejects_tampered_or_missing_prediction_provenance(tmp_path):
    path = tmp_path/'predictions.jsonl'
    write_predictions(path, rows())
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='SHA256'):
        calibrate(path, tmp_path/'bad.json')
    (tmp_path/'predictions.meta.json').unlink()
    with pytest.raises(FileNotFoundError, match='provenance'):
        calibrate(path, tmp_path/'missing.json')


def test_test_threshold_rejects_swapped_arm_or_stale_checkpoint(tmp_path):
    from drone_training.cli import _threshold, _evaluation_code_sha256
    checkpoint = tmp_path/'best_A.pt'; checkpoint.write_bytes(b'checkpoint-A')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    calibration = tmp_path/'calibration.json'
    calibration.write_text(json.dumps({'split':'validation','threshold':.5,'provenance':{
        'arm':'A','checkpoint_sha256':checkpoint_sha,
        'data_revision':'aaf817f51dd670579a8e2c30a5db4930cf5eeb7f',
        'code_sha256':_evaluation_code_sha256(), 'input_size':1280,
        'confidence_floor':.001, 'nms_iou':.7, 'max_det':100}}))
    args = SimpleNamespace(split='test', threshold_file=str(calibration), threshold=.25,
                           arm='A', checkpoint=str(checkpoint), size=1280)
    assert _threshold(args) == .5
    args.arm = 'B'
    with pytest.raises(ValueError, match='arm'):
        _threshold(args)
    args.arm = 'A'; checkpoint.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint'):
        _threshold(args)
    checkpoint.write_bytes(b'checkpoint-A'); args.size = 256
    with pytest.raises(ValueError, match='input_size'):
        _threshold(args)


def test_prediction_provenance_rejects_wrong_sidecar_sample_digest(tmp_path):
    path = tmp_path/'predictions.jsonl'
    write_predictions(path, rows())
    metadata_path = tmp_path/'predictions.meta.json'
    metadata = json.loads(metadata_path.read_text()); metadata['sample_digest'] = '0'*64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='sample digest'):
        load_prediction_provenance(path)


def test_compare_cli_rejects_swapped_calibrations(tmp_path):
    from drone_training.cli import main
    import copy
    a_path = tmp_path/'A_predictions.jsonl'; b_path = tmp_path/'B_predictions.jsonl'
    a_meta = write_predictions(a_path, rows(), arm='A', checkpoint_sha='a'*64)
    b_meta = write_predictions(b_path, copy.deepcopy(rows()), arm='B', checkpoint_sha='b'*64)
    a_cal = tmp_path/'A_calibration.json'; b_cal = tmp_path/'B_calibration.json'
    a_cal.write_text(json.dumps({'threshold':.5,'split':'validation','provenance':a_meta}))
    b_cal.write_text(json.dumps({'threshold':.5,'split':'validation','provenance':b_meta}))
    with pytest.raises(ValueError, match='A arm provenance mismatch'):
        main(['compare','--a-predictions',str(a_path),'--b-predictions',str(b_path),
              '--a-calibration',str(b_cal),'--b-calibration',str(a_cal),
              '--out',str(tmp_path/'comparison.json')])


def test_evaluate_dispatch_resolves_relative_manifest_once(tmp_path, monkeypatch):
    import drone_training.cli as cli
    import drone_training.evaluation as evaluation
    monkeypatch.chdir(tmp_path)
    data_root = tmp_path/'relative-data'; manifest = data_root/'manifests'/'validation.jsonl'
    manifest.parent.mkdir(parents=True); manifest.write_text('{}\n')
    checkpoint = tmp_path/'best_A.pt'; checkpoint.write_bytes(b'checkpoint')
    captured = {}
    monkeypatch.setattr(cli, '_load_detector', lambda _args: object())
    def fake_dataset(root, manifest_path, *_args, **_kwargs):
        captured['root'] = root; captured['manifest'] = manifest_path
        return object()
    monkeypatch.setattr(cli, '_dataset', fake_dataset)
    def fake_evaluate(_model, _dataset, _out, **kwargs):
        captured['provenance'] = kwargs['provenance']; return {'ok': True}
    monkeypatch.setattr(evaluation, 'evaluate', fake_evaluate)
    assert cli.main(['evaluate','--data-root','relative-data','--manifest','manifests/validation.jsonl',
                     '--split','validation','--checkpoint',str(checkpoint),'--arm','A',
                     '--out',str(tmp_path/'eval'),'--device','cpu']) == 0
    assert captured['manifest'] == str(manifest.resolve())
    assert captured['provenance']['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert captured['provenance']['input_size'] == 1280


def test_paired_video_bootstrap_requires_identical_frames_and_truth():
    from drone_training.evaluation import compare_predictions
    import copy
    baseline = rows(); proposed = copy.deepcopy(baseline)
    baseline[0]['predictions'] = []
    report = compare_predictions(baseline, proposed, resamples=50, seed=17)
    assert report['delta_B_minus_A']['recall'] == 1.0
    assert report['paired_video_bootstrap']['unit'] == 'source_video_sha256'
    assert report == compare_predictions(baseline, proposed, resamples=50, seed=17)
    proposed[0]['gt_boxes'] = [[1,1,5,5]]
    with pytest.raises(ValueError, match='ground truth'):
        compare_predictions(baseline, proposed, resamples=5)


def test_comparison_rejects_mismatched_input_frames():
    from drone_training.evaluation import compare_predictions
    with pytest.raises(ValueError, match='identical frames'):
        compare_predictions(rows(), rows()[:1])
