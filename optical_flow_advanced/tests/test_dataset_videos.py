import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module('video_comparison'), importlib.import_module('dataset_videos')


def test_letterbox_preserves_aspect_and_excludes_padding(modules):
    image = np.full((2048, 2448, 3), 127, np.uint8)
    canvas, valid = modules[0].letterbox(image, 1920, 1080)
    assert canvas.shape == (1080, 1920, 3)
    assert not canvas[:, :300].any()
    assert not valid[:, :300].any()
    assert valid[540, 960] == 255
    assert not valid[:8].any()
    assert np.unique(canvas).tolist() == [0, 127]


def test_author_uint8_wraparound_is_preserved(modules):
    center = np.zeros((64, 64, 3), np.uint8)
    neighbor = np.full_like(center, 200)

    def compensate(source, reference):
        return source, np.zeros_like(reference)

    difference, valid = modules[0].legacy_difference(neighbor, center, neighbor, compensate)
    assert np.all(difference == 72)
    assert np.all(valid == 255)


def test_statistics_use_only_common_valid_pixels(modules):
    image = np.array([[0, 9], [33, 255]], np.uint8)
    valid = np.array([[255, 255], [255, 0]], np.uint8)
    stats = modules[0].intensity_statistics(image, valid)
    assert stats['pixels'] == 3
    assert stats['sum'] == 42
    assert stats['active_gt8'] == 2
    assert stats['active_gt32'] == 1
    assert modules[0].intensity_statistics(image, np.zeros_like(valid)) is None
    samples = [{'parallax_raw': stats}, {'parallax_raw': None}]
    summary = modules[0].summarize_samples(samples)
    assert summary['parallax_raw']['sampled_frames'] == 1
    assert summary['parallax_raw']['mean_intensity'] == 14


def test_atomic_json_retries_transient_reader_lock(modules, tmp_path, monkeypatch):
    original_replace = Path.replace
    attempts = []

    def locked_replace(source, destination):
        attempts.append(str(source))
        if len(attempts) < 3:
            raise PermissionError('Simulated Windows reader without delete sharing')
        return original_replace(source, destination)

    monkeypatch.setattr(Path, 'replace', locked_replace)
    monkeypatch.setattr(modules[0].time, 'sleep', lambda seconds: None)
    path = tmp_path / 'progress.json'
    modules[0].write_json(path, {'done': 10})
    assert len(attempts) == 3
    assert json.loads(path.read_text()) == {'done': 10}


def test_permanent_write_failure_is_not_hidden(modules, tmp_path, monkeypatch):
    def locked_replace(source, destination):
        raise PermissionError('Persistent access failure')

    monkeypatch.setattr(Path, 'replace', locked_replace)
    monkeypatch.setattr(modules[0].time, 'sleep', lambda seconds: None)
    with pytest.raises(PermissionError, match='Persistent'):
        modules[0].replace_with_retry(tmp_path / 'temporary', tmp_path / 'destination', attempts=3)


def test_json_reader_retries_during_replacement(modules, tmp_path, monkeypatch):
    attempts = []

    def locked_read(path, **kwargs):
        attempts.append(path)
        if len(attempts) < 3:
            raise PermissionError('Simulated concurrent Windows replacement')
        return '{"done": 7}'

    monkeypatch.setattr(Path, 'read_text', locked_read)
    monkeypatch.setattr(modules[0].time, 'sleep', lambda seconds: None)
    assert modules[0].read_json(tmp_path / 'progress.json') == {'done': 7}
    assert len(attempts) == 3


def test_indexed_frames_keep_numeric_order_and_expose_gaps(modules, tmp_path):
    for name in ('Clip_1_00010.png', 'Clip_1_00002.png', 'Clip_2_00001.jpg'):
        (tmp_path / name).touch()
    grouped = modules[1].indexed_frames(tmp_path)
    assert [index for index, path in grouped[1]] == [2, 10]
    (tmp_path / 'Clip_1_00002.jpg').touch()
    with pytest.raises(ValueError, match='Duplicate'):
        modules[1].indexed_frames(tmp_path)


def test_round_robin_interleaves_datasets_shortest_first(modules):
    sequences = [{'dataset': dataset, 'name': str(count), 'cached_frame_count': count}
                 for dataset, count in [('ARD', 8), ('NPS', 4), ('NPS', 2), ('AOT', 5)]]
    result = modules[1].round_robin(sequences)
    assert [(item['dataset'], item['cached_frame_count']) for item in result] == [('NPS', 2), ('AOT', 5), ('ARD', 8), ('NPS', 4)]


def test_completed_resume_writes_terminal_state(modules, tmp_path, monkeypatch):
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps({'run_dir': str(tmp_path), 'output_root': str(tmp_path), 'sequences': [], 'workers': 2}))
    monkeypatch.setattr(modules[1], 'render_catalog', lambda manifest, reports: None)
    monkeypatch.setattr(modules[1].shutil, 'disk_usage', lambda path: SimpleNamespace(free=200 * 1024 ** 3))
    assert modules[1].run_manifest(manifest_path) == 0
    state = json.loads((tmp_path / 'progress.json').read_text())
    assert state['phase'] == 'completed'
    assert state['done'] == state['total'] == 0
    with pytest.raises(ValueError, match='resume'):
        modules[1].run_manifest(manifest_path)
    assert modules[1].run_manifest(manifest_path, resume=True) == 0


def test_temporal_window_does_not_drop_or_cross_frames(modules, tmp_path, monkeypatch):
    worker = modules[0]
    frames = []
    for number in range(5):
        path = tmp_path / f'{number}.png'
        cv2.imwrite(str(path), np.full((18, 32, 3), (number + 1) * 10, np.uint8))
        frames.append(str(path))
    calls, author_calls = [], []

    def compensated(previous, center, following, parameters, local):
        calls.append((local, int(previous[540, 960]), int(center[540, 960]), int(following[540, 960])))
        return np.zeros_like(center, dtype=np.float32), np.full_like(center, 255), {}

    def legacy(previous, center, following, method):
        author_calls.append(tuple(int(image[540, 960, 0]) for image in (previous, center, following)))
        return np.zeros(center.shape[:2], np.uint8), np.full(center.shape[:2], 255, np.uint8)

    class FakeWriter:
        def __init__(self, *args):
            self.process = SimpleNamespace(pid=0)

        def write(self, image):
            assert image.shape[:2] == (1080, 1920)

        def close(self):
            pass

        def finalize(self, ffprobe, expected_frames):
            assert expected_frames == 5
            return {'path': 'test-only', 'bytes': 0}

        def abort(self):
            pass

    method = SimpleNamespace(compensated_difference=compensated,
                             adaptive_residual=lambda difference, *args: difference.astype(np.uint8))
    monkeypatch.setattr(worker, 'load_methods', lambda manifest: (method, None))
    monkeypatch.setattr(worker, 'legacy_difference', legacy)
    monkeypatch.setattr(worker, 'VideoWriter', FakeWriter)
    manifest = {'source_hashes': {}, 'output_root': str(tmp_path / 'output'), 'run_dir': str(tmp_path),
                'opencv_threads': 1, 'parameters': {'adaptive_strength': 1.2, 'residual_floor': 2, 'residual_gain': 1.2},
                'ffmpeg': '', 'ffprobe': '', 'display_gain': 3, 'metric_stride': 30,
                'sequences': [{'id': 'NPS/test/sample', 'dataset': 'NPS', 'split': 'test', 'name': 'sample',
                               'kind': 'frames', 'frames': frames, 'fps': 30, 'original_reference': 'test'}]}
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    report = worker.process_sequence(manifest_path, 'NPS/test/sample')
    assert report['frames'] == 5
    assert [values[1:] for values in calls if values[0]] == [(10, 10, 20), (10, 20, 30), (20, 30, 40), (30, 40, 50), (40, 50, 50)]
    assert author_calls == [(10, 10, 30), (10, 20, 40), (10, 30, 50), (20, 40, 50), (30, 50, 50)]


def test_video_declared_frame_overcount_uses_decodable_length(modules, tmp_path, monkeypatch):
    worker = modules[0]
    source = tmp_path / 'overcounted.mp4'
    source.touch()
    frames = [np.full((18, 32, 3), (number + 1) * 10, np.uint8) for number in range(5)]

    class FakeCapture:
        def __init__(self, path):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, key):
            return 7 if key == cv2.CAP_PROP_FRAME_COUNT else 30

        def read(self):
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            pass

    finalized = []

    class FakeWriter:
        def __init__(self, *args):
            self.process = SimpleNamespace(pid=0)

        def write(self, image):
            assert image.shape[:2] == (1080, 1920)

        def close(self):
            pass

        def finalize(self, ffprobe, expected_frames):
            finalized.append(expected_frames)
            return {'path': 'test-only', 'bytes': 0}

        def abort(self):
            pass

    method = SimpleNamespace(
        compensated_difference=lambda previous, center, following, parameters, local:
            (np.zeros(center.shape, np.float32), np.full(center.shape, 255, np.uint8), {}),
        adaptive_residual=lambda difference, *args: difference.astype(np.uint8),
    )
    monkeypatch.setattr(worker.cv2, 'VideoCapture', FakeCapture)
    monkeypatch.setattr(worker, 'load_methods', lambda manifest: (method, None))
    monkeypatch.setattr(worker, 'legacy_difference', lambda previous, center, following, compensate:
                        (np.zeros(center.shape[:2], np.uint8), np.full(center.shape[:2], 255, np.uint8)))
    monkeypatch.setattr(worker, 'VideoWriter', FakeWriter)
    manifest = {
        'source_hashes': {}, 'output_root': str(tmp_path / 'output'), 'run_dir': str(tmp_path),
        'opencv_threads': 1,
        'parameters': {'adaptive_strength': 1.2, 'residual_floor': 2, 'residual_gain': 1.2},
        'ffmpeg': '', 'ffprobe': '', 'display_gain': 3, 'metric_stride': 30,
        'sequences': [{'id': 'ARD/test/sample', 'dataset': 'ARD', 'split': 'test', 'name': 'sample',
                       'kind': 'video', 'source': str(source), 'stage_source': False,
                       'original_reference': 'test'}],
    }
    manifest_path = tmp_path / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest))

    report = worker.process_sequence(manifest_path, 'ARD/test/sample')

    assert report['frames'] == report['source_frames'] == 5
    assert report['declared_source_frames'] == 7
    assert report['decoded_frame_count_adjusted'] is True
    assert finalized == [5, 5, 5]
