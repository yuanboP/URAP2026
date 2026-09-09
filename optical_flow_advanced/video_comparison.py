from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def timestamp():
    return datetime.now().astimezone().isoformat()


def replace_with_retry(source, destination, attempts=20):
    for attempt in range(attempts):
        try:
            Path(source).replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.01 * (attempt + 1))


def read_json(path, attempts=20):
    for attempt in range(attempts):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8'))
        except (PermissionError, FileNotFoundError, json.JSONDecodeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(0.01 * (attempt + 1))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    replace_with_retry(temporary, path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def load_methods(manifest):
    path = manifest['method_snapshot']
    spec = importlib.util.spec_from_file_location('pinned_parallax', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = Path(manifest['author_snapshot']).read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'motion_compensate')
    namespace = {'cv2': cv2, 'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), manifest['author_snapshot'], 'exec'), namespace)
    return module, namespace['motion_compensate']


def letterbox(image, width, height):
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    left, top = (width - resized_width) // 2, (height - resized_height) // 2
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    valid = np.zeros((height, width), np.uint8)
    if resized_width > 16 and resized_height > 16:
        valid[top + 8:top + resized_height - 8, left + 8:left + resized_width - 8] = 255
    return canvas, valid


def legacy_difference(previous, center, following, compensate):
    blurred = [cv2.cvtColor(cv2.GaussianBlur(image, (11, 11), 0), cv2.COLOR_BGR2GRAY)
               for image in (previous, center, following)]
    before = compensate(blurred[0], blurred[1])
    after = compensate(blurred[2], blurred[1])
    difference = (cv2.absdiff(blurred[1], before[0]) + cv2.absdiff(blurred[1], after[0])) / 2
    success, encoded = cv2.imencode('.jpg', difference)
    if not success:
        raise RuntimeError('Author FD5 JPEG conversion failed')
    valid = cv2.bitwise_and(cv2.bitwise_not(before[1]), cv2.bitwise_not(after[1]))
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE), valid


def intensity_statistics(image, valid):
    pixels = image[valid > 0]
    if not pixels.size:
        return None
    histogram = np.bincount(pixels, minlength=256)
    percentile = int(np.searchsorted(np.cumsum(histogram), pixels.size * 0.95))
    return {'pixels': int(pixels.size), 'sum': int(pixels.sum()),
            'mean': float(pixels.mean()), 'p95': percentile,
            'active_gt8': int((pixels > 8).sum()), 'active_gt32': int((pixels > 32).sum())}


def summarize_samples(samples):
    output = {}
    for method in ('author_fd5', 'global_only', 'parallax_raw', 'parallax'):
        values = [sample[method] for sample in samples if sample.get(method)]
        pixels = sum(value['pixels'] for value in values)
        output[method] = {'sampled_frames': len(values), 'valid_pixels': pixels,
                          'mean_intensity': sum(value['sum'] for value in values) / pixels if pixels else None,
                          'active_fraction_gt8': sum(value['active_gt8'] for value in values) / pixels if pixels else None,
                          'active_fraction_gt32': sum(value['active_gt32'] for value in values) / pixels if pixels else None}
    return output


def comparison_frame(rgb, original, global_only, new, identifier, frame_index, gain, original_failed=False):
    panels = [rgb]
    for image in (original, global_only, new):
        panels.append(cv2.cvtColor(np.uint8(np.clip(image.astype(np.float32) * gain, 0, 255)), cv2.COLOR_GRAY2BGR))
    titles = ['RGB reference', 'Author FD5 reference | t +/- 2',
              'Global-only control | t +/- 1', 'New parallax + suppression | t +/- 1']
    if original_failed:
        titles[1] = 'AUTHOR FD5 FAILED | excluded from metrics'
    rendered = []
    for panel, title in zip(panels, titles):
        panel = cv2.resize(panel, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (960, 35), (10, 14, 22), -1)
        cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        rendered.append(panel)
    canvas = np.vstack((np.hstack(rendered[:2]), np.hstack(rendered[2:])))
    cv2.rectangle(canvas, (0, 1048), (1920, 1080), (10, 14, 22), -1)
    label = f'{identifier} | source frame {frame_index} | motion display: shared linear x{gain:g} | no per-frame normalization'
    cv2.putText(canvas, label, (12, 1070), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


class VideoWriter:
    def __init__(self, ffmpeg, destination, fps, gray, log_directory):
        self.destination = Path(destination)
        self.temporary = self.destination.with_name(self.destination.stem + '.partial.mp4')
        self.log_path = Path(log_directory) / (self.destination.stem + '.encoder.log')
        self.log = self.log_path.open('wb')
        command = [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-y', '-threads', '1',
                   '-filter_threads', '1', '-f', 'rawvideo', '-pixel_format', 'gray' if gray else 'bgr24',
                   '-video_size', '1920x1080', '-framerate', str(fps), '-i', 'pipe:0', '-an',
                   '-c:v', 'h264_nvenc', '-preset', 'p4', '-rc', 'vbr', '-cq', '19', '-b:v', '0',
                   '-maxrate', '20M', '-bufsize', '40M', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                   str(self.temporary)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                        stderr=self.log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))

    def write(self, frame):
        self.process.stdin.write(frame.tobytes())

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        result = self.process.wait(timeout=180)
        self.log.close()
        if result:
            raise RuntimeError(f'Encoder failed ({result}): {self.log_path}')

    def abort(self):
        try:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            self.process.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()
        finally:
            self.log.close()

    def finalize(self, ffprobe, expected_frames):
        command = [ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                   'stream=width,height,nb_frames,avg_frame_rate,duration', '-of', 'json', str(self.temporary)]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
        stream = json.loads(result.stdout)['streams'][0]
        if int(stream.get('nb_frames', -1)) != expected_frames or (stream['width'], stream['height']) != (1920, 1080):
            raise RuntimeError(f'Output verification failed: {self.temporary}: {stream}')
        replace_with_retry(self.temporary, self.destination)
        return {'path': str(self.destination), 'bytes': self.destination.stat().st_size, 'stream': stream}


def stage_source(sequence, output_root, progress_path, stop_path):
    source = Path(sequence['source'])
    if not sequence.get('stage_source'):
        return source
    destination = Path(output_root) / 'source_cache' / sequence['dataset'] / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = source.stat().st_size
    if destination.exists() and destination.stat().st_size == expected:
        return destination
    temporary = destination.with_suffix(destination.suffix + '.partial')
    copied = 0
    last_update = 0.0
    with source.open('rb') as reader, temporary.open('wb') as writer:
        for block in iter(lambda: reader.read(8 * 1024 * 1024), b''):
            if Path(stop_path).exists():
                raise InterruptedError('Stop requested during source staging')
            writer.write(block)
            copied += len(block)
            if time.monotonic() - last_update > 2:
                write_json(progress_path, {'phase': 'staging', 'pid': os.getpid(), 'sequence': sequence['id'],
                                          'done': copied, 'total': expected, 'unit': 'bytes', 'updated_at': timestamp()})
                last_update = time.monotonic()
    if copied != expected:
        raise RuntimeError(f'Source staging size mismatch: {source}')
    replace_with_retry(temporary, destination)
    return destination


def process_sequence(manifest_path, sequence_id, max_frames=0, preview_root=None):
    manifest = read_json(manifest_path)
    for path, checksum in manifest['source_hashes'].items():
        if digest(path) != checksum:
            raise RuntimeError(f'Pinned source changed: {path}')
    sequence = next(item for item in manifest['sequences'] if item['id'] == sequence_id)
    output_root = Path(preview_root or manifest['output_root'])
    output = output_root / sequence['dataset'] / sequence['split'] / sequence['name']
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'report.json'
    if report_path.exists() and read_json(report_path).get('status') == 'completed':
        raise RuntimeError(f'Refusing to overwrite completed sequence: {output}')
    progress_path = output / 'progress.json'
    stop_path = Path(manifest['run_dir']) / 'STOP_REQUESTED'
    cv2.setNumThreads(manifest['opencv_threads'])
    cv2.setRNGSeed(20260903)
    module, compensate = load_methods(manifest)
    parameters = SimpleNamespace(**manifest['parameters'])
    started = timestamp()
    started_clock = time.monotonic()
    capture = None
    writers = []
    samples = []
    failures = Counter()
    done = 0
    try:
        if sequence['kind'] == 'video':
            source = stage_source(sequence, output_root, progress_path, stop_path)
            capture = cv2.VideoCapture(str(source))
            if not capture.isOpened():
                raise RuntimeError(f'Cannot open source video: {source}')
            source_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        else:
            source_total = len(sequence['frames'])
            fps = sequence['fps']
        if source_total < 1 or fps <= 0:
            raise ValueError('Invalid source frame count or FPS')
        declared_source_total = source_total
        total = min(source_total, max_frames) if max_frames else source_total
        writer_specs = [('new_motion.mp4', True), ('original_fd5.mp4', True), ('comparison.mp4', False)]
        for name, gray in writer_specs:
            writers.append(VideoWriter(manifest['ffmpeg'], output / name, fps, gray, output))
        cache = {}
        next_read = 0
        last_update = 0.0
        source_shape = None
        for frame_index in range(total):
            if stop_path.exists():
                raise InterruptedError('Stop requested')
            while next_read <= min(source_total - 1, frame_index + 2):
                if capture is not None:
                    success, frame = capture.read()
                    if not success:
                        if next_read == 0:
                            raise RuntimeError(f'Video declares {declared_source_total} frames but none can be decoded')
                        previous_total = source_total
                        source_total = next_read
                        total = min(source_total, max_frames) if max_frames else source_total
                        print(f'{timestamp()} DECODED_FRAME_COUNT_ADJUSTED declared={previous_total} actual={source_total}',
                              flush=True)
                        break
                else:
                    frame = cv2.imread(sequence['frames'][next_read])
                    if frame is None:
                        raise RuntimeError(f'Unreadable image: {sequence["frames"][next_read]}')
                if source_shape is None:
                    source_shape = list(frame.shape[:2])
                elif list(frame.shape[:2]) != source_shape:
                    raise RuntimeError('Source image dimensions changed within sequence')
                cache[next_read] = letterbox(frame, 1920, 1080)
                next_read += 1
            if frame_index >= total:
                break
            get_frame = lambda offset: cache[max(0, min(source_total - 1, frame_index + offset))][0]
            rgb, content_valid = cache[frame_index]
            gray_frames = [cv2.cvtColor(get_frame(offset), cv2.COLOR_BGR2GRAY) for offset in (-1, 0, 1)]
            cv2.setRNGSeed(20260903 + frame_index)
            global_raw, global_valid, _ = module.compensated_difference(*gray_frames, parameters, local=False)
            cv2.setRNGSeed(20260903 + frame_index)
            new_raw, new_valid, stats = module.compensated_difference(*gray_frames, parameters, local=True)
            global_map = np.uint8(np.clip(global_raw, 0, 255))
            raw_map = np.uint8(np.clip(new_raw, 0, 255))
            new_map = module.adaptive_residual(new_raw, new_valid, parameters.adaptive_strength,
                                               parameters.residual_floor, parameters.residual_gain)
            author_failed = False
            try:
                original, original_valid = legacy_difference(get_frame(-2), rgb, get_frame(2), compensate)
            except (cv2.error, ValueError, np.linalg.LinAlgError) as error:
                failures[type(error).__name__] += 1
                author_failed = True
                original = np.zeros_like(new_map)
                original_valid = np.zeros_like(new_valid)
                if sum(failures.values()) <= 5:
                    print(f'{timestamp()} AUTHOR_REFERENCE_FAILURE frame={frame_index}: {error}', flush=True)
            view = comparison_frame(rgb, original, global_map, new_map, sequence_id, frame_index,
                                    manifest['display_gain'], author_failed)
            for writer, image in zip(writers, (new_map, original, view)):
                writer.write(image)
            done = frame_index + 1
            if frame_index == min(30, total - 1):
                cv2.imwrite(str(output / 'preview.jpg'), view)
            if frame_index % manifest['metric_stride'] == 0 and 2 <= frame_index < source_total - 2:
                common_valid = cv2.bitwise_and(content_valid, cv2.bitwise_and(global_valid, new_valid))
                if not author_failed:
                    common_valid = cv2.bitwise_and(common_valid, original_valid)
                if author_failed:
                    common_valid[:] = 0
                sample = {'frame': frame_index, 'author_fd5': intensity_statistics(original, common_valid),
                          'global_only': intensity_statistics(global_map, common_valid),
                          'parallax_raw': intensity_statistics(raw_map, common_valid),
                          'parallax': intensity_statistics(new_map, common_valid), 'alignment': stats}
                samples.append(sample)
            for old_index in [index for index in cache if index < frame_index - 1]:
                del cache[old_index]
            if time.monotonic() - last_update > 2 or done == total:
                progress = {'phase': 'processing', 'pid': os.getpid(), 'sequence': sequence_id,
                            'done': done, 'total': total, 'source_total': source_total, 'unit': 'frames',
                            'started_at': started, 'updated_at': timestamp(), 'last_source_frame': frame_index,
                            'processing_fps': done / max(time.monotonic() - started_clock, 0.001),
                            'encoder_pids': [writer.process.pid for writer in writers],
                            'sampled_metrics': summarize_samples(samples),
                            'author_reference_failed_frames': sum(failures.values())}
                write_json(progress_path, progress)
                print(f'{timestamp()} {sequence_id} {done}/{total} fps={progress["processing_fps"]:.2f}', flush=True)
                last_update = time.monotonic()
        if capture is not None and total == source_total and not max_frames:
            extra_frame_available, _ = capture.read()
            if extra_frame_available:
                raise RuntimeError('Video contains more decoded frames than its declared frame count; refusing incomplete output')
        write_json(progress_path, dict(progress, phase='finalizing', updated_at=timestamp()))
        for writer in writers:
            writer.close()
        outputs = [writer.finalize(manifest['ffprobe'], total) for writer in writers]
        report = {'status': 'completed', 'sequence': sequence_id, 'dataset': sequence['dataset'],
                  'split': sequence['split'], 'started_at': started, 'completed_at': timestamp(),
                  'frames': total, 'source_frames': source_total, 'declared_source_frames': declared_source_total,
                  'decoded_frame_count_adjusted': source_total != declared_source_total,
                  'preview_only': bool(max_frames),
                  'fps': fps, 'source_shape': source_shape, 'output_shape': [1080, 1920], 'outputs': outputs,
                  'original_reference': sequence['original_reference'], 'author_reference_failed_frames': sum(failures.values()),
                  'metric_scope': 'Image-wide valid-region residual intensity; NOT detection AP or GT-based background/target preservation.',
                  'metrics': summarize_samples(samples), 'samples': samples,
                  'elapsed_seconds': time.monotonic() - started_clock}
        write_json(report_path, report)
        write_json(progress_path, dict(progress, phase='completed', updated_at=timestamp()))
        print(f'{timestamp()} COMPLETED {sequence_id} frames={total}', flush=True)
        return report
    except BaseException as error:
        for writer in writers:
            writer.abort()
        write_json(report_path, {'status': 'stopped' if isinstance(error, InterruptedError) else 'failed',
                                'sequence': sequence_id, 'frames_done': done, 'error': str(error), 'updated_at': timestamp()})
        raise
    finally:
        if capture is not None:
            capture.release()
