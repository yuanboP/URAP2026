import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a parallax-robust YOLOMG compensated-difference video."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--encoder", default="h264_nvenc")
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--start-frame", type=int, default=-1)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--local-grid-cols", type=int, default=36)
    parser.add_argument("--local-grid-rows", type=int, default=21)
    parser.add_argument("--local-neighbors", type=int, default=18)
    parser.add_argument("--local-radius", type=float, default=190.0)
    parser.add_argument("--local-max-shift", type=float, default=18.0)
    parser.add_argument("--local-apply-threshold", type=float, default=0.18)
    parser.add_argument("--track-scale", type=float, default=0.5)
    parser.add_argument("--adaptive-strength", type=float, default=1.2)
    parser.add_argument("--residual-floor", type=float, default=2.0)
    parser.add_argument("--residual-gain", type=float, default=1.2)
    parser.add_argument("--color-floor", type=float, default=32.0)
    parser.add_argument("--color-gamma", type=float, default=0.72)
    parser.add_argument("--diagnostics-dir", type=Path)
    return parser.parse_args()


def resize_1080p(frame):
    return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)


def grid_points(width, height):
    spacing_x = max(28, width // 36)
    spacing_y = max(22, height // 25)
    return np.asarray(
        [
            (np.float32(x), np.float32(y))
            for y in range(spacing_y, height - spacing_y, spacing_y)
            for x in range(spacing_x, width - spacing_x, spacing_x)
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)


def tracking_points(source_gray):
    height, width = source_gray.shape
    return grid_points(width, height)


def reliable_tracks(source_gray, reference_gray, track_scale):
    scale = float(np.clip(track_scale, 0.25, 1.0))
    if scale < 0.999:
        width = max(64, int(round(source_gray.shape[1] * scale)))
        height = max(64, int(round(source_gray.shape[0] * scale)))
        source_work = cv2.resize(source_gray, (width, height), interpolation=cv2.INTER_AREA)
        reference_work = cv2.resize(reference_gray, (width, height), interpolation=cv2.INTER_AREA)
    else:
        source_work = source_gray
        reference_work = reference_gray
    source_points = tracking_points(source_work)
    reference_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        source_work,
        reference_work,
        source_points,
        None,
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.003),
    )
    if reference_points is None or forward_status is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    source_back, backward_status, backward_error = cv2.calcOpticalFlowPyrLK(
        reference_work,
        source_work,
        reference_points,
        None,
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 0.003),
    )
    if source_back is None or backward_status is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    source = source_points.reshape(-1, 2)
    reference = reference_points.reshape(-1, 2)
    returned = source_back.reshape(-1, 2)
    valid = (forward_status.reshape(-1) == 1) & (backward_status.reshape(-1) == 1)
    if forward_error is not None:
        valid &= forward_error.reshape(-1) < 35.0
    if backward_error is not None:
        valid &= backward_error.reshape(-1) < 35.0
    valid &= np.linalg.norm(returned - source, axis=1) < 1.5
    valid &= np.linalg.norm(reference - source, axis=1) < 100.0 * scale
    valid &= np.isfinite(reference).all(axis=1)
    return (
        (source[valid] / scale).astype(np.float32),
        (reference[valid] / scale).astype(np.float32),
    )


def estimate_global_homography(source, reference):
    if len(source) < 15:
        return np.eye(3, dtype=np.float64), np.zeros(len(source), dtype=bool)
    homography, mask = cv2.findHomography(reference, source, cv2.RANSAC, 2.5)
    if homography is None or not np.isfinite(homography).all():
        return np.eye(3, dtype=np.float64), np.zeros(len(source), dtype=bool)
    inliers = mask.reshape(-1).astype(bool) if mask is not None else np.zeros(len(source), dtype=bool)
    return homography, inliers


def robust_local_value(values, distances, radius):
    median = np.median(values, axis=0)
    deviations = np.linalg.norm(values - median, axis=1)
    median_deviation = float(np.median(deviations))
    cutoff = max(0.75, 3.5 * 1.4826 * median_deviation)
    keep = deviations <= cutoff
    if int(keep.sum()) < 4:
        order = np.argsort(deviations)
        keep = np.zeros(len(values), dtype=bool)
        keep[order[: min(4, len(order))]] = True
    kept_values = values[keep]
    kept_distances = distances[keep]
    sigma = max(radius * 0.42, 1.0)
    weights = np.exp(-(kept_distances * kept_distances) / (2.0 * sigma * sigma))
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-6:
        return median.astype(np.float32), 0.0
    value = (kept_values * weights[:, None]).sum(axis=0) / weight_sum
    support = min(1.0, len(kept_values) / 8.0)
    proximity = float(np.exp(-float(kept_distances.min()) / max(radius, 1.0)))
    return value.astype(np.float32), support * proximity


def local_residual_field(
    source_points,
    reference_points,
    homography,
    shape,
    grid_cols,
    grid_rows,
    neighbors,
    radius,
    max_shift,
    apply_threshold,
):
    height, width = shape
    if len(source_points) < 12:
        return np.zeros((height, width, 2), np.float32), 0, 0.0, 0.0

    inverse = np.linalg.inv(homography)
    aligned_points = cv2.perspectiveTransform(source_points.reshape(-1, 1, 2), inverse).reshape(-1, 2)
    residuals = aligned_points - reference_points
    magnitudes = np.linalg.norm(residuals, axis=1)
    keep = np.isfinite(residuals).all(axis=1) & (magnitudes <= max_shift)
    controls = reference_points[keep]
    values = residuals[keep]
    control_p95 = float(np.percentile(magnitudes[keep], 95.0)) if int(keep.sum()) else 0.0
    if len(controls) < 12:
        return np.zeros((height, width, 2), np.float32), len(controls), 0.0, control_p95
    if control_p95 < apply_threshold:
        return np.zeros((height, width, 2), np.float32), len(controls), 0.0, control_p95

    node_x = np.linspace(0.0, width - 1.0, grid_cols, dtype=np.float32)
    node_y = np.linspace(0.0, height - 1.0, grid_rows, dtype=np.float32)
    node_field = np.zeros((grid_rows, grid_cols, 2), np.float32)
    node_confidence = np.zeros((grid_rows, grid_cols), np.float32)
    neighbor_count = min(max(4, neighbors), len(controls))

    for row, y in enumerate(node_y):
        for col, x in enumerate(node_x):
            delta = controls - np.asarray([x, y], dtype=np.float32)
            distance_squared = np.einsum("ij,ij->i", delta, delta)
            indices = np.argpartition(distance_squared, neighbor_count - 1)[:neighbor_count]
            distances = np.sqrt(distance_squared[indices])
            inside = distances <= radius
            if int(inside.sum()) < 4:
                continue
            value, confidence = robust_local_value(values[indices][inside], distances[inside], radius)
            node_field[row, col] = value
            node_confidence[row, col] = confidence

    dense_x = cv2.resize(node_field[..., 0], (width, height), interpolation=cv2.INTER_CUBIC)
    dense_y = cv2.resize(node_field[..., 1], (width, height), interpolation=cv2.INTER_CUBIC)
    confidence = cv2.resize(node_confidence, (width, height), interpolation=cv2.INTER_LINEAR)
    dense_x = cv2.GaussianBlur(dense_x, (0, 0), 7.0) * confidence
    dense_y = cv2.GaussianBlur(dense_y, (0, 0), 7.0) * confidence
    magnitude = cv2.magnitude(dense_x, dense_y)
    scale = np.minimum(1.0, max_shift / np.maximum(magnitude, 1e-6))
    field = np.dstack([dense_x * scale, dense_y * scale]).astype(np.float32)
    return field, len(controls), float(confidence.mean()), control_p95


_BASE_GRIDS = {}


def base_grids(shape):
    cached = _BASE_GRIDS.get(shape)
    if cached is not None:
        return cached
    height, width = shape
    grids = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    _BASE_GRIDS[shape] = grids
    return grids


def compensate(source_gray, reference_gray, args, local, return_details=False):
    source_points, reference_points = reliable_tracks(
        source_gray, reference_gray, args.track_scale
    )
    homography, inliers = estimate_global_homography(source_points, reference_points)
    size = (reference_gray.shape[1], reference_gray.shape[0])
    globally_aligned = cv2.warpPerspective(
        source_gray,
        homography,
        size,
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    globally_valid = cv2.warpPerspective(
        np.full_like(source_gray, 255),
        homography,
        size,
        flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
    )
    stats = {
        "tracked": int(len(source_points)),
        "global_inliers": int(inliers.sum()),
        "local_controls": 0,
        "local_confidence": 0.0,
        "mean_local_shift": 0.0,
        "p95_local_shift": 0.0,
        "control_p95_residual": 0.0,
    }
    if not local:
        if return_details:
            return globally_aligned, globally_valid, stats, {
                "homography": homography,
                "local_field": None,
            }
        return globally_aligned, globally_valid, stats

    field, controls, confidence, control_p95 = local_residual_field(
        source_points,
        reference_points,
        homography,
        reference_gray.shape,
        args.local_grid_cols,
        args.local_grid_rows,
        args.local_neighbors,
        args.local_radius,
        args.local_max_shift,
        args.local_apply_threshold,
    )
    base_x, base_y = base_grids(reference_gray.shape)
    map_x = base_x + field[..., 0]
    map_y = base_y + field[..., 1]
    aligned = cv2.remap(
        globally_aligned,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = cv2.remap(
        globally_valid,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )
    shift = cv2.magnitude(field[..., 0], field[..., 1])
    stats.update(
        {
            "local_controls": int(controls),
            "local_confidence": confidence,
            "control_p95_residual": control_p95,
            "mean_local_shift": float(shift.mean()),
            "p95_local_shift": float(np.percentile(shift, 95.0)),
        }
    )
    if return_details:
        return aligned, valid, stats, {
            "homography": homography,
            "local_field": field,
        }
    return aligned, valid, stats


def compensated_difference(previous_gray, reference_gray, following_gray, args, local):
    previous_aligned, previous_valid, previous_stats = compensate(
        previous_gray, reference_gray, args, local
    )
    following_aligned, following_valid, following_stats = compensate(
        following_gray, reference_gray, args, local
    )
    valid = cv2.bitwise_and(previous_valid, following_valid)
    previous_difference = cv2.absdiff(reference_gray, previous_aligned).astype(np.float32)
    following_difference = cv2.absdiff(reference_gray, following_aligned).astype(np.float32)
    difference = (previous_difference + following_difference) * 0.5
    difference[valid == 0] = 0.0
    return difference, valid, {"previous": previous_stats, "following": following_stats}


def compensated_difference_detailed(
    previous_gray, reference_gray, following_gray, args, local
):
    previous_aligned, previous_valid, previous_stats, previous_details = compensate(
        previous_gray, reference_gray, args, local, return_details=True
    )
    following_aligned, following_valid, following_stats, following_details = compensate(
        following_gray, reference_gray, args, local, return_details=True
    )
    valid = cv2.bitwise_and(previous_valid, following_valid)
    previous_difference = cv2.absdiff(reference_gray, previous_aligned).astype(np.float32)
    following_difference = cv2.absdiff(reference_gray, following_aligned).astype(np.float32)
    difference = (previous_difference + following_difference) * 0.5
    difference[valid == 0] = 0.0
    return (
        difference,
        valid,
        {"previous": previous_stats, "following": following_stats},
        {
            "homographies": {
                "previous": previous_details["homography"],
                "following": following_details["homography"],
            },
            "local_fields": {
                "previous": previous_details["local_field"],
                "following": following_details["local_field"],
            },
        },
    )


def adaptive_residual(difference, valid, strength, residual_floor=0.0, residual_gain=1.0):
    if strength <= 0.0:
        return np.uint8(np.clip(difference, 0.0, 255.0))
    local_mean = cv2.GaussianBlur(difference, (0, 0), 13.0)
    local_square_mean = cv2.GaussianBlur(difference * difference, (0, 0), 13.0)
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    expected = strength * (local_mean + 0.55 * local_std)
    residual = np.maximum(difference - expected, 0.0)
    residual *= 1.0 + 0.45 * strength
    residual = np.maximum(residual - residual_floor, 0.0) * residual_gain
    residual[valid == 0] = 0.0
    return np.uint8(np.clip(residual, 0.0, 255.0))


def adaptive_residual_variants(difference, valid, specs):
    """Evaluate several calibration settings while sharing Gaussian statistics."""
    cached_statistics = {}
    outputs = []
    for strength, residual_floor, residual_gain in specs:
        strength = float(strength)
        if strength <= 0.0:
            outputs.append(np.uint8(np.clip(difference, 0.0, 255.0)))
            continue
        if strength not in cached_statistics:
            local_mean = cv2.GaussianBlur(difference, (0, 0), 13.0)
            local_square_mean = cv2.GaussianBlur(difference * difference, (0, 0), 13.0)
            local_std = np.sqrt(
                np.maximum(local_square_mean - local_mean * local_mean, 0.0)
            )
            cached_statistics[strength] = (local_mean, local_std)
        local_mean, local_std = cached_statistics[strength]
        expected = strength * (local_mean + 0.55 * local_std)
        residual = np.maximum(difference - expected, 0.0)
        residual *= 1.0 + 0.45 * strength
        residual = np.maximum(residual - float(residual_floor), 0.0) * float(
            residual_gain
        )
        residual[valid == 0] = 0.0
        outputs.append(np.uint8(np.clip(residual, 0.0, 255.0)))
    return outputs


def colorize_difference(gray, color_floor, gamma=0.55):
    values = gray.astype(np.float32)
    low = float(np.percentile(values, 1.0))
    high = max(float(color_floor), float(np.percentile(values, 99.7)))
    if high <= low + 1e-6:
        normalized = np.zeros_like(gray)
    else:
        normalized_values = np.clip((values - low) / (high - low), 0.0, 1.0)
        normalized = np.uint8(np.power(normalized_values, gamma) * 255.0)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def label_panel(image, text):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def comparison_frame(old_color, new_color):
    top = cv2.resize(old_color, (1920, 540), interpolation=cv2.INTER_AREA)
    bottom = cv2.resize(new_color, (1920, 540), interpolation=cv2.INTER_AREA)
    return np.concatenate(
        [
            label_panel(top, "Original YOLOMG: global H/RANSAC"),
            label_panel(
                bottom,
                "Robust: local parallax warp + adaptive texture-noise calibration",
            ),
        ],
        axis=0,
    )


class FFmpegWriter:
    def __init__(self, ffmpeg, output, fps, width, height, encoder):
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            encoder,
        ]
        if encoder == "h264_nvenc":
            command += [
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "20",
                "-b:v",
                "12M",
                "-maxrate",
                "20M",
                "-bufsize",
                "40M",
            ]
        else:
            command += ["-preset", "medium", "-crf", "20"]
        command += [
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self.output = output
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame):
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        stderr = self.process.stderr.read().decode("utf-8", errors="replace")
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed for {self.output} with code {return_code}: {stderr.strip()}"
            )


def write_progress(path, payload):
    if path is None:
        return
    encoded = json.dumps(payload, indent=2)
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(20):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    path.write_text(encoded, encoding="utf-8")


def diagnostics_metrics(old_difference, new_difference):
    height, width = old_difference.shape
    near = np.s_[int(height * 0.72) : int(height * 0.96), int(width * 0.05) : int(width * 0.95)]
    far = np.s_[int(height * 0.50) : int(height * 0.68), int(width * 0.05) : int(width * 0.95)]

    def region(values):
        return {
            "mean": float(values.mean()),
            "p95": float(np.percentile(values, 95.0)),
            "p99": float(np.percentile(values, 99.0)),
        }

    return {
        "old_near": region(old_difference[near]),
        "old_far": region(old_difference[far]),
        "new_near": region(new_difference[near]),
        "new_far": region(new_difference[far]),
    }


def main():
    args = parse_args()
    if args.diagnostics_dir is not None:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 29.97
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = (
        max(0, args.start_frame)
        if args.start_frame >= 0
        else max(0, int(round(args.start_seconds * fps)))
    )
    available = max(0, source_frames - start_frame)
    if args.frame_count > 0:
        requested = min(available, args.frame_count)
    elif args.duration_seconds > 0:
        requested = min(available, int(round(args.duration_seconds * fps)))
    else:
        requested = available
    total = max(0, requested - 2)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(3):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Not enough frames in {args.input}")
        frames.append(resize_1080p(frame))

    writer = FFmpegWriter(args.ffmpeg, args.output, fps, 1920, 1080, args.encoder)
    comparison_writer = (
        FFmpegWriter(args.ffmpeg, args.comparison_output, fps, 1920, 1080, args.encoder)
        if args.comparison_output
        else None
    )
    need_old = comparison_writer is not None or args.diagnostics_dir is not None
    done = 0
    started = time.time()
    metric_samples = []
    latest_stats = {}
    while done < total:
        previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
        following_gray = cv2.cvtColor(frames[2], cv2.COLOR_BGR2GRAY)
        old_difference = None
        old_valid = None
        old_stats = None
        if need_old:
            old_difference, old_valid, old_stats = compensated_difference(
                previous_gray, reference_gray, following_gray, args, local=False
            )
        new_difference_raw, new_valid, new_stats = compensated_difference(
            previous_gray, reference_gray, following_gray, args, local=True
        )
        old_map = None
        old_color = None
        if old_difference is not None and old_valid is not None:
            old_map = np.uint8(np.clip(old_difference, 0.0, 255.0))
            old_map[old_valid == 0] = 0
            old_color = colorize_difference(old_map, 1.0, 0.55)
        new_map = adaptive_residual(
            new_difference_raw,
            new_valid,
            args.adaptive_strength,
            args.residual_floor,
            args.residual_gain,
        )
        new_color = colorize_difference(new_map, args.color_floor, args.color_gamma)
        writer.write(new_color)
        if comparison_writer is not None and old_color is not None:
            comparison_writer.write(comparison_frame(old_color, new_color))

        if old_map is not None and done % max(1, int(round(fps))) == 0:
            metrics = diagnostics_metrics(old_map, new_map)
            metrics["output_frame"] = done
            metrics["source_frame"] = start_frame + done + 1
            metric_samples.append(metrics)
        latest_stats = {"old": old_stats, "new": new_stats}
        if args.diagnostics_dir is not None and done == min(total - 1, int(round(fps))):
            cv2.imwrite(str(args.diagnostics_dir / "rgb.jpg"), frames[1])
            if old_color is not None:
                cv2.imwrite(str(args.diagnostics_dir / "old_difference.jpg"), old_color)
            cv2.imwrite(str(args.diagnostics_dir / "new_difference.jpg"), new_color)
            if old_color is not None:
                cv2.imwrite(
                    str(args.diagnostics_dir / "comparison.jpg"),
                    comparison_frame(old_color, new_color),
                )

        done += 1
        if done % 25 == 0 or done == total:
            write_progress(
                args.progress_json,
                {
                    "status": "running",
                    "done": done,
                    "total": total,
                    "input": str(args.input),
                    "output": str(args.output),
                    "comparison_output": str(args.comparison_output)
                    if args.comparison_output
                    else None,
                    "resolution": "1920x1080",
                    "stats": latest_stats,
                    "elapsed_seconds": time.time() - started,
                    "last_output_timestamp": time.time(),
                },
            )
            print(f"[{args.input.name}] {done}/{total}", flush=True)
        frames = frames[1:]
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(resize_1080p(frame))

    writer.close()
    if comparison_writer is not None:
        comparison_writer.close()
    capture.release()
    manifest = args.output.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "method": (
                    "YOLOMG PyrLK + global RANSAC homography + robust locally interpolated "
                    "background residual warp + local residual normalization"
                ),
                "resolution": "1920x1080",
                "input": str(args.input),
                "output": str(args.output),
                "comparison_output": str(args.comparison_output)
                if args.comparison_output
                else None,
                "frames": done,
                "fps": fps,
                "start_frame": start_frame,
                "input_frame_count": requested,
                "settings": {
                    "local_grid_cols": args.local_grid_cols,
                    "local_grid_rows": args.local_grid_rows,
                    "local_neighbors": args.local_neighbors,
                    "local_radius": args.local_radius,
                    "local_max_shift": args.local_max_shift,
                    "local_apply_threshold": args.local_apply_threshold,
                    "track_scale": args.track_scale,
                    "adaptive_strength": args.adaptive_strength,
                    "residual_floor": args.residual_floor,
                    "residual_gain": args.residual_gain,
                    "color_floor": args.color_floor,
                    "color_gamma": args.color_gamma,
                },
                "metric_samples": metric_samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_progress(
        args.progress_json,
        {
            "status": "completed",
            "done": done,
            "total": done,
            "manifest": str(manifest),
            "output": str(args.output),
            "comparison_output": str(args.comparison_output)
            if args.comparison_output
            else None,
            "last_output_timestamp": time.time(),
        },
    )
    print(f"[DONE] {args.output}", flush=True)


if __name__ == "__main__":
    main()
