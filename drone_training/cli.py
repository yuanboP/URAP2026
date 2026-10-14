"""Command line interface for the portable drone-training workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import urllib.request
from pathlib import Path
from typing import Any


TRAIN_MANIFEST = "release_v3_curated/manifests/flow_train_common.jsonl"
VAL_MANIFEST = "release_v3_curated/manifests/flow_validation.jsonl"
SAM_MANIFEST = "release_v3_curated/manifests/sam_positive_train.jsonl"
SAM_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
DATA_REVISION = "aaf817f51dd670579a8e2c30a5db4930cf5eeb7f"
EVAL_CONFIDENCE_FLOOR = .001
EVAL_NMS_IOU = .7
EVAL_MAX_DET = 100
WEIGHT_URLS = {
    "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt",
    "sam2.1_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    "sam2.1_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
}
WEIGHT_SHA256 = {
    "yolo11m.pt": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
    "sam2.1_hiera_large.pt": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    "sam2.1_hiera_tiny.pt": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
}


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _download(**kwargs):
    from .download import download_dataset
    return download_dataset(**kwargs)


def _train_yolo(config):
    from .trainer import train
    return train(config)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _weights(directory: str) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    files = []
    for name, url in WEIGHT_URLS.items():
        destination = root / name
        if not destination.is_file():
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(url) as source, temporary.open("wb") as target:
                while block := source.read(8 << 20): target.write(block)
            temporary.replace(destination)
        actual = _sha256(destination)
        if actual != WEIGHT_SHA256[name]:
            raise ValueError(f"official weight SHA256 mismatch for {destination}: expected {WEIGHT_SHA256[name]}, got {actual}")
        files.append({"name": name, "path": str(destination), "bytes": destination.stat().st_size,
                      "sha256": actual, "url": url})
    receipt = {"files": files, "sam2_source_commit": SAM_COMMIT}
    receipt_path = root / "weights_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["receipt"] = str(receipt_path)
    return receipt


def _dataset(root: str, manifest: str, split: str, size: int, limit: int, motion: bool = True, cache_dir=None, cache_gib=10):
    from .data import FrameDataset, load_manifest
    records = load_manifest(root, manifest, expected_split=split, require_motion=motion, limit=limit)
    return FrameDataset(records, root, size=size, augment=False, with_motion=motion, cache_dir=cache_dir, cache_gib=cache_gib)


def _load_detector(args):
    import torch
    from .models import build_pair
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("arm") is not None and checkpoint["arm"] != args.arm:
        raise ValueError(f"checkpoint arm {checkpoint['arm']} does not match requested arm {args.arm}")
    pair = build_pair(args.weights, seed=args.seed, device="cpu")
    state = checkpoint["models"][args.arm] if "models" in checkpoint else checkpoint.get("model", checkpoint)
    pair[args.arm].load_state_dict(state, strict=True)
    model = pair[args.arm]; del checkpoint, pair
    return model.to(args.device)


def _evaluation_code_sha256():
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in ("cli.py", "data.py", "evaluation.py", "models.py"):
        path = root / name
        digest.update(name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def _threshold(args) -> float:
    if args.split != "test": return float(args.threshold)
    if not args.threshold_file:
        raise ValueError("test evaluation requires --threshold-file produced by calibrate")
    artifact = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
    if artifact.get("split") != "validation":
        raise ValueError("--threshold-file must be a validation calibration artifact")
    provenance = artifact.get("provenance") or {}
    if provenance.get("arm") != args.arm:
        raise ValueError("calibration arm does not match requested evaluation arm")
    if provenance.get("checkpoint_sha256") != _sha256(Path(args.checkpoint)):
        raise ValueError("calibration checkpoint does not match evaluation checkpoint")
    if provenance.get("data_revision") != DATA_REVISION:
        raise ValueError("calibration data revision does not match pinned release")
    if provenance.get("code_sha256") != _evaluation_code_sha256():
        raise ValueError("calibration evaluation code does not match current code")
    expected_inference = {
        "input_size": int(args.size),
        "confidence_floor": EVAL_CONFIDENCE_FLOOR,
        "nms_iou": EVAL_NMS_IOU,
        "max_det": EVAL_MAX_DET,
    }
    for field, expected in expected_inference.items():
        if provenance.get(field) != expected:
            raise ValueError(f"calibration {field} does not match evaluation configuration")
    return float(artifact["threshold"])


def _doctor() -> dict[str, Any]:
    result: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "sam2_required_commit": SAM_COMMIT}
    for package in ("torch", "ultralytics", "pycocotools", "cv2", "huggingface_hub", "sam2"):
        try:
            module = __import__(package)
            result[package] = getattr(module, "__version__", "installed")
        except Exception as exc:
            result[package] = f"missing: {type(exc).__name__}"
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                      cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        pass
    result["ready"] = all(not str(result.get(name, "missing")).startswith("missing") for name in ("torch", "ultralytics", "pycocotools", "cv2", "huggingface_hub", "sam2"))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m drone_training", description="Matched RGB/RGBD drone detector training")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="download the pinned full dataset (or an explicit smoke subset)")
    download.add_argument("--data-root", required=True); download.add_argument("--smoke", action="store_true")
    download.add_argument("--train-limit", type=int, default=8); download.add_argument("--val-limit", type=int, default=4)
    download.add_argument("--workers", type=int, default=4)
    weights = sub.add_parser("weights", help="download official YOLO11m and SAM 2.1 weights with a SHA256 receipt")
    weights.add_argument("--out", required=True)

    train = sub.add_parser("train-yolo", help="train matched YOLO11m-P2 A/B arms; defaults to the full 10-epoch run")
    train.add_argument("--data-root", required=True); train.add_argument("--out", required=True); train.add_argument("--weights", required=True)
    train.add_argument("--train-manifest", default=TRAIN_MANIFEST); train.add_argument("--val-manifest", default=VAL_MANIFEST)
    train.add_argument("--epochs", type=int, default=10); train.add_argument("--steps", type=int, default=0)
    train.add_argument("--size", type=int, default=1280); train.add_argument("--batch", type=int, default=2); train.add_argument("--accumulation", type=int, default=1)
    train.add_argument("--lr", type=float, default=1e-4); train.add_argument("--seed", type=int, default=17); train.add_argument("--workers", type=int, default=4)
    train.add_argument("--device", default="cuda"); train.add_argument("--no-amp", action="store_true")
    train.add_argument("--checkpoint-every", type=int, default=100); train.add_argument("--val-every", type=int)
    train.add_argument("--train-limit", type=int, default=0); train.add_argument("--val-limit", type=int, default=0)
    train.add_argument("--cache-dir"); train.add_argument("--cache-gib", type=float, default=10); train.add_argument("--resume", nargs="?", const=True, default=False)
    train.add_argument("--stop-after", type=int, default=0)

    evaluate = sub.add_parser("evaluate", help="evaluate arm A or B in original coordinates")
    evaluate.add_argument("--data-root", required=True); evaluate.add_argument("--manifest", required=True); evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--checkpoint", required=True); evaluate.add_argument("--weights"); evaluate.add_argument("--arm", choices=("A", "B"), required=True)
    evaluate.add_argument("--out", required=True); evaluate.add_argument("--device", default="cuda"); evaluate.add_argument("--size", type=int, default=1280)
    evaluate.add_argument("--batch", type=int, default=2); evaluate.add_argument("--workers", type=int, default=4); evaluate.add_argument("--limit", type=int, default=0)
    evaluate.add_argument("--cache-dir"); evaluate.add_argument("--cache-gib", type=float, default=10)
    evaluate.add_argument("--threshold", type=float, default=.25); evaluate.add_argument("--threshold-file"); evaluate.add_argument("--seed", type=int, default=17)
    calibrate = sub.add_parser("calibrate", help="select a frozen threshold from validation predictions")
    calibrate.add_argument("--predictions", required=True); calibrate.add_argument("--out", required=True); calibrate.add_argument("--fp-per-frame", type=float)
    compare = sub.add_parser("compare", help="paired A/B comparison with video bootstrap for operating metrics")
    compare.add_argument("--a-predictions", required=True); compare.add_argument("--b-predictions", required=True)
    compare.add_argument("--a-calibration", required=True); compare.add_argument("--b-calibration", required=True)
    compare.add_argument("--out", required=True); compare.add_argument("--resamples", type=int, default=1000); compare.add_argument("--seed", type=int, default=17)

    pseudo = sub.add_parser("sam-pseudo", help="generate pseudo masks for all positive SAM training rows")
    pseudo.add_argument("--data-root", required=True); pseudo.add_argument("--manifest", default=SAM_MANIFEST); pseudo.add_argument("--out", required=True)
    pseudo.add_argument("--teacher-checkpoint", required=True); pseudo.add_argument("--device", default="cuda"); pseudo.add_argument("--limit", type=int, default=0)
    sam_train = sub.add_parser("sam-train", help="train the SAM 2.1 Tiny decoder from accepted pseudo masks")
    sam_train.add_argument("--pseudo-manifest", required=True); sam_train.add_argument("--student-checkpoint", required=True); sam_train.add_argument("--out", required=True)
    sam_train.add_argument("--epochs", type=int, default=5); sam_train.add_argument("--lr", type=float, default=1e-5); sam_train.add_argument("--device", default="cuda")
    sam_train.add_argument("--resume"); sam_train.add_argument("--checkpoint-every", type=int, default=25); sam_train.add_argument("--stop-after", type=int, default=0)
    sam_infer = sub.add_parser("sam-infer", help="run prompt-conditioned SAM inference for external proposal boxes")
    sam_infer.add_argument("--image", required=True); sam_infer.add_argument("--boxes", required=True); sam_infer.add_argument("--checkpoint", required=True)
    sam_infer.add_argument("--student-checkpoint", required=True); sam_infer.add_argument("--out", required=True); sam_infer.add_argument("--device", default="cuda")
    head_train = sub.add_parser("sam-head-train", help="train a drone-scoring proposal head")
    head_train.add_argument("--candidate-jsonl", required=True); head_train.add_argument("--data-root", required=True); head_train.add_argument("--student-checkpoint", required=True)
    head_train.add_argument("--out", required=True); head_train.add_argument("--device", default="cuda"); head_train.add_argument("--epochs", type=int, default=5)
    head_train.add_argument("--steps", type=int); head_train.add_argument("--batch", type=int, default=32); head_train.add_argument("--checkpoint-every", type=int, default=100)
    head_train.add_argument("--lr", type=float, default=1e-3); head_train.add_argument("--seed", type=int, default=17); head_train.add_argument("--resume", action="store_true"); head_train.add_argument("--stop-after", type=int, default=0)
    head_infer = sub.add_parser("sam-head-infer", help="score external proposals with the trained proposal head")
    head_infer.add_argument("--image", required=True); head_infer.add_argument("--proposals", required=True); head_infer.add_argument("--checkpoint", required=True)
    head_infer.add_argument("--student-checkpoint", required=True); head_infer.add_argument("--out", required=True); head_infer.add_argument("--device", default="cuda")
    sub.add_parser("doctor", help="report dependencies and accelerator availability without claiming training readiness")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download": result = _download(data_root=args.data_root, smoke=args.smoke, train_limit=args.train_limit, val_limit=args.val_limit, workers=args.workers)
    elif args.command == "weights": result = _weights(args.out)
    elif args.command == "train-yolo":
        config = {key: value for key, value in vars(args).items() if key not in {"command", "no_amp"}}
        config["amp"] = not args.no_amp
        result = _train_yolo(config)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        manifest = Path(args.manifest)
        manifest = manifest.resolve() if manifest.is_absolute() else (Path(args.data_root) / manifest).resolve()
        model = _load_detector(args); dataset = _dataset(args.data_root, str(manifest), args.split, args.size, args.limit, cache_dir=args.cache_dir, cache_gib=args.cache_gib)
        provenance = {"arm": args.arm, "checkpoint_sha256": _sha256(Path(args.checkpoint)),
                      "manifest_sha256": _sha256(manifest), "data_revision": DATA_REVISION,
                      "code_sha256": _evaluation_code_sha256(), "input_size": int(args.size),
                      "confidence_floor": EVAL_CONFIDENCE_FLOOR, "nms_iou": EVAL_NMS_IOU,
                      "max_det": EVAL_MAX_DET}
        result = evaluate(model, dataset, args.out, batch_size=args.batch, device=args.device,
                          confidence_floor=EVAL_CONFIDENCE_FLOOR, nms_iou=EVAL_NMS_IOU,
                          max_det=EVAL_MAX_DET, threshold=_threshold(args), workers=args.workers,
                          provenance=provenance)
    elif args.command == "calibrate":
        from .evaluation import calibrate
        result = calibrate(args.predictions, args.out, args.fp_per_frame)
    elif args.command == "compare":
        from .evaluation import compare_predictions, load_prediction_provenance
        def rows(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        calibration_a = json.loads(Path(args.a_calibration).read_text(encoding="utf-8")); calibration_b = json.loads(Path(args.b_calibration).read_text(encoding="utf-8"))
        prediction_a = load_prediction_provenance(args.a_predictions); prediction_b = load_prediction_provenance(args.b_predictions)
        for arm, calibration, prediction in (("A", calibration_a, prediction_a), ("B", calibration_b, prediction_b)):
            source = calibration.get("provenance") or {}
            if source.get("arm") != arm or prediction.get("arm") != arm:
                raise ValueError(f"comparison {arm} arm provenance mismatch")
            for field in ("checkpoint_sha256", "data_revision", "code_sha256", "input_size",
                          "confidence_floor", "nms_iou", "max_det"):
                if source.get(field) != prediction.get(field):
                    raise ValueError(f"comparison {arm} {field} provenance mismatch")
        if prediction_a.get("sample_digest") != prediction_b.get("sample_digest"):
            raise ValueError("comparison prediction sample digests differ")
        threshold_a = float(calibration_a["threshold"]); threshold_b = float(calibration_b["threshold"])
        result = compare_predictions(rows(args.a_predictions), rows(args.b_predictions), threshold_a, threshold_b, args.resamples, args.seed)
        output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif args.command == "sam-pseudo":
        from .sam import pseudo
        manifest = Path(args.manifest); manifest = manifest if manifest.is_absolute() else Path(args.data_root) / manifest
        result = pseudo({"data_root": args.data_root, "manifest": str(manifest), "out": args.out, "teacher_checkpoint": args.teacher_checkpoint, "device": args.device, "maximum": args.limit})
    elif args.command == "sam-train":
        from .sam import train_sam
        result = train_sam({"pseudo_manifest": args.pseudo_manifest, "student_checkpoint": args.student_checkpoint, "out": args.out, "epochs": args.epochs, "lr": args.lr, "device": args.device, "resume": args.resume, "checkpoint_every": args.checkpoint_every, "stop_after": args.stop_after})
    elif args.command == "sam-infer":
        from .sam import infer_sam
        boxes = json.loads(Path(args.boxes).read_text(encoding="utf-8")); result = infer_sam({"image": args.image, "boxes_xyxy": boxes, "checkpoint": args.checkpoint, "student_checkpoint": args.student_checkpoint, "device": args.device})
        import numpy as np
        output = Path(args.out).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
        masks = np.stack([row.pop("mask") for row in result]); mask_path = output.with_suffix(".masks.npz")
        np.savez_compressed(mask_path, masks=masks)
        for index, row in enumerate(result): row.update(mask_index=index, masks_npz=str(mask_path))
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    elif args.command == "sam-head-train":
        from .sam_head import train_head
        result = train_head(vars(args))
    elif args.command == "sam-head-infer":
        from PIL import Image
        import numpy as np
        from .sam_head import infer_head
        image = np.asarray(Image.open(args.image).convert("RGB")); proposals = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
        result = infer_head(image, proposals, args.checkpoint, student_checkpoint=args.student_checkpoint, device=args.device)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else: result = _doctor()
    _json(result)
    return 0
