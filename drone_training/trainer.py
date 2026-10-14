"""Deterministic, resumable matched A/B training loop."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from .models import build_pair


class DeterministicBatchSampler(Sampler[list[tuple[int, int]]]):
    """Generate each batch solely from experiment seed and presentation cursor."""

    def __init__(self, length: int, batch_size: int, steps: int, seed: int, start_step: int = 0) -> None:
        if length < 1 or batch_size < 1 or steps < 0:
            raise ValueError("length and batch_size must be positive; steps must be nonnegative")
        self.length, self.batch_size, self.steps = int(length), int(batch_size), int(steps)
        self.seed, self.start_step = int(seed), int(start_step)

    def __len__(self) -> int:
        return max(0, self.steps - self.start_step)

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        cached_epoch = -1
        permutation: torch.Tensor | None = None
        for step in range(self.start_step, self.steps):
            batch: list[tuple[int, int]] = []
            for position in range(step * self.batch_size, (step + 1) * self.batch_size):
                epoch, offset = divmod(position, self.length)
                if epoch != cached_epoch:
                    generator = torch.Generator().manual_seed(self.seed + epoch * 1_000_003)
                    permutation = torch.randperm(self.length, generator=generator)
                    cached_epoch = epoch
                assert permutation is not None
                index = int(permutation[offset])
                augseed = int((self.seed * 2_147_483_647 + epoch * self.length + offset) % (2**31 - 1))
                batch.append((index, augseed))
            yield batch


def _stable_config(config: dict[str, Any]) -> dict[str, Any]:
    ignored = {"resume", "stop_after", "checkpoint_every", "_dataset", "_val_dataset", "_collate_fn", "_model_factory", "_eval_fn", "out"}
    return {k: config[k] for k in sorted(config) if k not in ignored and not k.startswith("_")}


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _extend_presentation_chain(chain: str, signature: str) -> str:
    return hashlib.sha256(f"{chain}:{signature}".encode()).hexdigest()


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _dataset_identity(dataset: Any) -> dict[str, Any]:
    settings = {name: getattr(dataset, name) for name in ("size", "augment", "seed", "with_motion", "motion_config") if hasattr(dataset, name)}
    records = getattr(dataset, "records", None)
    if records is not None:
        rows = [{key: row.get(key) for key in ("sample_id", "source_sha256", "source_frame_id", "bbox_xyxy", "class_id")}
                for row in records]
        return {"type": f"{type(dataset).__module__}.{type(dataset).__qualname__}", "length": len(dataset), "records": _digest(rows), "settings": settings}
    return {"type": f"{type(dataset).__module__}.{type(dataset).__qualname__}", "length": len(dataset), "settings": settings}


def _experiment_identity(config: dict[str, Any], dataset: Any, val_dataset: Any | None) -> dict[str, Any]:
    code = {}
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).with_name(name) for name in ("models.py", "trainer.py", "data.py", "evaluation.py")]
    paths += [root / "docs" / "training-prompt" / "reference_implementation" / name
              for name in ("fusion_model.py", "yolo11m_p2.yaml", "yolomg_parallax_robust_difference.py")]
    for path in paths:
        code[str(path.relative_to(root))] = _file_digest(path) if path.is_file() else "missing"
    weights = Path(str(config.get("weights", "")))
    weight_identity: Any = str(config.get("weights", ""))
    if weights.is_file():
        weight_identity = {"path": str(weights.resolve()), "sha256": _file_digest(weights)}
    return {"config": _stable_config(config), "dataset": _dataset_identity(dataset),
            "validation_dataset": None if val_dataset is None else _dataset_identity(val_dataset),
            "model_code": code, "weights": weight_identity}


def _rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(payload: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def _reconcile_log(path: Path, completed_step: int) -> None:
    if not path.is_file():
        return
    kept = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and int(json.loads(line)["step"]) <= completed_step:
            kept.append(line)
    path.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")


def _append_log(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _reconcile_best(out: Path, best: dict[str, Any], completed_step: int) -> dict[str, Any]:
    for arm in ("A", "B"):
        path = out / f"best_{arm}.pt"
        if not path.is_file():
            best[arm] = None
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = best.get(arm)
        if int(payload.get("step", -1)) > completed_step or expected is None or int(payload.get("step", -1)) != int(expected["step"]):
            path.unlink()
            best[arm] = None
    return best


def _extract_loss(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor) and output.ndim == 0:
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0].sum()
    if isinstance(output, dict) and "loss" in output:
        return output["loss"].sum()
    raise TypeError("model forward did not return a scalar loss")


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _precision_policy(device: torch.device, requested_amp: bool, *, bf16_supported: bool | None = None):
    """Use stable BF16 autocast or explicit FP32; never silently select FP16."""
    if not requested_amp or device.type != "cuda":
        return "float32", False, None
    supported = torch.cuda.is_bf16_supported() if bf16_supported is None else bool(bf16_supported)
    if supported:
        return "bfloat16", True, torch.bfloat16
    return "float32", False, None


def _make_dataset(config: dict[str, Any]):
    if "_dataset" in config:
        return config["_dataset"]
    from .data import FrameDataset, load_manifest
    root = Path(config["data_root"])
    manifest = config.get("train_manifest", "release_v3_curated/manifests/flow_train_common.jsonl")
    records = load_manifest(root, manifest, expected_split="train", require_motion=True, limit=int(config.get("train_limit", 0)))
    return FrameDataset(records, root, size=int(config.get("size", 640)), augment=True, seed=int(config.get("seed", 17)),
                        with_motion=True, cache_dir=config.get("cache_dir"), cache_gib=float(config.get("cache_gib", 10)))


def _make_val_dataset(config: dict[str, Any]):
    if "_val_dataset" in config:
        return config["_val_dataset"]
    from .data import FrameDataset, load_manifest
    root = Path(config["data_root"])
    manifest = config.get("val_manifest", "release_v3_curated/manifests/flow_validation.jsonl")
    limit = int(config.get("val_limit", 0))
    records = load_manifest(root, manifest, expected_split="validation", require_motion=True, limit=limit)
    return FrameDataset(records, root, size=int(config.get("size", 640)), augment=False,
                        seed=int(config.get("seed", 17)), with_motion=True,
                        cache_dir=config.get("cache_dir"), cache_gib=float(config.get("cache_gib", 10)))


def _assert_disjoint(train_dataset: Any, val_dataset: Any) -> None:
    train_records = getattr(train_dataset, "records", None)
    val_records = getattr(val_dataset, "records", None)
    if train_records is None or val_records is None:
        return
    def keys(rows: Any) -> set[tuple[Any, ...]]:
        return {(row.get("source_sha256"),) for row in rows}
    overlap = keys(train_records) & keys(val_records)
    if overlap:
        examples = sorted(map(str, overlap))[:3]
        raise ValueError(f"train/validation sample overlap ({len(overlap)} samples), examples={examples}")


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Train both arms on exactly the same ordered, augmented presentations."""
    config = dict(config)
    out = Path(config["out"]).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "last.pt"
    log_path = out / "training_log.jsonl"
    resume = config.get("resume", False)
    if not resume:
        managed = [out / name for name in ("last.pt", "training_log.jsonl", "summary.json", "run_config.json", "model_report.json", "best_A.pt", "best_B.pt")]
        existing = [path.name for path in managed if path.exists()]
        if existing:
            raise FileExistsError(f"managed run evidence already exists ({', '.join(existing)}); use --resume or a new --out directory")
    seed = int(config.get("seed", 17)); device = torch.device(config.get("device", "cpu"))
    accumulation = int(config.get("accumulation", 1))
    batch_size = int(config.get("batch", 4)); workers = int(config.get("workers", 0))
    if accumulation < 1 or batch_size < 1:
        raise ValueError("batch and accumulation must be positive")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    dataset = _make_dataset(config)
    configured_val_every = config.get("val_every")
    val_every = ((len(dataset) + batch_size * accumulation - 1) // (batch_size * accumulation)
                 if configured_val_every is None else int(configured_val_every))
    config["val_every"] = val_every
    val_dataset = _make_val_dataset(config) if val_every > 0 else None
    if val_dataset is not None:
        _assert_disjoint(dataset, val_dataset)
    requested_steps = config.get("steps")
    if requested_steps in (None, 0):
        epochs = int(config.get("epochs", 10))
        steps = max(1, (len(dataset) * epochs + batch_size * accumulation - 1) // (batch_size * accumulation))
    else:
        steps = int(requested_steps)
    if steps < 1:
        raise ValueError("steps must be positive")
    config["steps"] = steps
    precision_name, autocast_enabled, autocast_dtype = _precision_policy(device, bool(config.get("amp", True)))
    config["precision"] = precision_name
    factory = config.get("_model_factory", build_pair)
    pair = factory(config.get("weights", "yolo11m.pt"), seed=seed, device=device, config_path=config.get("config_path"))
    models = {arm: pair[arm] for arm in ("A", "B")}
    for model in models.values(): model.train()
    optimizers = {arm: torch.optim.AdamW(models[arm].parameters(), lr=float(config.get("lr", 1e-4))) for arm in models}
    schedulers = {arm: torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[arm], T_max=max(1, steps)) for arm in models}
    scalers = {arm: torch.amp.GradScaler("cuda", enabled=False) for arm in models}
    immutable = _experiment_identity(config, dataset, val_dataset); fingerprint = _digest(immutable)
    completed_step = 0
    presentation_chain = hashlib.sha256(b"urap.presentation_chain.v1").hexdigest()
    best: dict[str, Any] = {"A": None, "B": None}
    resume_path = Path(resume) if isinstance(resume, (str, Path)) and not isinstance(resume, bool) else checkpoint
    if resume:
        # Always deserialize checkpoint metadata and RNG byte tensors on CPU.
        # load_state_dict copies model tensors into the existing device modules,
        # and Optimizer.load_state_dict casts parameter state to parameter devices.
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["fingerprint"] != fingerprint:
            raise ValueError("resume immutable training config differs from checkpoint")
        completed_step = int(state["completed_step"]); best = state.get("best", best)
        if "presentation_chain" in state:
            presentation_chain = str(state["presentation_chain"])
        else:
            for signature in state.get("presentation", []):
                presentation_chain = _extend_presentation_chain(presentation_chain, signature)
        expected_cursor = completed_step * accumulation
        if int(state.get("presentation_cursor", expected_cursor)) != expected_cursor:
            raise ValueError("checkpoint presentation cursor is inconsistent with completed optimizer steps")
        for arm in models:
            models[arm].load_state_dict(state["models"][arm]); optimizers[arm].load_state_dict(state["optimizers"][arm])
            schedulers[arm].load_state_dict(state["schedulers"][arm]); scalers[arm].load_state_dict(state["scalers"][arm])
        _restore_rng(state["rng"])
        _reconcile_log(log_path, completed_step)
        best = _reconcile_best(out, best, completed_step)
    else:
        _write_json(out / "run_config.json", {"fingerprint": fingerprint, "immutable_config": immutable})
        _write_json(out / "model_report.json", pair.get("report", {}))
        log_path.write_text("", encoding="utf-8")
    if completed_step >= steps:
        return {"completed_step": completed_step, "steps": steps, "precision": precision_name, "presentation_digest": presentation_chain, "checkpoint": str(checkpoint), "complete": True, "best": best,
                "presentations": completed_step * accumulation * batch_size,
                "unique_train_samples_seen": min(len(dataset), completed_step * accumulation * batch_size)}
    collate = config.get("_collate_fn")
    if collate is None:
        from .data import collate_samples
        collate = collate_samples
    start_micro = completed_step * accumulation; total_micro = steps * accumulation
    sampler = DeterministicBatchSampler(len(dataset), batch_size, total_micro, seed, start_micro)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, collate_fn=collate)
    stop_after = int(config.get("stop_after", 0)); checkpoint_every = max(1, int(config.get("checkpoint_every", 100)))
    if stop_after > 0 and completed_step >= stop_after:
        return {"completed_step": completed_step, "steps": steps, "precision": precision_name, "presentation_digest": presentation_chain,
                "checkpoint": str(checkpoint), "complete": completed_step >= steps, "best": best,
                "presentations": completed_step * accumulation * batch_size,
                "unique_train_samples_seen": min(len(dataset), completed_step * accumulation * batch_size)}
    eval_fn = config.get("_eval_fn")
    if val_every > 0 and eval_fn is None:
        from .evaluation import evaluate as eval_fn

    step_losses = {"A": 0.0, "B": 0.0}; step_motion_fallbacks = 0; clock = time.monotonic()
    for micro, batch in enumerate(loader, start=start_micro):
        signature = _digest({"sampler": "epoch_permutation_v1", "seed": seed, "length": len(dataset),
                             "batch_size": batch_size, "presentation_step": micro})
        presentation_chain = _extend_presentation_chain(presentation_chain, signature)
        batch = _move_batch(batch, device)
        step_motion_fallbacks += sum(bool(row.get("motion_fallback")) for row in batch.get("metadata", []))
        losses: dict[str, torch.Tensor] = {}
        for arm in ("A", "B"):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled, dtype=autocast_dtype):
                loss = _extract_loss(models[arm](batch)) / accumulation
            if not bool(torch.isfinite(loss).all()):
                for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"non-finite loss in arm {arm} at presentation {micro}")
            losses[arm] = loss
            step_losses[arm] += float(loss.detach())
        for arm in ("A", "B"):
            scalers[arm].scale(losses[arm]).backward()
        if (micro + 1) % accumulation:
            continue
        completed_step = (micro + 1) // accumulation
        for arm in ("A", "B"):
            scalers[arm].unscale_(optimizers[arm])
        gradients_finite = all(
            bool(torch.isfinite(parameter.grad).all())
            for arm in ("A", "B") for parameter in models[arm].parameters() if parameter.grad is not None
        )
        if not gradients_finite:
            for optimizer in optimizers.values(): optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite gradient before paired optimizer step {completed_step}")
        for arm in ("A", "B"):
            scalers[arm].step(optimizers[arm]); scalers[arm].update(); optimizers[arm].zero_grad(set_to_none=True); schedulers[arm].step()
        event = {"step": completed_step, "presentation_digest": presentation_chain,
                 "lr": {arm: schedulers[arm].get_last_lr()[0] for arm in ("A", "B")},
                 "total_loss": dict(step_losses), "motion_fallback_presentations": step_motion_fallbacks}
        _append_log(log_path, event)
        elapsed = time.monotonic() - clock; clock = time.monotonic()
        print(f"[train] step {completed_step}/{steps} loss_A={step_losses['A']:.6g} loss_B={step_losses['B']:.6g} motion_fallbacks={step_motion_fallbacks} elapsed={elapsed:.2f}s", flush=True)
        step_losses = {"A": 0.0, "B": 0.0}; step_motion_fallbacks = 0
        should_stop = stop_after > 0 and completed_step >= stop_after
        if val_every > 0 and (completed_step % val_every == 0 or completed_step == steps):
            for arm in ("A", "B"):
                metrics = eval_fn(models[arm], val_dataset, out / "validation" / arm / f"step_{completed_step:08d}",
                                  batch_size=int(config.get("batch", 2)), device=str(device), workers=workers)
                models[arm].train()
                metric = float(metrics.get("map50_95") or 0.0)
                if best[arm] is None or metric > float(best[arm]["metric"]):
                    best[arm] = {"metric": metric, "step": completed_step, "metrics": metrics,
                                 "subset": int(config.get("val_limit", 0)) or None}
                    _atomic_save({"arm": arm, "step": completed_step, "metric_name": "map50_95",
                                  "metrics": metrics, "model": models[arm].state_dict(), "fingerprint": fingerprint,
                                  "immutable_config": immutable, "model_report": pair.get("report", {}).get(arm, {})}, out / f"best_{arm}.pt")
        if completed_step % checkpoint_every == 0 or completed_step == steps or should_stop:
            payload = {"version": 1, "fingerprint": fingerprint, "immutable_config": immutable,
                       "completed_step": completed_step, "presentation_chain": presentation_chain,
                       "presentation_cursor": micro + 1,
                       "models": {a: models[a].state_dict() for a in models},
                       "optimizers": {a: optimizers[a].state_dict() for a in models},
                       "schedulers": {a: schedulers[a].state_dict() for a in models},
                       "scalers": {a: scalers[a].state_dict() for a in models}, "rng": _rng_state(),
                       "model_report": pair.get("report", {}), "best": best}
            _atomic_save(payload, checkpoint)
        if should_stop: break
    report = {"completed_step": completed_step, "steps": steps, "precision": precision_name,
            "presentation_digest": presentation_chain,
            "checkpoint": str(checkpoint), "complete": completed_step >= steps,
            "model_report": pair.get("report", {}), "best": best,
            "presentations": completed_step * accumulation * batch_size,
            "unique_train_samples_seen": min(len(dataset), completed_step * accumulation * batch_size)}
    _write_json(out / "summary.json", report)
    return report
