"""Matched YOLO11m-P2 model construction for the A/B experiment."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn


def _first_conv(root: nn.Module) -> tuple[nn.Module, str, nn.Conv2d]:
    for parent in root.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d):
                return parent, name, child
    raise ValueError("base model has no Conv2d input stem")


def _expand_stem(root: nn.Module) -> None:
    parent, name, old = _first_conv(root)
    if old.in_channels != 3 or old.groups != 1:
        raise ValueError(f"expected an ungrouped RGB stem, got {old.in_channels} channels")
    new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                    old.dilation, old.groups, old.bias is not None, old.padding_mode,
                    device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, name, new)


class FlowYoloFusionModel(nn.Module):
    """Give both experiment arms one public BCHW RGBD interface."""

    def __init__(self, base_model: nn.Module, variant: str) -> None:
        super().__init__()
        variant = variant.upper()
        if variant not in {"A", "B"}:
            raise ValueError(f"variant must be A or B, got {variant!r}")
        self.variant = variant
        self.base_model = base_model
        if variant == "B":
            _expand_stem(self.base_model)

    def forward(self, value: Any):
        image = value["img"] if isinstance(value, dict) else value
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 4:
            raise ValueError("expected BCHW RGBD input with exactly four channels")
        if self.variant == "B":
            return self.base_model(value)
        forwarded: Any = image[:, :3]
        if isinstance(value, dict):
            forwarded = dict(value)
            forwarded["img"] = image[:, :3]
        return self.base_model(forwarded)


def _reference_builder(weights: str | Path, seed: int, *, config_path: str | Path | None = None, **_: Any):
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "training-prompt" / "reference_implementation" / "fusion_model.py"
    yaml_path = Path(config_path) if config_path else source.with_name("yolo11m_p2.yaml")
    spec = importlib.util.spec_from_file_location("_urap_reference_fusion_model", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load published model builder: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wrapped = module.build_model("A", yaml_path, weights, seed)
    base = wrapped.base_model
    provenance = dict(getattr(wrapped, "load_provenance", {}))
    target_state = base.state_dict()
    keys = {name: "initialized" for name in target_state}
    transfers: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    if weights is not None:
        from ultralytics import YOLO
        source_state = YOLO(str(weights), task="detect", verbose=False).model.state_dict()
        inverse_layers = {**{index: index for index in range(17)}, 23: 17, 25: 19, 26: 20, 28: 22}

        def source_name(target_name: str) -> str | None:
            parts = target_name.split(".")
            if len(parts) < 3 or parts[0] != "model":
                return target_name
            target_layer = int(parts[1])
            if target_layer in inverse_layers:
                parts[1] = str(inverse_layers[target_layer])
                return ".".join(parts)
            if target_layer != 29:
                return None
            parts[1] = "23"
            if len(parts) > 3 and parts[2] in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}:
                branch = int(parts[3])
                if branch == 0:
                    return None
                parts[3] = str(branch - 1)
            return ".".join(parts)

        used: set[str] = set()
        for target_name, target_value in target_state.items():
            candidate = source_name(target_name)
            if candidate in source_state and source_state[candidate].shape == target_value.shape:
                keys[target_name] = "loaded_semantic" if candidate != target_name else "loaded"
                transfers.append({"source": candidate, "target": target_name, "status": keys[target_name]})
                used.add(candidate)
        rejected = [{"source": name, "reason": "no_shape_compatible_semantic_target"}
                    for name in source_state if name not in used]
    provenance["keys"] = keys
    provenance["transfers"] = transfers
    provenance["rejected"] = rejected
    return base, provenance


def _arm_report(model: FlowYoloFusionModel, source_report: dict[str, Any]) -> dict[str, Any]:
    source_keys = source_report.get("keys", {})
    keys: dict[str, str] = {}
    for name in model.state_dict():
        public_name = name
        base_name = name.removeprefix("base_model.")
        if model.variant == "B" and base_name.endswith("weight"):
            try:
                _, _, stem = _first_conv(model.base_model)
                stem_name = next(k for k, v in model.base_model.named_modules() if v is stem)
                if base_name == f"{stem_name}.weight":
                    keys[public_name] = source_keys.get(base_name, "loaded")
                    keys[f"{public_name}.motion"] = "initialized_zero"
                    continue
            except (StopIteration, ValueError):
                pass
        keys[public_name] = source_keys.get(base_name, "initialized")
    report = {k: v for k, v in source_report.items() if k != "keys"}
    report["keys"] = keys
    report["rejected"] = list(source_report.get("rejected", []))
    return report


def build_pair(
    weights: str | Path,
    seed: int = 17,
    device: str | torch.device = "cpu",
    *,
    config_path: str | Path | None = None,
    model_builder: Callable[..., tuple[nn.Module, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Build matched P2 arms from one initialized source model."""
    builder = model_builder or _reference_builder
    base, source_report = builder(weights, seed, config_path=config_path)
    a = FlowYoloFusionModel(copy.deepcopy(base), "A").to(device)
    b = FlowYoloFusionModel(copy.deepcopy(base), "B").to(device)
    return {"A": a, "B": b, "report": {"A": _arm_report(a, source_report), "B": _arm_report(b, source_report)}}
