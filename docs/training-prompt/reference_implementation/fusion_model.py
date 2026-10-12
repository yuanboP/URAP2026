"""YOLO RGB/motion fusion variants for the bounded 2026-10-01 pilot.

The public model interface accepts four-channel ``RGBD`` tensors.  Variant A
uses RGB only, variant B feeds D through a zero-initialized fourth stem input,
and variant C adds a zero-initialized stride-4 D branch to the P2 feature passed
to Detect.  C deliberately leaves every lower-resolution neck feature alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

Variant = Literal["A", "B", "C"]


def _first_conv(root: nn.Module) -> tuple[nn.Module, str, nn.Conv2d]:
    for parent in root.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Conv2d):
                return parent, name, child
    raise ValueError("base model has no Conv2d input stem")


def _expand_stem_to_four_channels(root: nn.Module) -> None:
    parent, name, old = _first_conv(root)
    if old.in_channels != 3 or old.groups != 1:
        raise ValueError(
            f"expected a three-channel, ungrouped input stem; got "
            f"in_channels={old.in_channels}, groups={old.groups}"
        )
    new = nn.Conv2d(
        4,
        old.out_channels,
        old.kernel_size,
        old.stride,
        old.padding,
        old.dilation,
        old.groups,
        old.bias is not None,
        old.padding_mode,
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3].copy_(old.weight)
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, name, new)


def _detect_module(root: nn.Module) -> nn.Module:
    sequence = getattr(root, "model", None)
    if sequence is not None and len(sequence):
        candidate = sequence[-1]
        if hasattr(candidate, "f"):
            return candidate
    raise ValueError("base model does not end in an Ultralytics-style Detect module")


def _infer_p2_channels(detect: nn.Module) -> int:
    try:
        first_tower = detect.cv2[0]
        first_layer = first_tower[0]
        conv = first_layer.conv if hasattr(first_layer, "conv") else first_layer
        return int(conv.in_channels)
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("could not infer the first Detect input channel count") from exc


def _copy_compatible_state(source: nn.Module, target: nn.Module) -> int:
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = 0
    for name, value in source_state.items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name].copy_(value)
            copied += value.numel()
    target.load_state_dict(target_state)
    return copied


def _transplant_yolo11_p3(source: nn.Module, target: nn.Module) -> dict[str, Any]:
    """Transplant YOLO11 P3 weights into the P2 topology by semantic layer map."""
    source_layers = source.model
    target_layers = target.model
    layer_map = {**{index: index for index in range(17)}, 17: 23, 19: 25, 20: 26, 22: 28}
    copied = 0
    mapped: list[tuple[int, int]] = []
    for source_index, target_index in layer_map.items():
        count = _copy_compatible_state(source_layers[source_index], target_layers[target_index])
        if count:
            copied += count
            mapped.append((source_index, target_index))

    # Old Detect branches P3/P4/P5 become target branches 1/2/3.  Class logits
    # are copied only when nc and tensor shape agree; nc=80 -> nc=1 stays fresh.
    source_detect = source_layers[-1]
    target_detect = target_layers[-1]
    source_state = source_detect.state_dict()
    target_state = target_detect.state_dict()
    detect_tensors = 0
    for source_name, value in source_state.items():
        parts = source_name.split(".")
        if len(parts) > 1 and parts[0] in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}:
            try:
                parts[1] = str(int(parts[1]) + 1)
            except ValueError:
                pass
        target_name = ".".join(parts)
        if target_name in target_state and target_state[target_name].shape == value.shape:
            target_state[target_name].copy_(value)
            copied += value.numel()
            detect_tensors += 1
    target_detect.load_state_dict(target_state)
    return {
        "strategy": "semantic_yolo11_p3_to_p2",
        "copied_parameters": copied,
        "layer_map": mapped,
        "detect_tensors": detect_tensors,
    }


class FlowYoloFusionModel(nn.Module):
    """Adapt an Ultralytics detection model to the pilot's RGBD contract."""

    def __init__(self, base_model: nn.Module, variant: Variant, p2_channels: int | None = None) -> None:
        super().__init__()
        normalized = str(variant).upper()
        if normalized not in {"A", "B", "C"}:
            raise ValueError(f"unknown fusion variant: {variant!r}")
        self.variant: Variant = normalized  # type: ignore[assignment]
        self.base_model = base_model
        self._motion: torch.Tensor | None = None
        self._detect_hook_handle: Any | None = None

        if self.variant == "B":
            _expand_stem_to_four_channels(self.base_model)

        if self.variant == "C":
            detect = _detect_module(self.base_model)
            channels = int(p2_channels or _infer_p2_channels(detect))
            hidden = max(8, min(32, channels // 4))
            self.motion_branch = nn.Sequential(
                nn.Conv2d(1, hidden, 3, stride=2, padding=1, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, channels, 3, stride=2, padding=1, bias=False),
            )
            nn.init.zeros_(self.motion_branch[-1].weight)
            self._detect_hook_handle = detect.register_forward_pre_hook(self._fuse_detect_p2)
        else:
            self.motion_branch = None

    def _fuse_detect_p2(self, _module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if self._motion is None:
            raise RuntimeError("motion input was not set before Detect forward")
        features = list(args[0])
        residual = self.motion_branch(self._motion)
        if residual.shape[-2:] != features[0].shape[-2:]:
            residual = nn.functional.interpolate(residual, size=features[0].shape[-2:], mode="bilinear", align_corners=False)
        features[0] = features[0] + residual
        return (features, *args[1:])

    @staticmethod
    def _image(value: Any) -> torch.Tensor:
        image = value["img"] if isinstance(value, dict) else value
        if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[1] != 4:
            raise ValueError("expected BCHW RGBD input with exactly four channels")
        return image

    def forward(self, value: Any):
        image = self._image(value)
        if self.variant == "B":
            return self.base_model(value)

        rgb = image[:, :3]
        forwarded = value
        if isinstance(value, dict):
            forwarded = dict(value)
            forwarded["img"] = rgb
        else:
            forwarded = rgb

        if self.variant == "C":
            self._motion = image[:, 3:4]
        try:
            return self.base_model(forwarded)
        finally:
            self._motion = None


def build_model(
    variant: Variant,
    config_path: str | Path,
    checkpoint_path: str | Path | None,
    seed: int,
) -> FlowYoloFusionModel:
    """Build one variant without resolving model names or downloading weights."""
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"model config does not exist: {config}")
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path).expanduser().resolve()
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    base_model = YOLO(str(config), task="detect", verbose=False).model
    # DetectionModel constructed directly from YAML keeps args as a plain dict;
    # the loss expects attribute-style hyperparameters normally installed by a
    # Trainer.  Populate the standard defaults for the standalone pilot loop.
    base_model.args = get_cfg(base_model.args)
    load_info: dict[str, Any] = {
        "config_path": str(config),
        "checkpoint_path": None if checkpoint is None else str(checkpoint),
        "strategy": "random_initialization",
        "copied_parameters": 0,
    }
    if checkpoint is not None:
        source_model = YOLO(str(checkpoint), task="detect", verbose=False).model
        if len(source_model.model) == len(base_model.model):
            load_info.update(
                strategy="shape_compatible_same_topology",
                copied_parameters=_copy_compatible_state(source_model, base_model),
            )
        elif len(source_model.model) == 24 and len(base_model.model) == 30:
            load_info.update(_transplant_yolo11_p3(source_model, base_model))
        else:
            raise ValueError(
                "unsupported checkpoint topology: "
                f"source_layers={len(source_model.model)}, target_layers={len(base_model.model)}"
            )
    wrapped = FlowYoloFusionModel(base_model, variant)
    wrapped.load_provenance = load_info
    wrapped.parameter_info = {
        "total": sum(parameter.numel() for parameter in wrapped.parameters()),
        "trainable": sum(parameter.numel() for parameter in wrapped.parameters() if parameter.requires_grad),
        "motion_branch": 0
        if wrapped.motion_branch is None
        else sum(parameter.numel() for parameter in wrapped.motion_branch.parameters()),
    }
    return wrapped
