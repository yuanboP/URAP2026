from __future__ import annotations

import copy

import torch
from torch import nn

from drone_training.models import FlowYoloFusionModel, build_pair


class _TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 2, 1, bias=True)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, value):
        image = value["img"] if isinstance(value, dict) else value
        return self.bn(self.stem(image))


def test_variant_a_ignores_motion_and_variant_b_starts_identically() -> None:
    torch.manual_seed(4)
    base = _TinyBase()
    a = FlowYoloFusionModel(copy.deepcopy(base), "A")
    b = FlowYoloFusionModel(copy.deepcopy(base), "B")
    for key, value in a.base_model.state_dict().items():
        if key == "stem.weight":
            assert torch.equal(value, b.base_model.state_dict()[key][:, :3])
        else:
            assert torch.equal(value, b.base_model.state_dict()[key])
    rgb = torch.randn(2, 3, 5, 5)
    d1 = torch.randn(2, 1, 5, 5)
    d2 = torch.randn(2, 1, 5, 5)

    assert torch.equal(a(torch.cat((rgb, d1), 1)), a(torch.cat((rgb, d2), 1)))
    assert torch.allclose(a(torch.cat((rgb, d1), 1)), b(torch.cat((rgb, d1), 1)))
    assert torch.count_nonzero(b.base_model.stem.weight[:, 3:]).item() == 0


def test_build_pair_uses_same_source_state_and_reports_every_target_key() -> None:
    calls: list[tuple[str, int]] = []

    def builder(weights: str, seed: int, **_kwargs):
        calls.append((weights, seed))
        torch.manual_seed(seed)
        model = _TinyBase()
        return model, {"source": weights, "keys": {"stem.weight": "loaded", "stem.bias": "loaded"}}

    pair = build_pair("generic.pt", seed=17, device="cpu", model_builder=builder)

    assert calls == [("generic.pt", 17)]
    assert set(pair) >= {"A", "B", "report"}
    assert torch.equal(pair["A"].base_model.stem.weight, pair["B"].base_model.stem.weight[:, :3])
    assert pair["report"]["A"]["keys"]["base_model.stem.weight"] == "loaded"
    assert pair["report"]["B"]["keys"]["base_model.stem.weight"] == "loaded"
    assert pair["report"]["B"]["keys"]["base_model.stem.weight.motion"] == "initialized_zero"
