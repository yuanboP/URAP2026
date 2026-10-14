from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import pytest
from torch import nn

from drone_training.trainer import DeterministicBatchSampler, _precision_policy, train


class _ToyDataset:
    def __init__(self, length: int = 9) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, request):
        index, augseed = request if isinstance(request, tuple) else (request, 0)
        generator = torch.Generator().manual_seed(int(augseed))
        noise = torch.rand((), generator=generator)
        image = torch.tensor([index / 10, float(noise), 0.25, 0.0]).view(4, 1, 1)
        return {"img": image, "target": torch.tensor([index % 3], dtype=torch.float32), "metadata": {"index": index}}


class _ToyModel(nn.Module):
    def __init__(self, arm: str) -> None:
        super().__init__()
        self.arm = arm
        self.linear = nn.Linear(4, 1)

    def forward(self, batch):
        prediction = self.linear(batch["img"].flatten(1)).squeeze(1)
        return ((prediction - batch["target"].squeeze(1)) ** 2).mean()


def _collate(samples):
    return {
        "img": torch.stack([sample["img"] for sample in samples]),
        "target": torch.stack([sample["target"] for sample in samples]),
        "metadata": [sample["metadata"] for sample in samples],
    }


def _factory(_weights, seed, device="cpu", **_kwargs):
    torch.manual_seed(seed)
    base = _ToyModel("A")
    a = copy.deepcopy(base).to(device)
    b = copy.deepcopy(base).to(device)
    b.arm = "B"
    return {"A": a, "B": b, "report": {"A": {}, "B": {}}}


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right):
            _assert_nested_equal(lhs, rhs)
    else:
        assert left == right


def test_sampler_is_cursor_reproducible_and_pairs_index_with_augmentation_seed() -> None:
    full = list(DeterministicBatchSampler(7, batch_size=3, steps=5, seed=19))
    resumed = list(DeterministicBatchSampler(7, batch_size=3, steps=5, seed=19, start_step=2))
    assert resumed == full[2:]
    assert all(isinstance(item, tuple) and len(item) == 2 for batch in full for item in batch)
    flattened = [index for batch in full for index, _ in batch]
    assert len(set(flattened[:7])) == 7
    assert flattened[7] in set(flattened[:7])


def test_amp_policy_uses_bfloat16_only_when_cuda_supports_it() -> None:
    assert _precision_policy(torch.device("cuda"), True, bf16_supported=True) == ("bfloat16", True, torch.bfloat16)
    assert _precision_policy(torch.device("cuda"), True, bf16_supported=False) == ("float32", False, None)
    assert _precision_policy(torch.device("cpu"), True, bf16_supported=True) == ("float32", False, None)
    assert _precision_policy(torch.device("cuda"), False, bf16_supported=True) == ("float32", False, None)


def test_interrupted_resume_matches_uninterrupted_weights_optimizer_and_samples(tmp_path: Path) -> None:
    common = {
        "out": str(tmp_path / "full"), "weights": "toy", "steps": 6, "batch": 2,
        "accumulation": 2, "seed": 23, "lr": 0.01, "device": "cpu", "amp": False,
        "checkpoint_every": 1, "val_every": 0, "workers": 0,
        "_dataset": _ToyDataset(), "_collate_fn": _collate, "_model_factory": _factory,
    }
    full = train(common)

    split_config = dict(common, out=str(tmp_path / "split"), stop_after=3)
    partial = train(split_config)
    assert partial["completed_step"] == 3
    resumed = train(dict(split_config, stop_after=0, resume=True))

    assert resumed["completed_step"] == full["completed_step"] == 6
    assert resumed["presentation_digest"] == full["presentation_digest"]
    full_state = torch.load(Path(common["out"]) / "last.pt", map_location="cpu", weights_only=False)
    resumed_state = torch.load(Path(split_config["out"]) / "last.pt", map_location="cpu", weights_only=False)
    assert "presentation_chain" in resumed_state
    assert "presentation" not in resumed_state
    assert (Path(common["out"]) / "training_log.jsonl").read_text() == (Path(split_config["out"]) / "training_log.jsonl").read_text()
    first_log = json.loads((Path(common["out"]) / "training_log.jsonl").read_text().splitlines()[0])
    assert set(first_log["total_loss"]) == {"A", "B"}
    assert "motion_fallback_presentations" in first_log
    for arm in ("A", "B"):
        for key in full_state["models"][arm]:
            assert torch.equal(full_state["models"][arm][key], resumed_state["models"][arm][key])
        _assert_nested_equal(full_state["optimizers"][arm], resumed_state["optimizers"][arm])


def test_resume_rejects_changed_immutable_training_config(tmp_path: Path) -> None:
    config = {
        "out": str(tmp_path), "weights": "toy", "steps": 4, "batch": 2, "seed": 2,
        "device": "cpu", "amp": False, "checkpoint_every": 1, "val_every": 0,
        "stop_after": 2, "_dataset": _ToyDataset(), "_collate_fn": _collate,
        "_model_factory": _factory,
    }
    train(config)
    original_run_config = (tmp_path / "run_config.json").read_bytes()
    changed = dict(config, resume=True, stop_after=0, batch=3)
    try:
        train(changed)
    except ValueError as exc:
        assert "immutable" in str(exc).lower()
    else:
        raise AssertionError("resume accepted a changed batch size")
    assert (tmp_path / "run_config.json").read_bytes() == original_run_config


def test_fresh_run_rejects_existing_managed_evidence(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text('{"complete": true}\n', encoding="utf-8")
    config = {"out": str(tmp_path), "weights": "toy", "steps": 1, "batch": 2,
              "device": "cpu", "amp": False, "val_every": 0, "_dataset": _ToyDataset(),
              "_collate_fn": _collate, "_model_factory": _factory}
    with pytest.raises(FileExistsError, match="--resume|new --out"):
        train(config)
    assert (tmp_path / "summary.json").read_text(encoding="utf-8") == '{"complete": true}\n'


def test_resume_rejects_changed_dataset_identity(tmp_path: Path) -> None:
    config = {
        "out": str(tmp_path), "weights": "toy", "steps": 3, "batch": 2, "seed": 2,
        "device": "cpu", "amp": False, "checkpoint_every": 1, "val_every": 0,
        "stop_after": 1, "_dataset": _ToyDataset(9), "_collate_fn": _collate,
        "_model_factory": _factory,
    }
    train(config)
    try:
        train(dict(config, resume=True, stop_after=0, _dataset=_ToyDataset(8)))
    except ValueError as exc:
        assert "immutable" in str(exc).lower()
    else:
        raise AssertionError("resume accepted a changed dataset")


def test_resume_deserializes_rng_checkpoint_on_cpu(monkeypatch, tmp_path: Path) -> None:
    config = {"out": str(tmp_path), "weights": "toy", "steps": 2, "batch": 2,
              "device": "cpu", "amp": False, "val_every": 0, "stop_after": 1,
              "_dataset": _ToyDataset(), "_collate_fn": _collate, "_model_factory": _factory}
    train(config)
    original_load = torch.load
    observed = []
    def recording_load(*args, **kwargs):
        if Path(args[0]).name == "last.pt": observed.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", recording_load)
    train(dict(config, resume=True, stop_after=0))
    assert observed and observed[0] == "cpu"


def test_validation_saves_independent_best_checkpoint_for_each_arm(tmp_path: Path) -> None:
    seen: list[str] = []

    def evaluate(model, _dataset, output_dir, **_kwargs):
        seen.append(model.arm)
        return {"map50_95": 0.2 if model.arm == "A" else 0.3}

    report = train({
        "out": str(tmp_path), "weights": "toy", "steps": 2, "batch": 2, "seed": 5,
        "device": "cpu", "amp": False, "checkpoint_every": 1, "val_every": 1,
        "_dataset": _ToyDataset(), "_val_dataset": _ToyDataset(3), "_collate_fn": _collate,
        "_model_factory": _factory, "_eval_fn": evaluate,
    })
    assert seen == ["A", "B", "A", "B"]
    assert (tmp_path / "best_A.pt").is_file()
    assert (tmp_path / "best_B.pt").is_file()
    assert report["best"]["A"]["metric"] == 0.2
    assert report["best"]["B"]["metric"] == 0.3


def test_validation_always_runs_at_final_step_and_accepts_none_metric(tmp_path: Path) -> None:
    seen: list[tuple[str, int]] = []

    def evaluate(model, _dataset, output_dir, **_kwargs):
        step = int(Path(output_dir).name.removeprefix("step_"))
        seen.append((model.arm, step))
        return {"map50_95": None}

    report = train({
        "out": str(tmp_path), "weights": "toy", "steps": 3, "batch": 2, "seed": 5,
        "device": "cpu", "amp": False, "checkpoint_every": 1, "val_every": 2,
        "_dataset": _ToyDataset(), "_val_dataset": _ToyDataset(3), "_collate_fn": _collate,
        "_model_factory": _factory, "_eval_fn": evaluate,
    })
    assert seen == [("A", 2), ("B", 2), ("A", 3), ("B", 3)]
    assert report["best"]["A"]["metric"] == 0.0


def test_default_validation_cadence_is_one_training_pass_and_final(tmp_path: Path) -> None:
    seen = []
    def evaluate(model, _dataset, output_dir, **_kwargs):
        seen.append((model.arm, int(Path(output_dir).name.removeprefix("step_"))))
        return {"map50_95": 0.0}
    train({"out": str(tmp_path), "weights": "toy", "steps": 4, "batch": 2,
           "accumulation": 2, "device": "cpu", "amp": False, "val_every": None,
           "_dataset": _ToyDataset(9), "_val_dataset": _ToyDataset(3),
           "_collate_fn": _collate, "_model_factory": _factory, "_eval_fn": evaluate})
    assert seen == [("A", 3), ("B", 3), ("A", 4), ("B", 4)]


def test_train_rejects_train_validation_sample_overlap(tmp_path: Path) -> None:
    train_dataset = _ToyDataset()
    train_dataset.records = [{"sample_id": "same", "source_sha256": "a", "source_frame_id": 1}]
    val_dataset = _ToyDataset()
    val_dataset.records = [{"sample_id": "same", "source_sha256": "a", "source_frame_id": 1}]
    config = {
        "out": str(tmp_path), "weights": "toy", "steps": 1, "batch": 1, "seed": 5,
        "device": "cpu", "amp": False, "val_every": 1, "_dataset": train_dataset,
        "_val_dataset": val_dataset, "_collate_fn": _collate, "_model_factory": _factory,
        "_eval_fn": lambda *_args, **_kwargs: {"map50_95": 0.0},
    }
    try:
        train(config)
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("accepted overlapping train and validation samples")
