from __future__ import annotations

import json
import pytest

from drone_training.cli import build_parser, main


def test_all_documented_commands_have_functional_help() -> None:
    parser = build_parser()
    expected = {"download", "weights", "train-yolo", "evaluate", "calibrate", "compare", "sam-pseudo",
                "sam-train", "sam-infer", "sam-head-train", "sam-head-infer", "doctor"}
    choices = next(action for action in parser._actions if action.dest == "command").choices
    assert set(choices) == expected
    for command in expected:
        try:
            main([command, "--help"])
        except SystemExit as exc:
            assert exc.code == 0


def test_train_yolo_dispatches_explicit_full_defaults(monkeypatch, tmp_path, capsys) -> None:
    captured = {}

    def fake_train(config):
        captured.update(config)
        return {"completed_step": 1, "complete": False}

    monkeypatch.setattr("drone_training.cli._train_yolo", fake_train)
    code = main(["train-yolo", "--data-root", str(tmp_path / "data"), "--out", str(tmp_path / "run"),
                 "--weights", str(tmp_path / "yolo11m.pt"), "--device", "cpu"])
    assert code == 0
    assert captured["epochs"] == 10
    assert captured["train_limit"] == 0
    assert captured["val_limit"] == 0
    assert captured["val_every"] is None
    assert captured["train_manifest"] == "release_v3_curated/manifests/flow_train_common.jsonl"
    assert captured["val_manifest"] == "release_v3_curated/manifests/flow_validation.jsonl"
    assert json.loads(capsys.readouterr().out)["complete"] is False


def test_download_requires_separate_smoke_flag(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr("drone_training.cli._download", lambda **kwargs: calls.append(kwargs) or {"ok": True})
    assert main(["download", "--data-root", str(tmp_path)]) == 0
    assert calls[-1]["smoke"] is False
    assert main(["download", "--data-root", str(tmp_path), "--smoke"]) == 0
    assert calls[-1]["smoke"] is True


def test_proposal_head_defaults_to_epoch_schedule() -> None:
    args = build_parser().parse_args(["sam-head-train", "--candidate-jsonl", "c.jsonl",
        "--data-root", "data", "--student-checkpoint", "tiny.pt", "--out", "run"])
    assert args.epochs == 5
    assert args.batch == 32
    assert args.steps is None


def test_weights_rejects_unverified_existing_file(monkeypatch, tmp_path) -> None:
    (tmp_path / "model.pt").write_bytes(b"not an official weight")
    monkeypatch.setattr("drone_training.cli.WEIGHT_URLS", {"model.pt": "https://example.invalid/model.pt"})
    monkeypatch.setattr("drone_training.cli.WEIGHT_SHA256", {"model.pt": "0" * 64})
    from drone_training.cli import _weights
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _weights(str(tmp_path))
