from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from drone_training.sam import ADAPTER_GRAPH_VERSION, _reference as load_sam_reference, _student_forward
from drone_training.sam import infer_sam, pseudo, train_sam
from drone_training.sam_head import _clip_box, build_proposal_supervision, infer_head, train_head


def _canonical_row(root: Path, *, split: str = "train") -> dict:
    digest = "a" * 64
    (root / "videos").mkdir(parents=True, exist_ok=True)
    (root / "videos" / f"{digest}.mp4").write_bytes(b"video")
    return {
        "sample_id": f"sample-{split}", "source_sha256": digest,
        "source_video_path": "../../videos/" + f"{digest}.mp4",
        "source_video_id": digest, "sequence_id": digest,
        "source_frame_id": 0, "video_frame_count": 1,
        "source_width": 16, "source_height": 12,
        "bbox_xyxy": [[1, 2, 8, 9]], "class_id": [0], "split": split,
    }


class _Reference:
    TINY_CONFIG = "tiny.yaml"
    LARGE_CONFIG = "large.yaml"

    def __init__(self):
        self.records = None
        self.train_args = None

    def normalize_manifest_rows(self, rows, manifest_dir):
        self.records = rows
        return [{"split": r["split"], "image_ref": r["source_video_path"], "xyxy": r["bbox_xyxy"][0]} for r in rows]

    def select_training_records(self, records, maximum, order):
        return records[: maximum or None]

    def generate_pseudo_masks(self, records, *args, **kwargs):
        self.records = records
        return [{"status": "accepted"} for _ in records]

    def train_decoder(self, *args, **kwargs):
        self.train_args = (args, kwargs)
        return {"steps_completed": 3}

    def make_training_state(self, *args, **kwargs):
        return dict(kwargs)


def test_pseudo_resolves_canonical_video_from_explicit_data_root_and_uses_train_only(tmp_path: Path) -> None:
    data_root = tmp_path / "release"
    empty = _canonical_row(data_root); empty["sample_id"]="empty-train"; empty["bbox_xyxy"]=[]; empty["class_id"]=[]
    rows = [_canonical_row(data_root), empty, _canonical_row(data_root, split="validation")]
    manifest = data_root / "metadata" / "manifests" / "sam.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    teacher = tmp_path / "large.pt"; teacher.write_bytes(b"teacher")
    reference = _Reference()

    result = pseudo({"data_root": data_root, "manifest": manifest, "out": tmp_path / "pseudo",
                     "teacher_checkpoint": teacher, "device": "cpu", "_reference": reference})

    assert result["selected"] == result["accepted"] == 1
    assert Path(reference.records[0]["image_ref"]) == data_root / "videos" / ("a" * 64 + ".mp4")


def test_pseudo_rejects_an_empty_train_pool_with_actionable_message(tmp_path: Path) -> None:
    data_root = tmp_path / "release"
    row = _canonical_row(data_root, split="validation")
    manifest = data_root / "sam.jsonl"; manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    teacher = tmp_path / "large.pt"; teacher.write_bytes(b"teacher")

    with pytest.raises(ValueError, match="train.*manifest|manifest.*train"):
        pseudo({"data_root": data_root, "manifest": manifest, "out": tmp_path / "pseudo",
                "teacher_checkpoint": teacher, "_reference": _Reference()})


def test_train_sam_forwards_resume_and_rejects_changed_checkpoint_identity(tmp_path: Path) -> None:
    pseudo_manifest = tmp_path / "pseudo.jsonl"
    pseudo_manifest.write_text(json.dumps({"status": "accepted", "split": "train"}) + "\n", encoding="utf-8")
    student = tmp_path / "tiny.pt"; student.write_bytes(b"tiny-v1")
    resume = tmp_path / "resume.pt"
    torch.save({"student_checkpoint_sha256": "0" * 64}, resume)

    with pytest.raises(ValueError, match="different student checkpoint"):
        train_sam({"pseudo_manifest": pseudo_manifest, "out": tmp_path / "model",
                   "student_checkpoint": student, "resume": resume, "_reference": _Reference()})

    identity = {"student_checkpoint_sha256": hashlib.sha256(student.read_bytes()).hexdigest(),
                "pseudo_manifest_sha256": hashlib.sha256(pseudo_manifest.read_bytes()).hexdigest(),
                "student_config": "tiny.yaml", "learning_rate": 1e-5,
                "adapter_graph_version":ADAPTER_GRAPH_VERSION,
                "adapter_source_sha256":hashlib.sha256((Path(__file__).parents[1]/"drone_training"/"sam.py").read_bytes()).hexdigest()}
    torch.save({"student_checkpoint_sha256": identity["student_checkpoint_sha256"], "adapter_identity": identity}, resume)
    reference = _Reference()
    result = train_sam({"pseudo_manifest": pseudo_manifest, "out": tmp_path / "model",
                        "student_checkpoint": student, "resume": resume, "_reference": reference})
    assert result["steps_completed"] == 3
    assert reference.train_args[1]["resume"] == str(resume.resolve())


def test_train_sam_stop_after_bounds_smoke_run_to_exact_number_of_rows(tmp_path: Path) -> None:
    pseudo_manifest = tmp_path / "pseudo.jsonl"
    pseudo_manifest.write_text("".join(json.dumps({"status": "accepted", "split": "train", "pseudo_cache_key": str(i)}) + "\n" for i in range(3)), encoding="utf-8")
    student = tmp_path / "tiny.pt"; student.write_bytes(b"tiny")
    reference = _Reference()
    train_sam({"pseudo_manifest": pseudo_manifest, "out": tmp_path / "model", "student_checkpoint": student,
               "stop_after": 2, "epochs": 8, "_reference": reference})
    assert Path(reference.train_args[0][0]) == pseudo_manifest.resolve()
    assert reference.train_args[1]["epochs"] == 8


def test_resume_rejects_checkpoint_missing_wrapper_identity(tmp_path: Path) -> None:
    pseudo_manifest = tmp_path / "pseudo.jsonl"
    pseudo_manifest.write_text(json.dumps({"status": "accepted", "split": "train"}) + "\n", encoding="utf-8")
    student = tmp_path / "tiny.pt"; student.write_bytes(b"tiny")
    resume = tmp_path / "old.pt"
    torch.save({"student_checkpoint_sha256": hashlib.sha256(student.read_bytes()).hexdigest()}, resume)
    with pytest.raises(ValueError, match="identity"):
        train_sam({"pseudo_manifest": pseudo_manifest, "out": tmp_path / "out", "student_checkpoint": student,
                   "resume": resume, "_reference": _Reference()})


def test_completed_resume_returns_without_entering_reference_loop(tmp_path: Path) -> None:
    pseudo_manifest=tmp_path/"pseudo.jsonl"; pseudo_manifest.write_text(json.dumps({"status":"accepted","split":"train"})+"\n")
    student=tmp_path/"tiny.pt"; student.write_bytes(b"tiny")
    from drone_training import sam as module
    identity={"student_checkpoint_sha256":hashlib.sha256(student.read_bytes()).hexdigest(),
              "pseudo_manifest_sha256":hashlib.sha256(pseudo_manifest.read_bytes()).hexdigest(),
              "student_config":"tiny.yaml","learning_rate":1e-5,"adapter_graph_version":ADAPTER_GRAPH_VERSION,
              "adapter_source_sha256":hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    resume=tmp_path/"done.pt"; torch.save({"student_checkpoint_sha256":identity["student_checkpoint_sha256"],
        "pseudo_manifest_sha256":identity["pseudo_manifest_sha256"],"adapter_identity":identity,"epoch":1,
        "next_step":0,"global_step":4,"elapsed_seconds":2.0,"best_training_loss":.2,"teacher_pseudo_coverage":{}},resume)
    reference=_Reference()
    result=train_sam({"pseudo_manifest":pseudo_manifest,"out":tmp_path/"out","student_checkpoint":student,
                      "resume":resume,"epochs":1,"_reference":reference})
    assert result["steps_completed"] == 4 and result["already_complete"] is True
    assert reference.train_args is None


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__(); self.weight = nn.Parameter(torch.tensor([0.0]))


class _Model:
    def __init__(self): self.sam_mask_decoder = _Decoder()
    def eval(self): return self


class _Predictor:
    def __init__(self, model): self.model = model; self.image = None
    def set_image(self, image): self.image = image
    def predict(self, *, box, multimask_output):
        value = float(self.model.sam_mask_decoder.weight.item())
        return np.full((1, *self.image.shape[:2]), value > 0), np.asarray([0.8]), np.asarray([value])


def test_training_forward_adds_frozen_no_memory_embedding_like_official_predictor() -> None:
    class Model:
        image_size=1024; sam_image_embedding_size=64; directly_add_no_mem_embed=True
        no_mem_embed=torch.tensor([[[5.0]]])
        def forward_image(self, image): return image
        def _prepare_backbone_features(self, _backbone):
            sizes=[(256,256),(128,128),(64,64)]
            return None,[torch.zeros(h*w,1,1) for h,w in sizes],None,sizes
        def _forward_sam_heads(self,image_embed,prompts,**kwargs):
            self.captured=image_embed
            return None,torch.zeros(1,1,4,4,requires_grad=True),torch.zeros(1,1),None
    model=Model()
    _student_forward(model,torch.zeros(1,3,1024,1024),[1,1,5,5])
    assert torch.all(model.captured == 5)
    assert model.no_mem_embed.grad is None


def test_infer_sam_reloads_trained_decoder_and_requires_external_proposals(tmp_path: Path) -> None:
    base = tmp_path / "tiny.pt"; base.write_bytes(b"base")
    trained = tmp_path / "trained.pt"
    torch.save({"state_format": "urap.sam_decoder_checkpoint.v1", "weights_scope": "sam_mask_decoder",
                "model": {"weight": torch.tensor([2.0])},
                "student_checkpoint_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "adapter_identity":{"adapter_graph_version":ADAPTER_GRAPH_VERSION}}, trained)
    built = []
    def build(_config, _checkpoint, **_kwargs):
        model = _Model(); built.append(model); return model

    image = np.zeros((6, 8, 3), dtype=np.uint8)
    results = infer_sam({"image": image, "boxes_xyxy": [[1, 1, 5, 5]], "checkpoint": trained,
                         "student_checkpoint": base, "device": "cpu", "_model_builder": build,
                         "_predictor_factory": _Predictor})
    assert built[0].sam_mask_decoder.weight.item() == 2.0
    assert results[0]["mask"].shape == (6, 8)
    assert results[0]["score_semantics"] == "sam_predicted_mask_quality_not_drone_confidence"
    with pytest.raises(ValueError, match="proposal"):
        infer_sam({"image": image, "boxes_xyxy": [], "checkpoint": trained,
                   "student_checkpoint": base, "_model_builder": build, "_predictor_factory": _Predictor})
    old=tmp_path/"old_graph.pt"; old_state=torch.load(trained,map_location="cpu",weights_only=False)
    old_state["adapter_identity"]={}; torch.save(old_state,old)
    with pytest.raises(ValueError,match="graph"):
        infer_sam({"image":image,"boxes_xyxy":[[1,1,5,5]],"checkpoint":old,
                   "student_checkpoint":base,"_model_builder":build,"_predictor_factory":_Predictor})


def test_infer_accepts_exact_checkpoint_structure_emitted_by_reference(tmp_path: Path) -> None:
    reference = load_sam_reference()
    source_model = _Model(); source_model.sam_mask_decoder.weight.data.fill_(3.0)
    optimizer = torch.optim.AdamW(source_model.sam_mask_decoder.parameters())
    base = tmp_path / "tiny.pt"; base.write_bytes(b"official-base")
    state = reference.make_training_state(source_model, optimizer, epoch=1, next_step=0, global_step=1,
                                          elapsed_seconds=0.1, student_checkpoint_sha256=hashlib.sha256(base.read_bytes()).hexdigest())
    state["adapter_identity"]={"adapter_graph_version":ADAPTER_GRAPH_VERSION}
    checkpoint = tmp_path / "trained.pt"; torch.save(state, checkpoint)
    assert list(state["model"]) == ["weight"]  # decoder-local keys, without sam_mask_decoder prefix
    built=[]
    def build(*_args, **_kwargs): model=_Model(); built.append(model); return model
    infer_sam({"image":np.zeros((5,5,3),np.uint8), "boxes_xyxy":[[0,0,4,4]], "checkpoint":checkpoint,
               "student_checkpoint":base, "device":"cpu", "_model_builder":build, "_predictor_factory":_Predictor})
    assert built[0].sam_mask_decoder.weight.item() == 3.0


def test_head_supervision_ignores_unmatched_unknown_frames_and_certifies_negatives() -> None:
    unknown = {"split":"train", "gt_boxes": [[0, 0, 10, 10]], "predictions": [[1, 1, 9, 9, .8, 0], [20, 20, 30, 30, .4, 0]],
               "boxes_complete_for_frame": False, "role": "positive"}
    complete = {"split":"train", "gt_boxes": [], "predictions": [[2, 2, 5, 5, .2, 0]],
                "boxes_complete_for_frame": True, "role": "negative_certified"}
    samples = build_proposal_supervision([unknown, complete])
    assert [sample["label"] for sample in samples] == [1, 0]
    assert samples[0]["target_box"] == [0.0, 0.0, 10.0, 10.0]
    assert build_proposal_supervision([{**complete, "split": None}]) == []


def test_refined_boxes_are_finite_clipped_and_nondegenerate() -> None:
    assert _clip_box([-5, 2, 30, 12], (10, 20)) == [0.0, 2.0, 20.0, 10.0]
    assert _clip_box([float("nan"), 0, 1, 1], (10, 20)) is None
    assert _clip_box([4, 4, 4, 8], (10, 20)) is None


def test_head_trains_changed_reloadable_weights_and_infers_without_ground_truth(tmp_path: Path) -> None:
    candidate = tmp_path / "candidates.jsonl"
    rows = [
        {"sample_id": "p", "source_sha256": "a" * 64, "source_frame_id": 0, "height": 20, "width": 20,
         "source_video_path": "videos/a.mp4", "split": "train", "role": "positive", "boxes_complete_for_frame": False,
         "gt_boxes": [[1, 1, 8, 8]], "predictions": [[1, 1, 8, 8, .8, 0]]},
        {"sample_id": "n", "source_sha256": "b" * 64, "source_frame_id": 0, "height": 20, "width": 20,
         "source_video_path": "videos/b.mp4", "split": "train", "role": "negative_certified", "boxes_complete_for_frame": True,
         "gt_boxes": [], "predictions": [[10, 10, 16, 16, .3, 0]]},
    ]
    candidate.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    base = tmp_path / "tiny.pt"; base.write_bytes(b"base")
    def features(_row, proposal):
        return torch.tensor([proposal[0] / 20, proposal[1] / 20, 1.0, 0.0])
    result = train_head({"candidate_jsonl": candidate, "out": tmp_path / "head", "student_checkpoint": base,
                         "steps": 4, "lr": .05, "device": "cpu", "seed": 3, "_feature_provider": features})
    state = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
    assert result["completed_step"] == 4 and result["positive_count"] == result["negative_count"] == 1
    assert any(torch.count_nonzero(value).item() for value in state["model"].values())
    output = infer_head(np.zeros((20, 20, 3), np.uint8), [[1, 1, 8, 8]], result["checkpoint"],
                        student_checkpoint=base, device="cpu", feature_provider=lambda _image, box: features({}, box))
    assert len(output) == 1 and 0 <= output[0]["drone_score"] <= 1
    assert "gt_boxes" not in output[0]


def test_head_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    candidate = tmp_path / "candidates.jsonl"
    rows = [{"sample_id":"p", "split":"train", "role":"positive", "boxes_complete_for_frame":False,
             "gt_boxes":[[0,0,5,5]], "predictions":[[0,0,5,5,.9,0]]},
            {"sample_id":"n", "split":"train", "role":"negative_certified", "boxes_complete_for_frame":True,
             "gt_boxes":[], "predictions":[[8,8,12,12,.2,0]]}]
    candidate.write_text("".join(json.dumps(x)+"\n" for x in rows), encoding="utf-8")
    base=tmp_path/"tiny.pt"; base.write_bytes(b"base")
    provider=lambda _row, box: torch.tensor([box[0],box[1],box[2],box[3]],dtype=torch.float32)/12
    common={"candidate_jsonl":candidate,"student_checkpoint":base,"steps":6,"lr":.02,"device":"cpu","seed":9,"_feature_provider":provider}
    full=train_head(dict(common,out=tmp_path/"full"))
    partial=train_head(dict(common,out=tmp_path/"split",stop_after=3))
    assert partial["completed_step"] == 3
    resumed=train_head(dict(common,out=tmp_path/"split",resume=True))
    a=torch.load(full["checkpoint"],map_location="cpu",weights_only=False)["model"]
    b=torch.load(resumed["checkpoint"],map_location="cpu",weights_only=False)["model"]
    assert all(torch.equal(a[key],b[key]) for key in a)


def test_head_batches_proposals_from_same_frame_and_keeps_feature_bank_on_cpu(tmp_path: Path) -> None:
    candidate=tmp_path/"candidates.jsonl"
    row={"sample_id":"frame", "split":"train", "role":"negative_certified", "boxes_complete_for_frame":True,
         "gt_boxes":[[0,0,5,5]], "predictions":[[0,0,5,5,.9,0],[8,8,12,12,.2,0]]}
    candidate.write_text(json.dumps(row)+"\n",encoding="utf-8")
    base=tmp_path/"tiny.pt"; base.write_bytes(b"base")
    class Provider:
        def __init__(self): self.calls=0
        def batch(self, _row, proposals):
            self.calls += 1
            return torch.tensor(proposals,dtype=torch.float32)/12
    provider=Provider()
    result=train_head({"candidate_jsonl":candidate,"out":tmp_path/"out","student_checkpoint":base,
                       "steps":2,"batch":1,"device":"cpu","_feature_provider":provider})
    state=torch.load(result["checkpoint"],map_location="cpu",weights_only=False)
    assert provider.calls == 1
    assert state["feature_storage_device"] == "disk_cpu_tensors"
    assert state["batch_size"] == 1
    train_head({"candidate_jsonl":candidate,"out":tmp_path/"out","student_checkpoint":base,
                "steps":3,"batch":1,"device":"cpu","resume":True,"_feature_provider":provider})
    assert provider.calls == 1
