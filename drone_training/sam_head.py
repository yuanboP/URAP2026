"""A proposal-conditioned drone classifier and box refiner on frozen SAM2 features."""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right-left) * max(0.0, bottom-top)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(union, 1e-9)


def _box_delta(proposal, target):
    px, py = (proposal[0]+proposal[2])/2, (proposal[1]+proposal[3])/2
    pw, ph = proposal[2]-proposal[0], proposal[3]-proposal[1]
    tx, ty = (target[0]+target[2])/2, (target[1]+target[3])/2
    tw, th = target[2]-target[0], target[3]-target[1]
    return [(tx-px)/pw, (ty-py)/ph, math.log(tw/pw), math.log(th/ph)]


def _apply_delta(box, delta):
    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2; w, h = box[2]-box[0], box[3]-box[1]
    cx, cy = x + delta[0]*w, y + delta[1]*h
    nw, nh = w*math.exp(float(np.clip(delta[2], -4, 4))), h*math.exp(float(np.clip(delta[3], -4, 4)))
    return [cx-nw/2, cy-nh/2, cx+nw/2, cy+nh/2]


def _clip_box(box, image_hw):
    height, width = map(float, image_hw)
    values = np.asarray(box, dtype=np.float64)
    if values.shape != (4,) or not np.isfinite(values).all(): return None
    x1,y1,x2,y2 = values
    clipped=[max(0.0,min(width,float(x1))),max(0.0,min(height,float(y1))),
             max(0.0,min(width,float(x2))),max(0.0,min(height,float(y2)))]
    return clipped if clipped[0] < clipped[2] and clipped[1] < clipped[3] else None


def build_proposal_supervision(rows, positive_iou=0.5, negative_iou=0.3):
    """Label proposals conservatively; ambiguous and incompletely annotated misses are ignored."""
    samples = []
    for row in rows:
        if row.get("split") != "train":
            continue
        truths = [[float(v) for v in box] for box in row.get("gt_boxes", [])]
        certified = row.get("boxes_complete_for_frame") is True or row.get("role") == "negative_certified"
        for prediction in row.get("predictions", []):
            proposal = [float(v) for v in prediction[:4]]
            overlaps = [_iou(proposal, truth) for truth in truths]
            best = int(np.argmax(overlaps)) if overlaps else -1
            maximum = overlaps[best] if overlaps else 0.0
            common = {"row": row, "proposal": proposal, "proposal_score": float(prediction[4]) if len(prediction) > 4 else 0.0}
            if maximum >= positive_iou:
                target = truths[best]
                samples.append({**common, "label": 1, "target_box": target, "box_delta": _box_delta(proposal, target)})
            elif maximum < negative_iou and certified:
                samples.append({**common, "label": 0, "target_box": None, "box_delta": [0.0]*4})
    return samples


class ProposalHead(nn.Module):
    def __init__(self, feature_dim, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(feature_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.classifier = nn.Linear(hidden, 1); self.regressor = nn.Linear(hidden, 4)

    def forward(self, features):
        value = self.backbone(features)
        return self.classifier(value).squeeze(-1), self.regressor(value)


def _decode_row(row, data_root):
    path = Path(row["source_video_path"])
    if not path.is_absolute(): path = Path(data_root) / path
    capture = cv2.VideoCapture(str(path))
    try:
        requested=int(row["source_frame_id"])
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES,requested):
            raise RuntimeError(f"cannot seek {path} frame {requested}")
        ok, bgr = capture.read(); actual=int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))-1
        if not ok or bgr is None or actual != requested:
            raise RuntimeError(f"exact frame decode failed for {path}: requested={requested} actual={actual}")
        expected=(row.get("height"),row.get("width"))
        if all(value is not None for value in expected) and tuple(bgr.shape[:2]) != tuple(map(int,expected)):
            raise RuntimeError(f"decoded frame shape {bgr.shape[:2]} does not match metadata {expected}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


class _OfficialFeatureProvider:
    def __init__(self, checkpoint, config, device):
        from sam2.build_sam import build_sam2
        self.device = device; self.model = build_sam2(config, str(checkpoint), device=device, mode="eval").eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)

    @torch.no_grad()
    def batch(self, image, proposals):
        import torch.nn.functional as F
        rgb = np.asarray(image); h, w = rgb.shape[:2]; side = int(self.model.image_size)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().to(self.device)[None] / 255
        tensor = F.interpolate(tensor, (side, side), mode="bilinear", align_corners=False)
        mean = torch.tensor([.485,.456,.406], device=self.device)[None,:,None,None]
        std = torch.tensor([.229,.224,.225], device=self.device)[None,:,None,None]
        backbone = self.model.forward_image((tensor-mean)/std)
        _, features, _, sizes = self.model._prepare_backbone_features(backbone)
        fmap = features[-1].permute(1,2,0).reshape(1,-1,*sizes[-1])
        output=[]
        for proposal in proposals:
            fh, fw = fmap.shape[-2:]; x1,y1,x2,y2 = proposal
            ix1, iy1 = int(np.floor(x1/w*fw)), int(np.floor(y1/h*fh))
            ix2, iy2 = int(np.ceil(x2/w*fw)), int(np.ceil(y2/h*fh))
            roi = fmap[..., max(0,iy1):max(iy1+1,min(fh,iy2)), max(0,ix1):max(ix1+1,min(fw,ix2))]
            pooled = roi.mean((-2,-1)).squeeze(0)
            geometry = torch.tensor([x1/w,y1/h,x2/w,y2/h,(x2-x1)*(y2-y1)/(w*h)], device=self.device)
            output.append(torch.cat((pooled, geometry)).cpu())
        return torch.stack(output)

    def __call__(self, image, proposal):
        return self.batch(image, [proposal])[0]


def _atomic_save(payload, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp"); torch.save(payload, temporary); temporary.replace(path)


def train_head(config):
    config = dict(config); candidate = Path(config["candidate_jsonl"]).resolve()
    rows = [json.loads(line) for line in candidate.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = build_proposal_supervision(rows, config.get("positive_iou", .5), config.get("negative_iou", .3))
    positives = sum(x["label"] for x in samples); negatives = len(samples)-positives
    if not positives or not negatives:
        raise ValueError(f"proposal head needs both matched positives and certified negatives; got positives={positives} negatives={negatives}")
    device = config.get("device", "cuda"); base = Path(config["student_checkpoint"]).resolve()
    output = Path(config["out"]).resolve(); feature_dir = output / "feature_cache"; feature_dir.mkdir(parents=True, exist_ok=True)
    provider = config.get("_feature_provider")
    if provider is None:
        provider = _OfficialFeatureProvider(base, config.get("student_config", "configs/sam2.1/sam2.1_hiera_t.yaml"), device)
        provider_input = lambda row: _decode_row(row, config["data_root"])
    else: provider_input = lambda row: row
    feature_locations = [None] * len(samples); groups = {}
    for index, sample in enumerate(samples):
        key = str(sample["row"].get("sample_id") or (sample["row"].get("source_sha256"), sample["row"].get("source_frame_id")))
        groups.setdefault(key, []).append((index, sample))
    extracted = 0; base_sha = _file_sha(base); code_sha = _file_sha(__file__)
    student_config=config.get("student_config", "configs/sam2.1/sam2.1_hiera_t.yaml")
    for group in groups.values():
        row = group[0][1]["row"]; proposals = [item[1]["proposal"] for item in group]
        cache_identity = json.dumps({"sample_id":row.get("sample_id"), "source_sha256":row.get("source_sha256"),
                                     "source_frame_id":row.get("source_frame_id"), "proposals":proposals,
                                     "student_checkpoint_sha256":base_sha,"student_config":student_config,
                                     "feature_pipeline_sha256":code_sha}, sort_keys=True, separators=(",",":"))
        cache_path = feature_dir / (hashlib.sha256(cache_identity.encode()).hexdigest() + ".pt")
        if cache_path.is_file():
            batch_features = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            if hasattr(provider, "batch"):
                batch_features = provider.batch(provider_input(row), proposals)
            else:
                batch_features = [provider(provider_input(row), proposal) for proposal in proposals]
            batch_features = torch.stack([torch.as_tensor(value,dtype=torch.float32).cpu() for value in batch_features])
            _atomic_save(batch_features, cache_path)
        if len(batch_features) != len(group): raise RuntimeError(f"feature cache count mismatch: {cache_path}")
        for offset, (index, _sample) in enumerate(group): feature_locations[index] = (cache_path, offset)
        extracted += 1
        progress={"frames_cached":extracted,"frames_total":len(groups),"proposals_total":len(samples)}
        progress_path=output/"feature_progress.json"; temporary=progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(progress,sort_keys=True,indent=2),encoding="utf-8"); temporary.replace(progress_path)
    first_cache=torch.load(feature_locations[0][0],map_location="cpu",weights_only=True)
    feature_dim=int(first_cache.shape[1])
    labels = torch.tensor([x["label"] for x in samples], dtype=torch.float32)
    deltas = torch.tensor([x["box_delta"] for x in samples], dtype=torch.float32)
    hidden = int(config.get("hidden", 128)); learning_rate = float(config.get("lr", 1e-3))
    identity = {"candidate_sha256": _file_sha(candidate), "student_checkpoint_sha256": base_sha,
                "feature_dim": feature_dim, "hidden": hidden, "positive_iou": config.get("positive_iou",.5),
                "negative_iou": config.get("negative_iou",.3), "learning_rate": learning_rate,
                "seed": int(config.get("seed", 17)), "batch_size":int(config.get("batch", 32)),
                "student_config":student_config,"feature_pipeline_sha256":code_sha}
    torch.manual_seed(identity["seed"])
    model = ProposalHead(feature_dim, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    checkpoint = output / "last.pt"; start_step = 0
    if config.get("resume"):
        if not checkpoint.is_file(): raise FileNotFoundError(f"resume checkpoint not found: {checkpoint}")
        resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if resumed.get("identity") != identity: raise ValueError("resume rejected because immutable proposal-head configuration changed")
        model.load_state_dict(resumed["model"], strict=True); optimizer.load_state_dict(resumed["optimizer"])
        torch.set_rng_state(resumed["torch_rng_state"]); start_step = int(resumed["completed_step"])
    initial = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    batches_per_epoch=math.ceil(len(samples)/identity["batch_size"])
    steps = int(config["steps"]) if config.get("steps") is not None else batches_per_epoch*int(config.get("epochs",5))
    stop_after = int(config.get("stop_after", 0)); checkpoint_every=max(1,int(config.get("checkpoint_every",100)))
    end_step = min(steps, stop_after) if stop_after else steps
    if end_step < start_step: raise ValueError("stop_after is behind the resumed checkpoint cursor")
    model.train()
    loaded=OrderedDict()
    def load_features(indices):
        values=[]
        for index in indices.tolist():
            path,offset=feature_locations[index]
            if path not in loaded:
                loaded[path]=torch.load(path,map_location="cpu",weights_only=True)
                if len(loaded)>8: loaded.popitem(last=False)
            values.append(loaded[path][offset])
        return torch.stack(values)
    def save_head(completed):
        _atomic_save({"state_format":"urap.sam_proposal_head.v1", "model":model.state_dict(), "identity":identity,
                      "optimizer":optimizer.state_dict(), "torch_rng_state":torch.get_rng_state(),
                      "completed_step":completed, "planned_steps":steps,
                      "unique_proposals_seen":min(len(samples),completed*identity["batch_size"]),
                      "positive_count":positives, "negative_count":negatives,
                      "feature_storage_device":"disk_cpu_tensors", "batch_size":identity["batch_size"]}, checkpoint)
    for step in range(start_step, end_step):
        indices = (torch.arange(identity["batch_size"]) + step * identity["batch_size"]) % len(samples)
        batch_features=load_features(indices).to(device); batch_labels=labels[indices].to(device); batch_deltas=deltas[indices].to(device)
        logits, predicted = model(batch_features)
        classification = nn.functional.binary_cross_entropy_with_logits(logits, batch_labels)
        positive=batch_labels > .5
        regression = nn.functional.smooth_l1_loss(predicted[positive], batch_deltas[positive]) if positive.any() else predicted.sum()*0
        loss = classification + regression; optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (step+1)%checkpoint_every==0: save_head(step+1)
    changed = any(not torch.equal(initial[k], v.detach().cpu()) for k,v in model.state_dict().items())
    if end_step > start_step and not changed: raise RuntimeError("proposal head optimizer did not change model weights")
    save_head(end_step)
    return {"checkpoint":str(checkpoint), "completed_step":end_step, "planned_steps":steps,
            "unique_proposals_seen":min(len(samples),end_step*identity["batch_size"]),
            "positive_count":positives, "negative_count":negatives}


def infer_head(image, proposals, checkpoint, *, student_checkpoint, student_config="configs/sam2.1/sam2.1_hiera_t.yaml",
               device="cuda", feature_provider=None):
    if not proposals: return []
    state = torch.load(checkpoint, map_location="cpu", weights_only=False); identity = state["identity"]
    if state.get("state_format") != "urap.sam_proposal_head.v1": raise ValueError("unsupported proposal head checkpoint")
    if identity["student_checkpoint_sha256"] != _file_sha(student_checkpoint):
        raise ValueError("proposal head belongs to a different SAM2 base checkpoint")
    provider = feature_provider or _OfficialFeatureProvider(student_checkpoint, student_config, device)
    values = provider.batch(image, proposals) if hasattr(provider, "batch") else [provider(image, box) for box in proposals]
    features = torch.stack([torch.as_tensor(value, dtype=torch.float32) for value in values]).to(device)
    model = ProposalHead(identity["feature_dim"], identity["hidden"]).to(device)
    model.load_state_dict(state["model"], strict=True); model.eval()
    with torch.no_grad(): logits, deltas = model(features)
    output=[]
    for box,logit,delta in zip(proposals,logits,deltas):
        refined=_clip_box(_apply_delta(box,delta.cpu().tolist()),np.asarray(image).shape[:2])
        if refined is not None:
            output.append({"proposal_xyxy":list(map(float,box)),"drone_score":float(torch.sigmoid(logit)),
                           "refined_xyxy":refined,"score_semantics":"proposal_drone_probability"})
    return output
