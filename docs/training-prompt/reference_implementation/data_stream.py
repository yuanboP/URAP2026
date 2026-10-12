"""Bounded, order-preserving source-video stream backed by MotionCache."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from motion_cache import CODEC_VERSION, MotionCache, build_cache_key, decode_motion_blob, encode_motion_blob


def _video_path(record):
    return Path(record.get("source_video_path", record.get("video_path", ""))).resolve()


def _frame_ids(record):
    ids = record.get("neighbor_frame_ids", record.get("frame_ids"))
    if ids is None or len(ids) != 3: raise ValueError(f"record {record.get('sample_id')} lacks t-1,t,t+1 ids")
    return [int(x) for x in ids]


def _decode_frames(path, ids, base, expected_shape=None, diagnostics=None):
    cv2.setNumThreads(1); cv2.setRNGSeed(17)
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("decode_retry_count", 0); diagnostics.setdefault("decode_retry_details", [])
    def opened_capture():
        value=cv2.VideoCapture(str(path))
        if not value.isOpened():
            value.release(); raise RuntimeError(f"cannot open video: {path}")
        return value
    def read_checked(value, zero, seek, shape):
        if seek and not value.set(cv2.CAP_PROP_POS_FRAMES, zero):
            return None,{"reason":"seek_rejected","actual_frame_id":None}
        ok,frame=value.read(); reported=value.get(cv2.CAP_PROP_POS_FRAMES)
        actual=int(round(reported))-1 if reported is not None else None
        if not ok or frame is None:
            return None,{"reason":"read_failed","actual_frame_id":actual}
        if actual != zero:
            return None,{"reason":"position_mismatch","actual_frame_id":actual}
        if shape is not None and tuple(frame.shape[:2]) != tuple(shape):
            return None,{"reason":"shape_mismatch","actual_frame_id":actual,
                         "actual_shape":list(frame.shape[:2]),"expected_shape":list(shape)}
        return frame,None
    capture = opened_capture(); frames=[]
    try:
        contiguous = all(ids[index] == ids[0] + index for index in range(len(ids)))
        for index, frame_id in enumerate(ids):
            zero = frame_id - int(base)
            if zero < 0: raise ValueError(f"negative resolved frame index: {frame_id} base={base}")
            shape=tuple(expected_shape) if expected_shape is not None else (tuple(frames[0].shape[:2]) if frames else None)
            frame,error=read_checked(capture,zero,index == 0 or not contiguous,shape)
            if error is not None:
                initial=dict(error); capture.release(); capture=None
                failures=[]
                for attempt in range(1,3):
                    fresh=opened_capture(); frame,retry_error=read_checked(fresh,zero,True,shape)
                    if retry_error is None:
                        capture=fresh
                        diagnostics["decode_retry_count"] += 1
                        diagnostics["decode_retry_details"].append({"requested_frame_id":zero,
                            "initial":initial,"successful_attempt":attempt})
                        break
                    failures.append({"attempt":attempt,**retry_error}); fresh.release()
                if frame is None:
                    raise RuntimeError(f"exact decode failed: requested={zero}, initial={initial}, retries={failures}, video={path}")
            frames.append(frame)
    finally:
        if capture is not None: capture.release()
    return frames


def _default_motion(previous, center, following, signature):
    script = Path(signature["flow_script"]).resolve()
    spec = importlib.util.spec_from_file_location("full_stream_flow", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in (previous, center, following)]
    args = type("FlowArgs", (), signature["flow_params"])()
    difference, valid, _ = module.compensated_difference(gray[0], gray[1], gray[2], args, local=True)
    motion2 = np.rint(np.asarray(difference, np.float32) * 2).astype(np.uint16)
    if motion2.size and int(motion2.max()) > 510: raise ValueError("computed D*2 exceeds 510")
    return motion2, np.asarray(valid, np.uint8)


def _worker(record, signature, cache_entry, motion_fn):
    # Contract: neighbor_frame_ids are canonical zero-based decoder indices.
    # source_frame_index_base documents annotation provenance only and is never subtracted here.
    started=time.perf_counter(); path=_video_path(record); ids=_frame_ids(record); base=0
    diagnostics={}; expected=(int(record["source_height"]),int(record["source_width"]))
    if cache_entry is not None:
        center=_decode_frames(path,ids,base,expected,diagnostics)[1]
        blob=Path(cache_entry["path"]).read_bytes(); motion2,valid=decode_motion_blob(blob,cache_entry["checksum"])
        return {"rgb":cv2.cvtColor(center,cv2.COLOR_BGR2RGB),"motion2":motion2,"valid":valid,"blob":None,"checksum":cache_entry["checksum"],"cache_hit":True,"decode_seconds":time.perf_counter()-started,"flow_seconds":0.,**diagnostics}
    frames=_decode_frames(path,ids,base,expected,diagnostics); decoded=time.perf_counter(); fn=motion_fn or _default_motion
    motion2,valid=fn(frames[0],frames[1],frames[2],signature); flowed=time.perf_counter(); blob,checksum=encode_motion_blob(motion2,valid)
    return {"rgb":cv2.cvtColor(frames[1],cv2.COLOR_BGR2RGB),"motion2":None,"valid":None,"blob":blob,"checksum":checksum,"cache_hit":False,"decode_seconds":decoded-started,"flow_seconds":flowed-decoded,**diagnostics}


def _default_view(record, view_kind, aug_seed, size):
    from augment_full import make_view_spec
    return make_view_spec(record, view_kind, aug_seed, size=size)


def _apply_default(rgb,motion,valid,boxes,spec,size):
    from augment_full import apply_view
    return apply_view(rgb,motion,valid,boxes,spec,size=size)


class FullDataStream:
    def __init__(self, records, cache_root, signature, workers=2, prefetch=4, cache_gib=55, min_free_gib=20,
                 *, size=1280, view_fn=None, view_spec_fn=None, motion_fn=None):
        self.records=list(records); self.signature=dict(signature); self.workers=int(workers); self.prefetch=int(prefetch); self.size=int(size)
        if self.workers<1 or self.prefetch<1: raise ValueError("workers and prefetch must be positive")
        for name in ("flow_sha256","params_sha256"):
            if name not in self.signature: raise ValueError(f"signature lacks {name}")
        self.signature.setdefault("rng_seed",17); self.signature.setdefault("codec_version",CODEC_VERSION)
        if self.signature["codec_version"] != CODEC_VERSION: raise ValueError(f"codec_version must be {CODEC_VERSION}")
        self.cache_signature = hashlib.sha256(json.dumps(self.signature,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
        self.cache=MotionCache(cache_root,cache_gib,min_free_gib); self.view_fn=view_fn; self.view_spec_fn=view_spec_fn; self.motion_fn=motion_fn
        self.cursor = 0

    def _key(self, record):
        return build_cache_key(source_sha256=record.get("source_sha256"),frame_ids=_frame_ids(record),flow_sha256=self.signature["flow_sha256"],params_sha256=self.signature["params_sha256"],rng_seed=self.signature["rng_seed"],codec_version=self.signature["codec_version"])

    def iter_presentations(self, plan, cursor=0):
        rows=list(plan); cursor=int(cursor)
        if cursor<0 or cursor>len(rows): raise ValueError("cursor outside plan")
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            pending={}; submit=cursor; emit=cursor
            while emit<len(rows):
                while submit<len(rows) and len(pending)<self.prefetch:
                    row=rows[submit]; record=self.records[int(row["sample_index"])]; key=self._key(record); entry=self.cache.lookup(key)
                    serial=None if entry is None else {"path":str(entry["path"]),"checksum":entry["checksum"]}
                    pending[submit]=(pool.submit(_worker,record,self.signature,serial,self.motion_fn),row,record,key,serial)
                    submit+=1
                future,row,record,key,serial=pending.pop(emit)
                result=future.result()
                cache_bypass = None
                if result["blob"] is not None:
                    result["motion2"],result["valid"]=decode_motion_blob(result["blob"],result["checksum"])
                    protected=[item[4]["path"] for item in pending.values() if item[4] is not None]
                    try:
                        self.cache.put_blob(key,result["blob"],result["checksum"],result["motion2"].shape,json.dumps(self.signature,sort_keys=True),protected=protected)
                    except OSError as exc:
                        # Cache is an optimization. Preserve the computed sample and surface bypass diagnostics.
                        self.cache.record_bypass(); cache_bypass=f"{type(exc).__name__}: {exc}"
                motion=result["motion2"].astype(np.float32)/510.0
                boxes=np.asarray(record.get("boxes_xyxy",record.get("bbox_xyxy",[])),np.float32).reshape(-1,4)
                if self.view_fn is not None:
                    spec=(self.view_spec_fn or (lambda r,k,s,size: {"view_kind":k,"aug_seed":s}))(record,int(row["view_kind"]),int(row["aug_seed"]),self.size)
                    applied=self.view_fn(result["rgb"],motion,result["valid"],boxes,spec,size=self.size)
                else:
                    spec=_default_view(record,int(row["view_kind"]),int(row["aug_seed"]),self.size)
                    applied=_apply_default(result["rgb"],motion,result["valid"],boxes,spec,self.size)
                if isinstance(applied, dict):
                    image,out_boxes,classes,meta=applied["img"],applied["bboxes"],applied["cls"],applied.get("metadata",{})
                    output_valid=applied.get("valid")
                else:
                    image,out_boxes,classes,meta=applied
                    output_valid=None
                image=np.asarray(image,np.float32)
                if image.shape!=(4,self.size,self.size): raise ValueError(f"view returned image shape {image.shape}")
                self.cursor = emit + 1
                yield {"img":image,"bboxes":np.asarray(out_boxes,np.float32).reshape(-1,4),"cls":np.asarray(classes,np.float32).reshape(-1,1),"valid":None if output_valid is None else np.asarray(output_valid,bool),"sample_id":record["sample_id"],"role":record["role"],"view_meta":{**dict(meta),"cache_hit":result["cache_hit"],"cache_bypass":cache_bypass,"decode_retry_count":result["decode_retry_count"],"decode_retry_details":result["decode_retry_details"]},"decode_seconds":result["decode_seconds"],"flow_seconds":result["flow_seconds"],"decode_retry_count":result["decode_retry_count"],"decode_retry_details":result["decode_retry_details"]}
                emit+=1

    def cache_stats(self):
        return self.cache.cache_stats()

    def state_dict(self):
        return {"cursor": self.cursor, "record_count": len(self.records), "workers": self.workers,
                "prefetch": self.prefetch, "size": self.size, "cache_signature": self.cache_signature,
                "cache_stats": self.cache_stats()}
