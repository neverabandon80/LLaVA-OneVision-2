"""
VSI-590k Packed WebDataset Visualizer
======================================

Lazy-loads packed-sample tars from
  /mnt/vast/data/llava_onevision_2_data/final_round/vsi590k_image_video_merge_rope_packed/webdataset

Each packed sample (`ps_XXXXXXXX.json`) bundles N sub-samples:
  - images:           List[List[str]]      # [sub_idx][frame_idx] -> "img{i}_sub{j}.jpg"
  - prompts:          List[List[str]]      # [sub_idx][turn_idx]  -> user message
  - captions:         List[List[str]]      # [sub_idx][turn_idx]  -> assistant message
  - sample_count:     int                  # number of sub-samples
  - patch_positions:  List[List[str]]      # per-sub-sample, .npy filename per frame ('' = reuse prev)
  - fps:              List[float|int]      # per sub-sample
  - timestamp_decimal:List[int|None]       # per sub-sample (1 or 2)

The corresponding tar member naming:
  - JSON:   ps_XXXXXXXX.json
  - Frames: ps_XXXXXXXX.img{sub}_{frame}.jpg
  - Patch:  ps_XXXXXXXX.img{sub}_{frame}.npy   (only present where patch_positions[sub][frame] != '')

Run:
  python tools/vsi_viz/server.py --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


DEFAULT_ROOT = "/mnt/vast/data/llava_onevision_2_data/final_round/vsi590k_image_video_merge_rope_packed/webdataset"


class TarShard:
    """Lazy index of a single shard. Holds a per-shard lock + open file handle."""

    __slots__ = ("path", "_lock", "_fh", "_tar", "_members", "_packed_keys")

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._fh: Optional[object] = None
        self._tar: Optional[tarfile.TarFile] = None
        self._members: Optional[dict[str, tarfile.TarInfo]] = None
        self._packed_keys: Optional[list[str]] = None

    def _ensure_open(self) -> None:
        if self._tar is None:
            self._fh = open(self.path, "rb")
            self._tar = tarfile.open(fileobj=self._fh, mode="r|")
            # Stream once to build name->TarInfo map (offsets recorded internally
            # by tarfile so subsequent extractfile() works on the same handle).
            self._members = {}
            packed_keys: list[str] = []
            # Re-open in random-access mode so extractfile works with offsets:
            self._tar.close()
            self._fh.close()
            self._fh = open(self.path, "rb")
            self._tar = tarfile.open(fileobj=self._fh, mode="r:")
            for ti in self._tar:
                if ti.isfile():
                    self._members[ti.name] = ti
                    if ti.name.endswith(".json"):
                        # ps_XXXXXXXX.json
                        key = ti.name[:-5]
                        packed_keys.append(key)
            packed_keys.sort()
            self._packed_keys = packed_keys

    @property
    def packed_keys(self) -> list[str]:
        with self._lock:
            self._ensure_open()
            assert self._packed_keys is not None
            return self._packed_keys

    def read_member(self, name: str) -> Optional[bytes]:
        with self._lock:
            self._ensure_open()
            assert self._members is not None and self._tar is not None
            ti = self._members.get(name)
            if ti is None:
                return None
            f = self._tar.extractfile(ti)
            if f is None:
                return None
            return f.read()

    def member_exists(self, name: str) -> bool:
        with self._lock:
            self._ensure_open()
            assert self._members is not None
            return name in self._members


class ShardRegistry:
    """LRU pool of opened TarShards (limit open file handles)."""

    def __init__(self, root: Path, max_open: int = 8):
        self.root = root
        self.max_open = max_open
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, TarShard] = OrderedDict()
        # Static list of shard names from filesystem
        names = sorted(p.name for p in root.glob("pretrain-*.tar"))
        if not names:
            raise RuntimeError(f"No pretrain-*.tar found under {root}")
        self.shard_names: list[str] = names

    def get(self, name: str) -> TarShard:
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
            if name not in set(self.shard_names):
                raise KeyError(name)
            shard = TarShard(self.root / name)
            self._cache[name] = shard
            # Evict oldest if needed
            while len(self._cache) > self.max_open:
                _, old = self._cache.popitem(last=False)
                # Close handle defensively
                try:
                    if old._tar is not None:
                        old._tar.close()
                    if old._fh is not None:
                        old._fh.close()
                except Exception:
                    pass
            return shard


IMG_NAME_RE = re.compile(r"^img(?P<sub>\d+)_(?P<frame>\d+)\.jpg$")


def parse_packed_json(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


def build_subsample_view(meta: dict, sub_idx: int) -> dict:
    """Build a clean per-sub-sample view from the packed JSON.

    Returns dict with:
      - images:    [{"frame": int, "filename": str, "timestamp": float|None,
                     "timestamp_str": str|None, "has_patch": bool}]
      - turns:     [{"role": "user"|"assistant", "content": str}]
      - fps:       float|int
      - timestamp_decimal: int|None
      - patch_summary: {"unique_count": int, "total": int}
    """
    sub_count = int(meta.get("sample_count", len(meta["images"])))
    if not (0 <= sub_idx < sub_count):
        raise IndexError(f"sub_idx {sub_idx} out of range [0,{sub_count})")

    images: list[str] = meta["images"][sub_idx]
    prompts: list[str] = meta["prompts"][sub_idx]
    captions: list[str] = meta["captions"][sub_idx]
    fps_list = meta.get("fps") or []
    td_list = meta.get("timestamp_decimal") or []
    pp_list = meta.get("patch_positions") or []

    fps = fps_list[sub_idx] if sub_idx < len(fps_list) else None
    td = td_list[sub_idx] if sub_idx < len(td_list) else None
    if td is None:
        td = 1
    patch_positions = pp_list[sub_idx] if sub_idx < len(pp_list) else []

    image_entries = []
    unique_patch = 0
    for fi, fname in enumerate(images):
        pp_entry = patch_positions[fi] if fi < len(patch_positions) else ""
        has_patch = bool(pp_entry)
        if has_patch:
            unique_patch += 1
        ts: Optional[float] = None
        ts_str: Optional[str] = None
        if fps:
            try:
                ts = round(fi / float(fps), int(td))
                ts_str = f"<{ts:.{int(td)}f}s>"
            except Exception:
                ts = None
        tar_member = f"img{sub_idx}_{fi}.jpg"
        image_entries.append(
            {
                "frame": fi,
                "filename": fname,
                "tar_member": tar_member,
                "timestamp": ts,
                "timestamp_str": ts_str,
                "has_patch": has_patch,
            }
        )

    # Multi-turn dialog interleaving
    turns = []
    n_turns = max(len(prompts), len(captions))
    for t in range(n_turns):
        if t < len(prompts):
            turns.append({"role": "user", "content": prompts[t]})
        if t < len(captions):
            turns.append({"role": "assistant", "content": captions[t]})

    return {
        "sub_idx": sub_idx,
        "sub_count": sub_count,
        "fps": fps,
        "timestamp_decimal": td,
        "num_frames": len(images),
        "num_turns": len(prompts),
        "patch_summary": {"unique": unique_patch, "total": len(images)},
        "images": image_entries,
        "turns": turns,
    }


def make_app(root: Path, max_open: int = 8) -> FastAPI:
    registry = ShardRegistry(root, max_open=max_open)
    app = FastAPI(title="VSI-590k Packed Visualizer")

    @app.get("/api/shards")
    def list_shards():
        return {"root": str(root), "count": len(registry.shard_names), "shards": registry.shard_names}

    @app.get("/api/shard/{shard_name}/keys")
    def shard_keys(shard_name: str):
        try:
            shard = registry.get(shard_name)
        except KeyError:
            raise HTTPException(404, f"Unknown shard: {shard_name}")
        keys = shard.packed_keys
        return {"shard": shard_name, "count": len(keys), "keys": keys}

    @app.get("/api/shard/{shard_name}/sample/{key}")
    def packed_sample(shard_name: str, key: str):
        try:
            shard = registry.get(shard_name)
        except KeyError:
            raise HTTPException(404, f"Unknown shard: {shard_name}")
        payload = shard.read_member(f"{key}.json")
        if payload is None:
            raise HTTPException(404, f"No JSON for key {key}")
        meta = parse_packed_json(payload)
        sub_count = int(meta.get("sample_count", len(meta["images"])))
        return {
            "shard": shard_name,
            "key": key,
            "sub_count": sub_count,
            "fps": meta.get("fps"),
            "timestamp_decimal": meta.get("timestamp_decimal"),
            "frames_per_sub": [len(g) for g in meta["images"]],
            "turns_per_sub": [len(p) for p in meta["prompts"]],
        }

    @app.get("/api/shard/{shard_name}/sample/{key}/sub/{sub_idx}")
    def packed_subsample(shard_name: str, key: str, sub_idx: int):
        try:
            shard = registry.get(shard_name)
        except KeyError:
            raise HTTPException(404, f"Unknown shard: {shard_name}")
        payload = shard.read_member(f"{key}.json")
        if payload is None:
            raise HTTPException(404, f"No JSON for key {key}")
        meta = parse_packed_json(payload)
        try:
            view = build_subsample_view(meta, sub_idx)
        except IndexError as e:
            raise HTTPException(400, str(e))
        view["shard"] = shard_name
        view["key"] = key
        return view

    @app.get("/api/shard/{shard_name}/image/{key}/{member}")
    def packed_image(shard_name: str, key: str, member: str):
        # member like "img0_5.jpg"
        if not IMG_NAME_RE.match(member):
            raise HTTPException(400, f"Bad image member: {member}")
        try:
            shard = registry.get(shard_name)
        except KeyError:
            raise HTTPException(404, f"Unknown shard: {shard_name}")
        data = shard.read_member(f"{key}.{member}")
        if data is None:
            raise HTTPException(404, f"No image {key}.{member}")
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/shard/{shard_name}/patch/{key}/{member}")
    def packed_patch(shard_name: str, key: str, member: str):
        # member like "img0_5.npy"
        try:
            shard = registry.get(shard_name)
        except KeyError:
            raise HTTPException(404, f"Unknown shard: {shard_name}")
        data = shard.read_member(f"{key}.{member}")
        if data is None:
            raise HTTPException(404, f"No patch {key}.{member}")
        arr = np.load(io.BytesIO(data))
        return JSONResponse(
            {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "min": [int(x) for x in arr.min(axis=0).tolist()] if arr.ndim == 2 else None,
                "max": [int(x) for x in arr.max(axis=0).tolist()] if arr.ndim == 2 else None,
                "n_tokens": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            }
        )

    # Static frontend
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (static_dir / "index.html").read_text(encoding="utf-8")

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--max-open", type=int, default=8, help="Max simultaneously-open tar shards")
    args = ap.parse_args()

    import uvicorn

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Root not found: {root}")

    app = make_app(root, max_open=args.max_open)
    print(f"[vsi_viz] root={root}  serving http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
