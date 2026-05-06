"""Convert /vlm/data/llava_next_raw_full/megatron_format_780k.json (JSON array)
into a JSONL file where each line is:

    {
      "id": "<20-digit-decimal-derived-from-sha1>",
      "messages": [{"role": ..., "content": ...}, ...],
      "images": ["/vlm/data/llava_next_raw_full/images/<rel>", ...]
    }

- Image paths are rewritten from relative to absolute under IMAGE_ROOT.
- Text-only samples (no "images" key) are kept with images=[].
- id is deterministic: first 20 decimal digits of sha1(messages + images_rel).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

SRC = "/vlm/data/llava_next_raw_full/megatron_format_780k.json"
DST = "/ov2/dataset_jsonl/llava_next/megatron_format_780k.jsonl"
IMAGE_ROOT = "/vlm/data/llava_next_raw_full/images"


def make_id(messages: list, images_rel: list) -> str:
    """Deterministic 20-digit decimal id from sample content."""
    payload = json.dumps(
        {"m": messages, "i": images_rel},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()
    big = int(digest, 16)
    return str(big)[:20].zfill(20)


def main() -> None:
    src = Path(SRC)
    dst = Path(DST)
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    print(f"Loaded {n:,} samples. Writing to {dst}")

    n_with_img = 0
    n_text_only = 0

    with dst.open("w", encoding="utf-8") as fo:
        for i, sample in enumerate(data):
            messages = sample.get("messages", [])
            images_rel = sample.get("images", []) or []

            if images_rel:
                n_with_img += 1
                images_abs = [os.path.join(IMAGE_ROOT, p) for p in images_rel]
            else:
                n_text_only += 1
                images_abs = []

            out = {
                "id": make_id(messages, images_rel),
                "messages": messages,
                "images": images_abs,
            }
            fo.write(json.dumps(out, ensure_ascii=False) + "\n")

            if (i + 1) % 100_000 == 0:
                print(f"  ... {i + 1:,}/{n:,}")

    print("Done.")
    print(f"  with images : {n_with_img:,}")
    print(f"  text-only   : {n_text_only:,}")
    print(f"  total       : {n_with_img + n_text_only:,}")
    print(f"  output      : {dst}")


if __name__ == "__main__":
    main()
