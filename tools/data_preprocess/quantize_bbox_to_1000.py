"""Quantize bbox floats [x1,y1,x2,y2] in [0,1] -> integers in [0,999].

Only operates on samples whose source is vg/coco (the only datasets that contain
grounding boxes). Format wrapping unchanged: '[542,349,612,409]'.

Backs up the original to <path>.f32.bak before in-place rewrite.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import time

JSONL = "/ov2/dataset_jsonl/llava_next/megatron_format_780k.local.jsonl"
BACKUP = JSONL + ".f32.bak"
TMP = JSONL + ".q1000.tmp"

BBOX_RE = re.compile(
    r"\[\s*(0?\.\d{1,4}|0|1\.0+)\s*,\s*(0?\.\d{1,4}|0|1\.0+)\s*,"
    r"\s*(0?\.\d{1,4}|0|1\.0+)\s*,\s*(0?\.\d{1,4}|0|1\.0+)\s*\]"
)
GROUNDING_SOURCES = {"vg", "coco"}


def src_of(images: list[str]) -> str:
    if not images:
        return ""
    parts = images[0].split("/")
    try:
        i = parts.index("llava_next")
        return parts[i + 1]
    except (ValueError, IndexError):
        return ""


def quantize_match(m: re.Match) -> str:
    out = []
    for v in m.groups():
        f = float(v)
        q = round(f * 1000)
        if q < 0:
            q = 0
        elif q > 999:
            q = 999
        out.append(str(q))
    return "[" + ",".join(out) + "]"


def main() -> int:
    if not os.path.exists(BACKUP):
        print(f"Backing up {JSONL} -> {BACKUP}")
        shutil.copy2(JSONL, BACKUP)
    else:
        print(f"Backup already exists at {BACKUP}, skipping copy")

    t0 = time.time()
    n_lines = 0
    n_grounding_samples = 0
    n_substitutions = 0

    with open(BACKUP) as fi, open(TMP, "w") as fo:
        for line in fi:
            n_lines += 1
            s = json.loads(line)
            src = src_of(s["images"])
            if src in GROUNDING_SOURCES:
                changed = False
                local_subs = 0
                for m in s["messages"]:
                    c = m.get("content")
                    if not isinstance(c, str):
                        continue
                    new_c, k = BBOX_RE.subn(quantize_match, c)
                    if k:
                        m["content"] = new_c
                        local_subs += k
                        changed = True
                if changed:
                    n_grounding_samples += 1
                    n_substitutions += local_subs
            fo.write(json.dumps(s, ensure_ascii=False) + "\n")
            if n_lines % 100_000 == 0:
                print(f"  {n_lines:,} lines  ({time.time() - t0:.1f}s)")

    os.replace(TMP, JSONL)
    print()
    print(f"DONE in {time.time() - t0:.1f}s")
    print(f"  lines               : {n_lines:,}")
    print(f"  samples with bbox   : {n_grounding_samples:,}")
    print(f"  bbox substitutions  : {n_substitutions:,}")
    print(f"  output              : {JSONL}")
    print(f"  backup              : {BACKUP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
