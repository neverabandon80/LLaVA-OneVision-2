"""Copy LLaVA-Next image dataset from /mnt/vlmdata/.../llava_next_raw_format
to /ov2/dataset_sft_source/llava_next using 32 parallel processes.

Only copies images referenced by the jsonl. Skips files that already exist
on the target with matching size. Logs to /tmp/copy_llava_next.log.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

OLD_ROOT = "/vlm/data/llava_next_raw_full/images"
SRC_ROOT = "/mnt/vlmdata/data/train_images/llava_next_raw_format"
DST_ROOT = "/ov2/dataset_sft_source/llava_next"
JSONL = "/ov2/dataset_jsonl/llava_next/megatron_format_780k.jsonl"
LOG = "/tmp/copy_llava_next.log"
WORKERS = 32


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def copy_one(rel: str) -> tuple[str, str, int]:
    """Returns (rel, status, bytes_copied). status in {copied, skipped, missing, error:<msg>}."""
    src = os.path.join(SRC_ROOT, rel)
    dst = os.path.join(DST_ROOT, rel)
    try:
        try:
            src_st = os.stat(src)
        except FileNotFoundError:
            return rel, "missing", 0
        if os.path.exists(dst):
            try:
                if os.path.getsize(dst) == src_st.st_size:
                    return rel, "skipped", 0
            except OSError:
                pass
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".part"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return rel, "copied", src_st.st_size
    except Exception as e:
        return rel, f"error:{type(e).__name__}:{e}", 0


def main() -> None:
    open(LOG, "w").close()
    t0 = time.time()

    log("Step 1: collect unique relative image paths from jsonl ...")
    rels: set[str] = set()
    with open(JSONL) as f:
        for line in f:
            s = json.loads(line)
            for p in s["images"]:
                if p.startswith(OLD_ROOT + "/"):
                    rels.add(p[len(OLD_ROOT) + 1 :])
                else:
                    rels.add(p.lstrip("/"))
    rels_list = sorted(rels)
    log(f"  {len(rels_list):,} unique image paths to copy")
    log(f"  src: {SRC_ROOT}")
    log(f"  dst: {DST_ROOT}")

    log("Step 2: pre-create top-level dst dirs ...")
    tops = sorted({r.split('/', 1)[0] for r in rels_list})
    for t in tops:
        os.makedirs(os.path.join(DST_ROOT, t), exist_ok=True)
    log(f"  {len(tops)} top-level dirs ready: {tops}")

    log(f"Step 3: copy with {WORKERS} processes ...")
    counts = {"copied": 0, "skipped": 0, "missing": 0, "error": 0}
    bytes_copied = 0
    n = len(rels_list)
    last_print = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, (rel, status, nbytes) in enumerate(
            ex.map(copy_one, rels_list, chunksize=64), 1
        ):
            if status == "copied":
                counts["copied"] += 1
                bytes_copied += nbytes
            elif status == "skipped":
                counts["skipped"] += 1
            elif status == "missing":
                counts["missing"] += 1
            else:
                counts["error"] += 1
                if counts["error"] <= 20:
                    log(f"  ERROR {rel}: {status}")

            now = time.time()
            if i % 5000 == 0 or now - last_print > 30:
                rate = i / (now - t0)
                gb = bytes_copied / 1024**3
                eta = (n - i) / rate if rate > 0 else 0
                log(
                    f"  {i:>7,}/{n:,} "
                    f"copied={counts['copied']:,} skip={counts['skipped']:,} "
                    f"miss={counts['missing']:,} err={counts['error']:,} "
                    f"| {gb:.2f} GiB | {rate:.0f} files/s | ETA {eta / 60:.1f} min"
                )
                last_print = now

    log("")
    log("=" * 60)
    log(f"DONE in {(time.time() - t0)/60:.1f} min")
    log(f"  total      : {n:,}")
    log(f"  copied     : {counts['copied']:,}  ({bytes_copied/1024**3:.2f} GiB)")
    log(f"  skipped    : {counts['skipped']:,}")
    log(f"  missing    : {counts['missing']:,}")
    log(f"  error      : {counts['error']:,}")


if __name__ == "__main__":
    sys.exit(main())
