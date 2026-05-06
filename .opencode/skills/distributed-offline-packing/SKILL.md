---
name: distributed-offline-packing
description: Bilingual guide for running offline_packing/auto_pipe.sh across multiple nodes to produce padding-free packed WebDataset shards for SFT, with Energon Metadataset assembly
compatibility: opencode
metadata:
  domain: data-pipeline
  framework: llava-onevision2
  repo: llava-onevision2
---

## Purpose / 用途

Use this skill when packing a large SFT JSONL (hundreds of thousands to millions of samples) into Energon WebDataset shards at a fixed sequence length, using `offline_packing/auto_pipe.sh` parallelized across multiple nodes.

当需要把大规模 SFT JSONL（几十万到几百万样本）按固定序列长度打包成 Energon WebDataset shards，并通过 `offline_packing/auto_pipe.sh` 在多台机器上并行处理时，使用这个 skill。

## Prerequisites / 前置条件

- All nodes share the same NFS mount (data + repo + output)
- Same docker image on every node (must contain transformers, energon, project repo)
- `offline_packing/auto_pipe.sh` and stage scripts `s1_split_json_to_samples.py` … `s4_bins_to_webdataset.py`
- Tokenizer + image processor available locally (HF format model dir)
- Source JSONL where each line has `images`, `prompts`, `captions` (multi-turn list-of-lists) and image paths are usable as-is

所有节点共享同一个 NFS（数据+代码+输出）。每台机器使用同一个 docker 镜像，里面要有 transformers、energon、项目代码。需要 `offline_packing/auto_pipe.sh` 和 s1–s4 四个 stage 脚本。Tokenizer 和 image processor 是本地 HF 格式目录。源 JSONL 每行包含 `images / prompts / captions`（多轮是 list-of-list），图片路径可直接使用。

## Architecture / 架构

### Pipeline Stages / 流水线阶段

```
JSONL (N samples)
  ├─ s1_split_json_to_samples.py   # validate + drop bad/missing-image samples
  │                                # output: per-sample serialized records
  ├─ s2_compute_token_lengths.py   # tokenize prompts/captions, compute image-patch tokens
  │                                # output: length array per sample
  ├─ s3_bin_packing.py             # BFD (Best-Fit-Decreasing) into bins of capacity L
  │                                # output: bin assignment
  └─ s4_bins_to_webdataset.py      # write tar shards + idx + .nv-meta/{dataset.yaml,split.yaml,sample_loader.py}
```

`auto_pipe.sh` runs all four stages sequentially on **one node** for **one input file**. To use N nodes, split the JSONL into N parts and run `auto_pipe.sh` independently on each — s3 BFD does NOT shard across nodes, so each node packs its own slice.

`auto_pipe.sh` 在一台机器上对一个输入文件顺序跑完四个 stage。要用 N 台机器就把 JSONL 切成 N 份，每台独立跑一次 `auto_pipe.sh`——s3 BFD 不能跨节点共享，所以每台只 pack 自己的那一份。

### Key Design Decisions / 关键设计

- **Per-node shard prefix**: pass distinct `--shard-prefix <name>_<a|b|...>` so tar files don't collide on shared NFS
- **Sample class**: choose based on data shape — `PackedCaptioningSample` for image+text packed turns, `MultiMixQASample` for QA-style. The class controls the auto-generated `sample_loader.py`
- **`--no-npy` flag**: only set when JSONL has no precomputed `patch_positions` field. Without `--no-npy` s2 expects per-sample `.npy` files; missing files cause noisy warnings AND fall back to slow real-image tokenization
- **Sequence length L**: scan token lengths first to confirm drop rate is acceptable
- **Output layout per node**: `<output_root>/node_<x>/webdataset/{*.tar, *.idx, .nv-meta/}`
- **Top-level Metadataset yaml** combines all node outputs into one logical dataset

## Step-by-step Workflow / 操作步骤

### 0. Pre-flight: choose L / 预检：选 L

Scan token lengths once on the full JSONL using `s2_compute_token_lengths.py` (or a quick standalone script). Pick L so that drop rate is acceptable (typically <0.1%).

```bash
# Inside container, on any single node
python offline_packing/s2_compute_token_lengths.py \
    --jsonl <path/to/full.jsonl> \
    --tokenizer <path/to/tokenizer> \
    --image-processor Qwen2_5_VLProcessor \
    --factor 48 --min-pixels 3136 --max-pixels 4000000 \
    --output <path/to/token_lens.txt>

# Then quickly inspect distribution (max, p99, count > L) before committing to L
```

### 1. Split JSONL across nodes / 切分 JSONL

```bash
TOTAL=$(wc -l < full.jsonl)
HALF=$(( (TOTAL + 1) / 2 ))
split -l $HALF -d --additional-suffix=.jsonl full.jsonl part_
# produces part_00.jsonl, part_01.jsonl
```

For >2 nodes, adjust `-l` accordingly.

### 2. Mount NFS on every node / 挂载 NFS

Make sure the same NFS is mounted on every node at the same path so paths in JSONL and outputs match.

### 3. Launch container on every node / 每台启动容器

Use the project's standard docker image with the repo bind-mounted. Working directory should be the repo root.

### 4. Run auto_pipe.sh in parallel on each node / 并行启动

On node A:

```bash
cd <repo_root>
bash offline_packing/auto_pipe.sh \
    --jsonl <data_root>/part_00.jsonl \
    --tokenizer <tokenizer_path> \
    --image-processor Qwen2_5_VLProcessor \
    --factor 48 --min-pixels 3136 --max-pixels 4000000 \
    --image-root / \
    --sample-class PackedCaptioningSample \
    --shard-prefix <dataset_name>_a \
    --output-dir <output_root>/node_a \
    --seq-len 4096 \
    --no-npy \
    2>&1 | tee <log_dir>/node_a.log
```

On node B (in parallel):

```bash
bash offline_packing/auto_pipe.sh \
    --jsonl <data_root>/part_01.jsonl \
    ... \
    --shard-prefix <dataset_name>_b \
    --output-dir <output_root>/node_b \
    ... \
    2>&1 | tee <log_dir>/node_b.log
```

> [!IMPORTANT]
> - **`--shard-prefix` must differ** between nodes so tar filenames don't collide
> - **`--output-dir` must differ** between nodes
> - `--image-root /` if JSONL paths are absolute; otherwise set it to the image root prefix
> - Add `--no-npy` if the JSONL has no `patch_positions` field

### 5. Verify outputs / 验证产物

```bash
# Tar count per node should match s4 log
ls <output_root>/node_a/webdataset/*.tar | wc -l
ls <output_root>/node_b/webdataset/*.tar | wc -l

# Bin count + capacity utilization printed by s3
grep -E "(bins|util|efficiency)" <log_dir>/node_a.log

# Inspect one tar to confirm sample schema
mkdir -p /tmp/tar_inspect && cd /tmp/tar_inspect
tar -xf <output_root>/node_a/webdataset/<prefix>-000000.tar
ls | head
python -c "import json; d=json.load(open(open(__import__('glob').glob('*.json')[0]).name)); print(list(d.keys()))"
```

Expected JSON top-level keys for `PackedCaptioningSample`:
`images`, `prompts`, `captions`, `sample_count`, `patch_positions`, `timestamp_decimal`.

`patch_positions=[[""]]` is normal under `--no-npy` — the auto-generated `sample_loader.py` handles it via `sample.get(..., None)`.

### 6. Write top-level Metadataset yaml / 写顶层 Metadataset yaml

```yaml
__module__: megatron.energon
__class__: Metadataset
splits:
  train:
    datasets:
      - weight: <num_samples_in_part_00>
        path: <output_root>/node_a/webdataset
        subflavors:
          augmentation: false
      - weight: <num_samples_in_part_01>
        path: <output_root>/node_b/webdataset
        subflavors:
          augmentation: false
```

Notes / 注意:
- `weight` is sample-count proportional (use the original input line count of each part, not the bin count)
- Use **absolute paths** — Energon resolves them as-is
- Add `val` split only if you actually need one; for SFT-only pipelines omit it
- `subflavors` is optional; here we mark `augmentation: false` since data is already packed

## Common Pitfalls / 常见坑

### Pitfall 1: forgetting `--no-npy`
Without it s2 hunts for per-sample `.npy` files and either spams warnings OR falls back to the slow real-image tokenization path. Always check the source JSONL for a `patch_positions` field first.

### Pitfall 2: same `--shard-prefix` on multiple nodes
Tar filenames collide on shared NFS, second node overwrites the first. Use distinct suffixes (`_a`, `_b`, `_n0`, `_n1`, …).

### Pitfall 3: starting fresh while old `rm -rf` still running on NFS
Removing millions of small files from NFS can take 10+ minutes. Don't wait — use a fresh `_v2` output dir, and `mv` (atomic rename on same FS) when done if you want the original name back.

### Pitfall 4: `du -sh` / `rm -rf` exceeding bash 120s timeout
Run them as `nohup ... &` and poll the PID, or run them in tmux.

### Pitfall 5: container vs host timezone skew
Container time may differ from host time but wall clock is the same. Don't be confused by log timestamps when comparing across docker exec sessions.

### Pitfall 6: choosing the wrong `--sample-class`
The class determines the auto-generated `sample_loader.py` and how downstream code unpacks tars. Confirm by reading the dataclass file (e.g. `aiak_training_llm/data/multimodal/flavors/packed_captioning.py`) and matching its fields to your JSONL shape.

### Pitfall 7: `weight` confusion in Metadataset yaml
Use sample counts (or proportional integers), not bin counts. Energon samples each dataset proportional to weight.

## Performance Reference / 性能参考

For ~390k samples per node at L=4096 on a multi-core machine with NFS storage:
- s1 (split + validate): ~30 min (NFS small-file IO bound; gets worse if concurrent `rm` is running)
- s2 (token lengths, with `--no-npy`): ~5 min
- s3 (BFD bin packing): <1 min
- s4 (tar writing): ~5 min
- **Total wall time per node: ~40–45 min**

Two nodes in parallel ≈ same wall time as one node, so 2× throughput.

Typical packing efficiency at L=4096 with avg ~10 samples/bin: **>99% capacity utilization**.

## Quick Sanity Checklist / 快速自检清单

Before declaring done:

- [ ] Tar count per node matches s4 log
- [ ] s3 log shows >95% capacity utilization
- [ ] At least one tar inspected; JSON keys match the chosen sample class
- [ ] `sample_count` in tar JSON > 0 (not all 1; means packing actually worked)
- [ ] Top-level `dataset.yaml` exists with absolute paths and correct weights
- [ ] (Optional) Energon load smoke test passes

## Related Files / 相关文件

- `offline_packing/auto_pipe.sh` — pipeline driver
- `offline_packing/s1_split_json_to_samples.py` — JSONL → per-sample records
- `offline_packing/s2_compute_token_lengths.py` — token length computation
- `offline_packing/s3_bin_packing.py` — BFD bin packing
- `offline_packing/s4_bins_to_webdataset.py` — tar + idx + .nv-meta writer
- `aiak_training_llm/data/multimodal/flavors/packed_captioning.py` — `PackedCaptioningSample` dataclass
- `aiak_training_llm/data/multimodal/flavors/multi_mix_qa.py` — `MultiMixQASample` dataclass
- `aiak_megatron/examples/multimodal/sft_dataset.yaml` — Metadataset yaml template
