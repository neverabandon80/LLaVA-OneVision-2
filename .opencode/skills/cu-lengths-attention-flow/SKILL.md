---
name: cu-lengths-attention-flow
description: Bilingual guide for understanding how cu_lengths controls attention behavior across ViT and LLM stages, and how patch_positions scope differs between the two
compatibility: opencode
metadata:
  domain: model-architecture
  framework: megatron-energon
  repo: llava-onevision2
---

## Purpose / 用途

Use this skill when reasoning about attention boundaries in the LLaVA-OneVision2 forward pass — specifically how `cu_lengths` and `patch_positions` control attention at different stages of the model.

在分析 LLaVA-OneVision2 前向传播中的 attention 边界时使用这个 skill——具体来说，`cu_lengths` 和 `patch_positions` 如何在模型的不同阶段控制 attention。

This skill is specifically for:

- Understanding the difference between ViT-level and LLM-level attention control
- Debugging packed vs non-packed attention behavior
- Reasoning about cross-sample isolation in packed sequences
- Understanding why `patch_positions` grouping does NOT carry into the LLM

这个 skill 专门用于：

- 理解 ViT 层和 LLM 层 attention 控制的区别
- 调试 packed 和 non-packed 的 attention 行为
- 分析 packed 序列中跨样本隔离机制
- 理解为什么 `patch_positions` 的分组不会延续到 LLM 中

---

## Key Files / 关键文件

| File | Role |
|---|---|
| `aiak_training_llm/train/pretrain/pretrain_llava_onevision2.py` | Forward function — decides packed vs non-packed path based on `cu_lengths` shape |
| `aiak_training_llm/train/sft/utils.py` | `_get_packed_sequence_params()` — builds `PackedSeqParams` from attention_mask for SFT |
| `aiak_training_llm/data/multimodal/task_encoder.py` | `batch()` — sets `cu_lengths` to `[[0]]` (dummy) for non-packed, or stacks real `cu_lengths` for packed |
| `aiak_training_llm/data/multimodal/task_encoder.py` | `pack_selected_samples()` — constructs `cu_lengths = [0, len_1, len_1+len_2, ...]` for offline packed data |
| `aiak_training_llm/models/llava_onevision2/onevision_encoder_model.py` | ViT encoder — uses `patch_positions` for local/shared attention |
| `aiak_training_llm/data/multimodal/qwen2vl_task_encoder.py` | `process_sft_qa()` — generates `patch_positions` from `image_grid_thw` |

---

## Core Concept: Two Independent Attention Control Mechanisms / 核心概念：两套独立的 Attention 控制机制

### Overview Diagram / 概览图

```
┌─────────────────────────────────────────────────┐
│  ViT Encoder                                    │
│                                                 │
│  Control: patch_positions (temporal dimension)  │
│  Effect:  Local/shared attention                │
│           e.g. 4 images share one attention     │
│           window via same temporal index        │
│                                                 │
│  Output:  visual embeddings                     │
└──────────────────┬──────────────────────────────┘
                   │  (embeddings replace image
                   │   placeholder tokens)
                   ▼
┌─────────────────────────────────────────────────┐
│  LLM Decoder                                    │
│                                                 │
│  Control: cu_lengths (cumulative sub-seq lens)  │
│  Effect:  Determines attention domain           │
│                                                 │
│  NON-PACKED: cu_lengths == [[0]]                │
│    → full causal attention                      │
│    → ALL tokens see ALL previous tokens         │
│    → patch_positions grouping is GONE           │
│                                                 │
│  PACKED: cu_lengths = [0, a, a+b, ...]          │
│    → block-diagonal causal attention            │
│    → sub-sequences isolated from each other     │
│    → within each sub-seq: full causal           │
└─────────────────────────────────────────────────┘
```

---

## Mechanism Details / 机制详解

### 1. `cu_lengths` Generation / `cu_lengths` 的产生

#### Source A: Offline Packed Data (`PackedCaptioningSample`)

```python
# task_encoder.py → pack_selected_samples()
cu_lengths = [0]
for sample in samples:
    current_length += sample.total_len
    cu_lengths.append(current_length)
# Result: [0, 512, 1024, 1389] — 3 sub-samples packed together
```

Each sub-sample was independently encoded (tokenized + image processed) then concatenated into one long sequence. `cu_lengths` records the boundaries.

每个子样本独立编码（tokenize + 图像处理），然后拼接成一个长序列。`cu_lengths` 记录边界。

#### Source B: Non-Packed Single Samples

```python
# task_encoder.py → batch()
if self.is_packing_enabled or int(os.environ.get("OFFLINE_PACKED_DATA", 0)) == 1:
    cu_lengths = torch.stack([s.cu_lengths for s in samples])
else:
    cu_lengths = torch.tensor([[0]], dtype=torch.int32)  # dummy value
```

Non-packed samples get `cu_lengths = [[0]]` with shape `[1, 1]`.

非 packed 样本得到 `cu_lengths = [[0]]`，shape 为 `[1, 1]`。

### 2. Forward Function Branching / 前向函数分支

In `pretrain_llava_onevision2.py`:

```python
if cu_lengths.shape == torch.Size([1, 1]):
    # ===== NON-PACKED PATH =====
    # Uses attn_mask for padding_causal attention
    # Every token attends to all previous tokens (full causal)
    # packed_seq_params = None
    for i in range(attn_mask.shape[0]):
        loss_mask[i, (attn_mask[i] == False).sum() - 1] = 0
else:
    # ===== PACKED PATH =====
    # micro-batch-size must be 1 for packing
    assert cu_lengths.shape[0] == 1
    attn_mask = None  # not needed — cu_seqlens defines boundaries
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_lengths[0],      # → Flash Attention kernel
        cu_seqlens_kv=cu_lengths[0],
        max_seqlen_q=max_lengths[0].item(),
        max_seqlen_kv=max_lengths[0].item(),
    )
```

### 3. LLM Attention Behavior / LLM 层的 Attention 行为

**Non-packed (`cu_lengths == [[0]]`) → Full Causal Attention**：

- The entire sequence is ONE attention domain
- Token at position `i` can attend to positions `0..i`
- ALL visual tokens (from any image) + ALL text tokens are mutually visible
- Compute cost: O(n²) where n = total sequence length
- `packed_seq_params = None` → standard causal mask

整个序列是一个 attention 域，所有 visual token（来自任何图片）和所有 text token 互通可见。计算量 O(n²)。

**Packed (`cu_lengths = [0, a, a+b, ...]`) → Block-Diagonal Causal Attention**：

- Each sub-sequence `[cu_lengths[i], cu_lengths[i+1])` is an independent attention domain
- Within each sub-sequence: full causal attention
- Between sub-sequences: ZERO attention (completely isolated)
- Implemented via Flash Attention's `cu_seqlens` parameter
- Compute cost: O(a² + b² + c² + ...) << O((a+b+c+...)²)

每个子序列 `[cu_lengths[i], cu_lengths[i+1])` 是独立的 attention 域，子序列之间完全隔离。通过 Flash Attention 的 `cu_seqlens` 参数实现。

### 4. Sequence Parallelism Padding / 序列并行填充

When `args.sequence_parallel` is enabled, the sequence must be divisible by TP size (and TP×CP×2 if CP > 1). For packed sequences, padding tokens are appended as a dummy extra sub-sequence:

当启用 `sequence_parallel` 时，序列长度必须被 TP size 整除。对 packed 序列，padding token 作为一个额外的 dummy 子序列追加：

```python
if packed_seq_params is not None:
    new_end_q = packed_seq_params.cu_seqlens_q[-1:] + pad_size
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=torch.cat([packed_seq_params.cu_seqlens_q, new_end_q]),
        cu_seqlens_kv=torch.cat([packed_seq_params.cu_seqlens_kv, new_end_kv]),
        max_seqlen_q=max(packed_seq_params.max_seqlen_q, pad_size),
        max_seqlen_kv=max(packed_seq_params.max_seqlen_kv, pad_size),
    )
```

---

## Critical Insight: `patch_positions` Scope / 关键洞察：`patch_positions` 的作用域

### ViT Layer: `patch_positions` Controls Attention Grouping

In the ViT encoder, `patch_positions` has a temporal dimension (t, h, w). Images sharing the same temporal index share one attention window. For example, 4 images treated as "video frames" share attention via their temporal coordinates.

在 ViT encoder 中，`patch_positions` 有时间维度 (t, h, w)。共享相同时间索引的图片共享一个 attention window。例如，4 张图片被当作"视频帧"通过时间坐标共享 attention。

### LLM Layer: `patch_positions` Has NO Effect on Attention

**`patch_positions` is NOT used to control LLM attention.** It is only passed through the data pipeline for potential use in position embeddings or other purposes, but the LLM's attention boundaries are controlled EXCLUSIVELY by `cu_lengths`.

**`patch_positions` 不控制 LLM 的 attention。** 它只是在数据 pipeline 中传递，可能用于 position embedding 等目的，但 LLM 的 attention 边界完全由 `cu_lengths` 控制。

This means:

这意味着：

| Scenario | ViT Attention | LLM Attention |
|---|---|---|
| 4 images with shared `patch_positions` temporal index, non-packed | 4 images share attention window | ALL tokens (all 4 images + text) in full causal — **no grouping** |
| 4 images with shared `patch_positions` temporal index, packed (separate sub-samples) | 4 images share attention window | Each sub-sample isolated via `cu_seqlens` — **inter-sample isolation** |
| Single image, non-packed | Standard ViT attention | Full causal over entire sequence |

### Why This Matters / 为什么这很重要

If you assume ViT-level grouping persists into the LLM, you will misunderstand the compute profile:

如果你假设 ViT 层的分组延续到 LLM 中，会误解计算特征：

- **Non-packed**: LLM always does full causal attention over the entire sequence. A sample with 4 high-res images has O((4×img_tokens + text_tokens)²) attention cost — there is NO per-image isolation in the LLM.
- **Packed**: The isolation is between SAMPLES (sub-sequences), not between images within a sample. A packed sample containing samples A (2 images) and B (1 image) isolates A from B, but within A, both images + text are fully visible to each other.

- **Non-packed**: LLM 总是对整个序列做 full causal attention。一个包含 4 张高分辨率图片的样本，attention 代价为 O((4×img_tokens + text_tokens)²)——LLM 中没有按图片隔离。
- **Packed**: 隔离是在样本（子序列）之间，不是在同一样本内的图片之间。一个包含样本 A（2 张图）和样本 B（1 张图）的 packed 样本，A 和 B 互相隔离，但 A 内部的两张图 + 文本完全互通。

---

## SFT Path: Attention Mask Based `cu_seqlens` / SFT 路径：基于 attention_mask 的 cu_seqlens

In the SFT training path (`sft/utils.py`), `cu_seqlens` can also be derived from the attention mask using sample-ID encoding:

在 SFT 训练路径中，`cu_seqlens` 也可以从 attention mask 推导：

```python
# sft/utils.py → _get_packed_sequence_params()
# attention_mask encodes sample IDs: [[1,1,2,2,2,3,3,4,5,5,5,0,0]]
# → cu_seqlens = [0, 2, 5, 7, 8, 11, 13]
reduced_mask = torch.bincount(attention_mask.view(-1), minlength=max_num + 1)
cu_seqlens = reduced_mask[1:].cumsum(dim=0).to(torch.int32)
cu_seqlens[-1] = attention_mask.shape[1]  # include padding
cu_seqlens = torch.cat((zero, cu_seqlens))
```

This achieves the same block-diagonal attention as the pretrain path's `cu_lengths` mechanism.

这与 pretrain 路径的 `cu_lengths` 机制实现相同的 block-diagonal attention。

---

## Quick Reference Table / 快速参考表

| Field | Where Set | What It Controls |
|---|---|---|
| `cu_lengths` | `task_encoder.batch()` or `pack_selected_samples()` | LLM attention boundaries (packed vs full causal) |
| `packed_seq_params` | `pretrain_*.py` forward function | Flash Attention kernel parameter (`cu_seqlens_q/kv`) |
| `patch_positions` | `qwen2vl_task_encoder.process_sft_qa()` | ViT local attention grouping (temporal dimension) |
| `attn_mask` | `encode_sample()` | Padding mask for non-packed; set to `None` for packed |
| `max_lengths` | `pack_selected_samples()` | Max sub-sequence length in packed sample (for Flash Attention) |

| Shape of `cu_lengths` | Meaning | LLM Attention Type |
|---|---|---|
| `[1, 1]` (value `[[0]]`) | Non-packed / dummy | Full causal |
| `[1, P]` where P > 1 | Packed with P-1 sub-samples | Block-diagonal causal |

---

## Common Pitfalls / 常见误区

1. **Assuming ViT attention grouping carries into LLM** — It does NOT. `patch_positions` only affects ViT; LLM uses `cu_lengths`.

   **假设 ViT 的 attention 分组延续到 LLM** — 不会。`patch_positions` 只影响 ViT；LLM 使用 `cu_lengths`。

2. **Confusing packed sample isolation with image-level isolation** — `cu_lengths` boundaries separate SAMPLES, not images within a sample.

   **混淆 packed 样本隔离和图片级隔离** — `cu_lengths` 边界分隔的是样本，不是样本内的图片。

3. **Forgetting micro-batch-size=1 constraint for packing** — The code asserts `cu_lengths.shape[0] == 1` in the packed path.

   **忘记 packing 要求 micro-batch-size=1** — 代码在 packed 路径中断言 `cu_lengths.shape[0] == 1`。

4. **Ignoring SP padding for packed sequences** — When sequence parallelism is enabled, padding tokens are added as a dummy sub-sequence in `cu_seqlens`, not ignored.

   **忽略 packed 序列的 SP 填充** — 启用序列并行时，padding token 作为 dummy 子序列加入 `cu_seqlens`，不是被忽略。
