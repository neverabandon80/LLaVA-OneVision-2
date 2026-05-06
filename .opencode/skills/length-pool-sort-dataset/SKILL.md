---
name: length-pool-sort-dataset
description: Bilingual guide for understanding LengthPoolSortDataset cross-rank length synchronization mechanism in multi-GPU training
compatibility: opencode
metadata:
  domain: distributed-training
  framework: megatron-energon
  repo: llava-onevision2
---

## Purpose / 用途

Use this skill when analyzing, debugging, or tuning `LengthPoolSortDataset` — the cross-rank length synchronization mechanism used in this repository's training pipeline.

在分析、调试或调优本仓库训练 pipeline 中的跨 rank 长度同步机制 `LengthPoolSortDataset` 时，使用这个 skill。

This skill is specifically for:

- `aiak_training_llm/data/multimodal/length_sort_dataset.py`
- Understanding why training speed improves when `length_sort_pool_size > 0`
- Tuning `pool_size` for optimal multi-GPU efficiency
- Diagnosing rank synchronization bottlenecks

这个 skill 专门用于：

- `aiak_training_llm/data/multimodal/length_sort_dataset.py`
- 理解为什么 `length_sort_pool_size > 0` 时训练速度提升
- 调优 `pool_size` 以获得最佳多卡效率
- 排查 rank 间同步瓶颈

## Core mechanism / 核心机制

### Three-step pipeline / 三步流水线

```
上游 dataset → 累积 pool_size 个 sample → 按序列长度排序 → 用确定性 seed shuffle → 逐个 yield
```

```python
for batch_idx, sample in enumerate(self.dataset):
    pool.append(sample)
    if len(pool) >= self.pool_size:
        pool.sort(key=self.key_fn)                     # 1. 按长度排序
        shuffle_seed = 42 + batch_idx                   # 2. 确定性 seed
        random.Random(shuffle_seed).shuffle(pool)       # 3. 同 seed shuffle
        for s in pool:
            yield s
        pool.clear()
```

### Pipeline position / 在 pipeline 中的位置

```
CrudeWebdataset → ShuffleBuffer → cook_crude_sample → encode_sample
    → LengthPoolSortDataset → BatchDataset → EpochizeDataset → LogSampleDataset
```

Inserted after `encode_sample` (where `total_len` / `tokens` are available) and before `BatchDataset`.

插在 `encode_sample` 之后（此时已有 `total_len` / `tokens`）、`BatchDataset` 之前。

Activated by: `--length-sort-pool-size N` (where N > 0).

通过 `--length-sort-pool-size N`（N > 0）激活。

## Why it accelerates training / 为什么能加速训练

### The problem / 问题

In multi-GPU data-parallel training, all ranks must synchronize at each step (gradient all-reduce). If different ranks process samples of very different lengths, fast ranks idle waiting for slow ranks.

多卡数据并行训练中，所有 rank 每步都要同步（梯度 all-reduce）。如果不同 rank 处理的 sample 长度差异很大，快的 rank 空等慢的 rank。

```
无 pool sort:
  Rank 0 step 100: 长度 200 → 0.5s    ┐
  Rank 1 step 100: 长度 5000 → 3.0s   ├→ 所有 rank 等 3.0s
  Rank 2 step 100: 长度 300 → 0.6s    ┘
```

### The solution / 解决方案

Sort + same-seed shuffle ensures all ranks yield samples of approximately the same length at the same time.

排序 + 同 seed shuffle 保证所有 rank 在同一时刻输出近似相同长度的 sample。

```
有 pool sort:
  Rank 0 step 100: 长度 ~200 → 0.5s   ┐
  Rank 1 step 100: 长度 ~200 → 0.5s   ├→ 所有 rank 等 0.5s
  Rank 2 step 100: 长度 ~200 → 0.5s   ┘
```

### Why it works — step by step / 为什么有效——逐步分析

#### Step 1: Sort aligns the i-th position across ranks / 排序对齐各 rank 第 i 个位置

Each rank's data comes from different shards of the same dataset, so length distributions are approximately identical. After sorting, the i-th position in each rank's pool corresponds to the i-th quantile of its length distribution. Similar distributions → similar quantiles → similar lengths.

各 rank 的数据来自同一数据集的不同 shard，长度分布近似相同。排序后，各 rank pool 中第 i 个位置对应各自长度分布的第 i 个分位数。分布相似 → 分位数相似 → 长度相似。

```
Rank 0 排序后: [100, 102, 105, 108, ..., 4998, 5000]
Rank 1 排序后: [101, 103, 106, 109, ..., 4999, 5001]
Rank 2 排序后: [ 99, 104, 107, 110, ..., 4997, 5002]
```

#### Step 2: Same seed preserves alignment after shuffle / 同 seed 保持 shuffle 后的对齐

`shuffle_seed = 42 + batch_idx` is identical across all ranks → same permutation applied to all pools. Since the i-th position had similar lengths, the shuffled output at the same position still has similar lengths.

`shuffle_seed = 42 + batch_idx` 对所有 rank 相同 → 相同排列应用于所有 pool。由于第 i 个位置的长度相似，shuffle 后同一位置的输出长度仍然相似。

```
permutation = [3, 0, 4, 1, 2]:
  Rank 0 输出: [108, 100, 5000, 102, 105]
  Rank 1 输出: [109, 101, 5001, 103, 106]
  Rank 2 输出: [110,  99, 5002, 104, 107]
  → 同一位置长度近似一致
```

### Why both sort AND shuffle are needed / 为什么排序和 shuffle 缺一不可

| 方案 | 效果 |
|---|---|
| 只 sort 不 shuffle | 所有 rank 先跑短 sample 后跑长 sample。前期 step 极快，后期 step 极慢。训练动态不稳定 |
| sort + shuffle | 各 step 长度随机但 rank 间一致。step 耗时平稳，rank 间同步开销小 |
| 不 sort 不 shuffle | 各 rank 同一 step 长度随机且不一致，快的等慢的 |

**Sort solves cross-rank synchronization. Shuffle solves temporal uniformity. Both are required.**

**排序解决跨 rank 同步问题，shuffle 解决时序均匀性问题。二者缺一不可。**

## pool_size tuning / pool_size 调优

| pool_size | 跨 rank 长度同步精度 | 内存开销 | 首个 pool 输出延迟 |
|---|---|---|---|
| 小（~100） | 较差，各 rank 分位数估计不准 | 低 | 低 |
| 中（~1000-10000） | 好，推荐范围 | 中 | 中 |
| 大（~全量） | 完美同步 | 高 | 高 |
| 极限 = dataset 大小 | 等价全局排序 | 不实际 | 不实际 |

**Larger pool_size → more accurate quantile estimation across ranks → better synchronization → less idle time.**

**pool_size 越大 → 各 rank 分位数估计越准 → 同步越好 → 空等越少。**

Rule of thumb: pool_size should be significantly larger than batch_size, ideally 10x-100x.

经验法则：pool_size 应远大于 batch_size，理想情况下 10x-100x。

## Multi-worker behavior (num_workers > 1) / 多 worker 行为

When `num_workers > 1`, each worker runs an independent `LengthPoolSortDataset` instance with its own pool.

当 `num_workers > 1` 时，每个 worker 运行独立的 `LengthPoolSortDataset` 实例，各自维护独立 pool。

- **Correctness**: No issue. Each worker sorts its own shard subset independently.
- **Synchronization effect**: Diluted. DataLoader interleaves outputs from multiple workers, partially disrupting the within-pool length ordering.
- **Recommendation**: `num_workers=1` gives the strongest synchronization effect. If `num_workers > 1` is needed for I/O throughput, increase `pool_size` proportionally.

- **正确性**：无问题。每个 worker 独立排序自己的 shard 子集。
- **同步效果**：被稀释。DataLoader 交替取多个 worker 的输出，部分打乱 pool 内的长度顺序。
- **建议**：`num_workers=1` 同步效果最强。如果需要 `num_workers > 1` 提升 I/O 吞吐，可按比例增大 `pool_size`。

## Checkpoint resume caveat / checkpoint 恢复注意事项

`save_state` / `restore_state` delegate directly to the upstream dataset. **The pool's internal state (accumulated but not yet yielded samples) is NOT saved.**

`save_state` / `restore_state` 直接委托给上游 dataset。**pool 内部状态（已累积但未 yield 的 sample）不会被保存。**

On resume, up to `pool_size - 1` samples may be re-ordered differently or skipped. This is generally negligible for large datasets.

恢复时，最多 `pool_size - 1` 个 sample 可能会被不同地排序或跳过。对于大数据集来说通常可以忽略。

## key_fn / 排序键

```python
key_fn=lambda s: getattr(s, "total_len", len(getattr(s, "tokens")))
```

Prefers `total_len` attribute (set by encode_sample), falls back to `len(tokens)`.

优先使用 `total_len` 属性（由 encode_sample 设置），回退到 `len(tokens)`。

## What to check during debugging / 调试时要检查什么

- Is `--length-sort-pool-size` actually set and > 0?
- What is the actual length distribution of the dataset? (High variance → more benefit from pool sort)
- Are all ranks receiving data from the same underlying dataset with similar length distributions?
- Is `num_workers` > 1 diluting the synchronization effect?
- After enabling, did step time variance across steps decrease?
- Is memory usage acceptable for the chosen `pool_size`?

- `--length-sort-pool-size` 是否真正设置且 > 0？
- 数据集的实际长度分布如何？（方差越大 → pool sort 收益越大）
- 所有 rank 是否从同一底层数据集接收长度分布相似的数据？
- `num_workers` > 1 是否稀释了同步效果？
- 启用后，各 step 的 step time 方差是否降低了？
- 所选 `pool_size` 的内存占用是否可接受？

## Expected outputs when using this skill / 使用本 skill 时的期望输出

When asked to analyze a training speed issue related to length sorting, return:

当被要求分析与长度排序相关的训练速度问题时，应返回：

1. Whether `LengthPoolSortDataset` is active and with what `pool_size`
2. The dataset's length distribution characteristics (high/low variance)
3. The relationship between `pool_size`, `batch_size`, and `num_workers`
4. Whether cross-rank synchronization is the bottleneck
5. Recommended `pool_size` adjustment if needed

1. `LengthPoolSortDataset` 是否激活，`pool_size` 是多少
2. 数据集的长度分布特征（高/低方差）
3. `pool_size`、`batch_size`、`num_workers` 之间的关系
4. 跨 rank 同步是否是瓶颈
5. 如需要，推荐的 `pool_size` 调整
