#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import json
import os
import sys
from copy import deepcopy
from os.path import dirname

import torch
from einops import rearrange
from safetensors.torch import load_file, save_file


SCRIPT_DIR = dirname(os.path.abspath(__file__))
sys.path.append(dirname(dirname(dirname(SCRIPT_DIR))))

from convert_checkpoint.arguments import parse_args
from convert_checkpoint.custom.llava_onevision2.util import (
    load_huggingface_checkpoint,
    load_megatron_checkpoint,
    load_megatron_checkpoint_tp_pp_ep,
    save_huggingface_checkpoint,
    save_megatron_checkpoint,
)


# Keys that hold the patch embedding weight (Conv2d in HF, Linear in mcore).
# The Conv2d weight is 4-D [O, C, H, W]; the Linear weight is 2-D [O, C*H*W].
# ColumnParallelLinear shards along dim 0 (output dimension).
PATCH_WEIGHT_KEY_MCORE = "vision_model.patch_embed.proj.weight"
PATCH_WEIGHT_KEY_HF = "visual.embeddings.patch_embedding.weight"


args = parse_args()
name_map = {}  # megatron -> huggingface
with open(args.common_config_path, "r", encoding="utf-8") as f:
    name_map = json.loads(f.read())


def _conv2d_to_linear(weight: torch.Tensor) -> torch.Tensor:
    """Reshape Conv2d weight [O, C, H, W] -> Linear weight [O, C*H*W]."""
    if weight.dim() == 4:
        return weight.reshape(weight.shape[0], -1)
    return weight


def _linear_to_conv2d(weight: torch.Tensor, in_channels: int = 3, patch_size: int | None = None) -> torch.Tensor:
    """Reshape Linear weight [O, C*H*W] -> Conv2d weight [O, C, H, W].

    If ``patch_size`` is None, infer it from the flattened input dim assuming
    a square patch: H = W = sqrt(C*H*W / in_channels).
    """
    if weight.dim() == 2:
        if patch_size is None:
            flat = weight.shape[1]
            assert flat % in_channels == 0, f"flat dim {flat} not divisible by in_channels {in_channels}"
            spatial = flat // in_channels
            patch_size = int(round(spatial ** 0.5))
            assert patch_size * patch_size == spatial, (
                f"cannot infer square patch_size from flat dim {flat} / in_channels {in_channels}"
            )
        return weight.reshape(weight.shape[0], in_channels, patch_size, patch_size)
    return weight


def _shard_along_dim0(weight: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    """Shard a weight tensor along dim 0 for the given TP rank."""
    if tp_size <= 1:
        return weight
    shard_size = weight.shape[0] // tp_size
    return weight[tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()


def _gather_along_dim0(shards: list) -> torch.Tensor:
    """Gather TP shards back into a full weight tensor along dim 0."""
    return torch.cat(shards, dim=0)


def _shards_are_replicated(shards: list) -> bool:
    """Detect whether a list of per-TP-rank tensors are identical replicas.

    The patch embedding has two valid mcore layouts:
      1. ``ParallelPatchEmbed`` (``PATCH_EMBED_TYPE=TP_LINEAR``):
         Each TP rank stores a distinct shard of shape
         ``[embed_dim/tp, C, H, W]`` (or 2-D ``[embed_dim/tp, C*H*W]``).
         These shards must be concatenated along dim 0.
      2. ``PatchEmbed`` (``CONV2D``) or ``TorchLinearPatchEmbed`` (``LINEAR``):
         Each TP rank stores the *full* replicated weight of shape
         ``[embed_dim, C, H, W]`` (or 2-D ``[embed_dim, C*H*W]``).
         These must NOT be concatenated; we just take rank 0.

    We detect (2) by checking that all shards have the same shape AND
    bitwise-equal contents (replicas saved by every TP rank).
    """
    if len(shards) <= 1:
        return False
    ref = shards[0]
    for s in shards[1:]:
        if s.shape != ref.shape:
            return False
        if not torch.equal(s, ref):
            return False
    return True


def _merge_patch_shards(shards: list) -> torch.Tensor:
    """Merge per-TP-rank patch_embed shards into one full weight.

    Auto-detects replicated (CONV2D / LINEAR patch_embed) vs TP-sharded
    (TP_LINEAR / ParallelPatchEmbed) layouts.
    """
    if len(shards) == 1:
        return shards[0]
    if _shards_are_replicated(shards):
        print(
            f" > patch_embed shards are REPLICATED across TP={len(shards)} "
            f"(shape {list(shards[0].shape)}); using rank 0 copy"
        )
        return shards[0]
    print(
        f" > patch_embed shards are TP-SHARDED across TP={len(shards)} "
        f"(per-rank shape {list(shards[0].shape)}); concatenating along dim 0"
    )
    return _gather_along_dim0(shards)


def _get_non_ep_model_source(state_dict):
    """Return model dict from PP rank 0, TP rank 0.

    `load_megatron_checkpoint` returns either:
      - 1D layout (TP only):  state_dict[tp]            -> dict with "model"
      - 2D layout (TP+PP):    state_dict[pp][tp]        -> dict with "model"
    The vision patch embedding always lives on PP rank 0, so we take [0][0].
    """
    first = state_dict[0]
    if isinstance(first, dict):
        # 1D layout: state_dict[tp_rank] is a ckpt dict -> tp=0
        return first["model"]
    first_rank = first[0]
    if isinstance(first_rank, dict):
        # 2D layout: state_dict[pp_rank][tp_rank] -> pp=0, tp=0
        return first_rank["model"]
    raise TypeError("Unsupported non-EP checkpoint structure")


def _get_non_ep_tp_shard(state_dict, tp_rank):
    """Return model dict at PP rank 0 for a specific TP rank.

    Handles both 1D layout (state_dict[tp]) and 2D layout (state_dict[pp][tp]).
    Patch embedding always resides on PP rank 0.
    """
    first = state_dict[0]
    if isinstance(first, dict):
        # 1D layout (PP=1): state_dict is indexed directly by tp_rank
        return state_dict[tp_rank]["model"]
    # 2D layout (PP>1): state_dict[pp][tp]; patch_embed is on pp=0
    return state_dict[0][tp_rank]["model"]


def _get_non_ep_tp_size(state_dict):
    """Return the TP size of a non-EP state_dict (1D or 2D layout)."""
    first = state_dict[0]
    if isinstance(first, dict):
        # 1D: outer dim IS tp
        return len(state_dict)
    # 2D: state_dict[pp][tp], TP size = len(state_dict[0])
    return len(first)


if (args.load_platform, args.save_platform) == ("mcore", "huggingface"):
    """ megatron to huggingface """
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert vision patch from Megatron Core to HuggingFace ======")
    target = {}
    if args.expert_parallel_size is not None:
        state_dict = load_megatron_checkpoint_tp_pp_ep(args.load_ckpt_path)
        source = state_dict[0][0][0]["model"]
    else:
        state_dict = load_megatron_checkpoint(args.load_ckpt_path)
        source = _get_non_ep_model_source(state_dict)

    tp = args.tensor_model_parallel_size

    for k1, k2 in name_map.items():
        if k1 == PATCH_WEIGHT_KEY_MCORE:
            # Gather TP shards (or detect replicas) and convert Linear 2-D back to Conv2d 4-D for HF.
            if tp > 1:
                shards = []
                if args.expert_parallel_size is not None:
                    for tp_rank in range(tp):
                        shards.append(state_dict[tp_rank][0][0]["model"][k1])
                else:
                    for tp_rank in range(tp):
                        shards.append(_get_non_ep_tp_shard(state_dict, tp_rank)[k1])
                full_weight = _merge_patch_shards(shards)
            else:
                full_weight = source[k1]

            # Convert from Linear 2-D to Conv2d 4-D
            target[k2] = _linear_to_conv2d(full_weight)
            print(f" > {k1} -> {k2}  (merged TP shards, reshaped to Conv2d {list(target[k2].shape)})")
        else:
            target[k2] = source[k1]
    save_huggingface_checkpoint(target, args.save_ckpt_path)

elif (args.load_platform, args.save_platform) == ("huggingface", "mcore"):
    """ huggingface to megatron """
    print(" ====== convert vision patch from HuggingFace to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    source = load_huggingface_checkpoint(args.load_ckpt_path)

    # Build per-TP-rank state dicts with proper sharding.
    state_dict = []
    for tp_rank in range(tp):
        target = {}
        for k1, k2 in name_map.items():
            if k1 == PATCH_WEIGHT_KEY_MCORE:
                # Convert Conv2d 4-D -> Linear 2-D, then shard for this TP rank.
                full_weight = _conv2d_to_linear(source[k2])
                target[k1] = _shard_along_dim0(full_weight, tp_rank, tp)
                if tp_rank == 0:
                    print(
                        f" > {k1}  (Conv2d {list(source[k2].shape)} -> Linear {list(full_weight.shape)} "
                        f"-> TP shard {list(target[k1].shape)})"
                    )
            else:
                target[k1] = source[k2]
                if tp_rank == 0:
                    print(f" > {k1}")
        # TE 2.2 BasicOperation.set_extra_state expects a torch.Tensor (calls .numel()).
        target["vision_model.pre_layernorm._extra_state"] = torch.tensor([])
        state_dict.append({"model": target})
    save_megatron_checkpoint(state_dict, os.path.join(args.save_ckpt_path, "release"))

elif (args.load_platform, args.save_platform) == ("mcore", "mcore"):
    """ megatron to megatron """
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)
    print(" ====== convert vision patch from Megatron Core to Megatron Core ======")
    tp = args.tensor_model_parallel_size
    if args.expert_parallel_size is not None:
        state_dict = load_megatron_checkpoint_tp_pp_ep(args.load_ckpt_path)
        source = state_dict[0][0][0]["model"]
    else:
        state_dict = load_megatron_checkpoint(args.load_ckpt_path)
        source = _get_non_ep_model_source(state_dict)

    # First, gather the full patch weight from the source (may be sharded).
    if PATCH_WEIGHT_KEY_MCORE in source:
        if args.expert_parallel_size is not None:
            source_tp = len(state_dict)
        else:
            source_tp = _get_non_ep_tp_size(state_dict)
        if source_tp > 1:
            shards = []
            if args.expert_parallel_size is not None:
                for tp_rank in range(source_tp):
                    shards.append(state_dict[tp_rank][0][0]["model"][PATCH_WEIGHT_KEY_MCORE])
            else:
                for tp_rank in range(source_tp):
                    shards.append(_get_non_ep_tp_shard(state_dict, tp_rank)[PATCH_WEIGHT_KEY_MCORE])
            full_patch_weight = _merge_patch_shards(shards)
        else:
            full_patch_weight = source[PATCH_WEIGHT_KEY_MCORE]
        # Handle old Conv2d format
        full_patch_weight = _conv2d_to_linear(full_patch_weight)
    else:
        full_patch_weight = None

    # Build per-TP-rank state dicts with proper sharding.
    target_state_dict = []
    for tp_rank in range(tp):
        target = {}
        for k in source.keys():
            if k == PATCH_WEIGHT_KEY_MCORE and full_patch_weight is not None:
                target[k] = _shard_along_dim0(full_patch_weight, tp_rank, tp)
                if tp_rank == 0:
                    print(f" > {k}  (reshard to TP={tp}: {list(full_patch_weight.shape)} -> {list(target[k].shape)})")
            else:
                target[k] = source[k]
                if tp_rank == 0:
                    print(f" > {k}")
        target_state_dict.append({"model": target})
    save_megatron_checkpoint(target_state_dict, os.path.join(args.save_ckpt_path, "release"))
else:
    raise NotImplementedError
