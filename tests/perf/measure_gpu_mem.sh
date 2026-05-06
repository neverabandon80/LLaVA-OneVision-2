#!/usr/bin/env bash
set -euo pipefail

# Phase 4 A3: peak GPU memory of LLM validator vs phase1-baseline.
# Auto-skips when F2-dense env unset or no GPU.

if [[ -z "${OV2_REAL_FIXTURE:-}" ]]; then
    echo "SKIP: F2-dense unavailable (OV2_REAL_FIXTURE unset)"; exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "SKIP: no GPU"; exit 0
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LLM="${OV2_F2_LLM:-/ov2/pretrain_models/Qwen3-1.7B-Base}"
[[ -e "$LLM" ]] || { echo "SKIP: missing $LLM"; exit 0; }

LOG=$(mktemp)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$LOG" &
SMI_PID=$!

env PYTHONPATH="$REPO/transformers_impl" python -c "
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = AutoModelForCausalLM.from_pretrained('$LLM', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to('cuda:0').eval()
t = AutoTokenizer.from_pretrained('$LLM', use_fast=True)
i = t('Hello, my dog is cute', return_tensors='pt').to('cuda:0')
with torch.no_grad():
    _ = m(**i).logits
"

kill "$SMI_PID" 2>/dev/null || true
wait "$SMI_PID" 2>/dev/null || true

PEAK=$(sort -n "$LOG" | tail -1)
echo "peak_gpu_mib=$PEAK"
rm -f "$LOG"
