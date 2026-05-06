#!/usr/bin/env bash
set -euo pipefail

# Phase 3 A2: peak RSS comparison new vs phase1-baseline tag.
# Auto-skips when F2-dense env not set.

if [[ -z "${OV2_REAL_FIXTURE:-}" ]]; then
    echo "SKIP: F2-dense unavailable (OV2_REAL_FIXTURE unset)"
    exit 0
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VIT="${OV2_F2_VIT:-/ov2/pretrain_models/onevision-encoder-large}"
LLM="${OV2_F2_LLM:-/ov2/pretrain_models/Qwen3-1.7B-Base}"
PROC="${OV2_F2_PROCESSOR:-/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct}"

for p in "$VIT" "$LLM" "$PROC"; do
    if [[ ! -e "$p" ]]; then
        echo "SKIP: missing $p"
        exit 0
    fi
done

OUT_NEW="${TMPDIR:-/tmp}/ov2_p3_rss_new"
OUT_OLD="${TMPDIR:-/tmp}/ov2_p3_rss_old"
LOG_NEW="${TMPDIR:-/tmp}/ov2_p3_rss_new.log"
LOG_OLD="${TMPDIR:-/tmp}/ov2_p3_rss_old.log"

run_new() {
    rm -rf "$OUT_NEW"
    /usr/bin/time -v env PYTHONPATH="$REPO/transformers_impl" \
        python -m transformers_impl.merge_ov2 merge \
            --variant dense --vit "$VIT" --llm "$LLM" --processor "$PROC" \
            --out "$OUT_NEW" \
            --validate-skip vit --validate-skip llm --validate-skip e2e \
        2> "$LOG_NEW"
}

run_old() {
    rm -rf "$OUT_OLD"
    /usr/bin/time -v python "$REPO/tests/legacy/merge_ov2_dense_legacy.py" \
        --vit_path "$VIT" --llm_path "$LLM" --processor_path "$PROC" \
        --output_path "$OUT_OLD" --skip_validation \
        2> "$LOG_OLD"
}

parse_rss() {
    grep "Maximum resident set size" "$1" | awk '{print $NF}'
}

run_new
run_old

NEW_RSS=$(parse_rss "$LOG_NEW")
OLD_RSS=$(parse_rss "$LOG_OLD")
RATIO=$(awk -v n="$NEW_RSS" -v o="$OLD_RSS" 'BEGIN { printf "%.3f", n / o }')

echo "new_rss_kb=$NEW_RSS"
echo "old_rss_kb=$OLD_RSS"
echo "ratio=$RATIO"

awk -v r="$RATIO" 'BEGIN { exit !(r <= 0.70) }' || {
    echo "FAIL: ratio $RATIO > 0.70"
    exit 1
}
echo "PASS: ratio $RATIO <= 0.70 (>=30% reduction)"
