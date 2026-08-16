#!/bin/bash
# run_eval_interpretation.sh — 跑单个 model 的 Interpretation 评估（所有 6 个 lang_pair）
# Judge: gpt-4o-mini
#
# 用法：
#   bash run_eval_interpretation.sh Qwen3.5-9B
#   bash run_eval_interpretation.sh Llama-3.3-70B-Instruct
#   MAX_SAMPLES=5 bash run_eval_interpretation.sh Qwen3-8B   # 测试

set -e

MODEL=${1:?"用法: bash run_eval_interpretation.sh <MODEL>"}
LANG_PAIRS=(fa-en en-fa ko-en fi-en fr-en ja-en)

MAX_SAMPLES_FLAG=""
if [ -n "${MAX_SAMPLES}" ]; then MAX_SAMPLES_FLAG="--max-samples ${MAX_SAMPLES}"; fi

echo "============================================================"
echo "  [${MODEL}] Interpretation 评估"
echo "  Judge: gpt-4o-mini"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for lp in "${LANG_PAIRS[@]}"; do
    echo "[${MODEL}] ${lp} ... $(date '+%H:%M:%S')"
    python evaluation/eval_interpretation.py \
        --lang-pair "$lp" \
        --model "$MODEL" \
        $MAX_SAMPLES_FLAG
done

echo "[${MODEL}] 完成. $(date '+%Y-%m-%d %H:%M:%S')"
