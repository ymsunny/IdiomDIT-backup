#!/bin/bash
# run_eval_detection.sh — 跑单个 model 的 Detection 评估（所有 6 个 lang_pair）
# Judge: gpt-4o-mini，读 detection/detection_inference.json，写 evaluation/detection_score.json
#
# 设计为可并行：多个 model 后台同时跑
#
# 用法：
#   bash run_eval_detection.sh Qwen3.5-9B
#   bash run_eval_detection.sh Llama-3.3-70B-Instruct
#   MAX_SAMPLES=5 bash run_eval_detection.sh Qwen3-8B   # 测试

set -e

MODEL=${1:?"用法: bash run_eval_detection.sh <MODEL>"}
LANG_PAIRS=(fa-en en-fa ko-en fi-en fr-en ja-en)

MAX_SAMPLES_FLAG=""
if [ -n "${MAX_SAMPLES}" ]; then MAX_SAMPLES_FLAG="--max-samples ${MAX_SAMPLES}"; fi

echo "============================================================"
echo "  [${MODEL}] Detection 评估"
echo "  Judge: gpt-4o-mini"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

for lp in "${LANG_PAIRS[@]}"; do
    echo "[${MODEL}] ${lp} ... $(date '+%H:%M:%S')"
    python evaluation/eval_detection.py \
        --lang-pair "$lp" \
        --model "$MODEL" \
        $MAX_SAMPLES_FLAG
done

echo "[${MODEL}] 完成. $(date '+%Y-%m-%d %H:%M:%S')"
