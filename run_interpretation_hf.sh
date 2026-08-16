#!/bin/bash
# Interpretation 推理（HuggingFace，4 模型各占 1 GPU 并行）
#
# 优先跑 fi-en fr-en Qwen3.5-9B，完成后 4 模型并行跑所有语言对
# 已完成的语言对自动跳过
#
# 用法：
#   bash run_interpretation_hf.sh
#   MAX_SAMPLES=5 bash run_interpretation_hf.sh   # 测试

set -e

ALL_LANG_PAIRS=(fa-en en-fa ko-en fi-en fr-en ja-en)
MODELS=(Qwen3.5-9B Qwen3-8B Qwen3.5-4B Qwen3-4B)

MAX_SAMPLES_FLAG=""
if [ -n "${MAX_SAMPLES}" ]; then MAX_SAMPLES_FLAG="--max-samples ${MAX_SAMPLES}"; fi

echo "============================================================"
echo "  Interpretation 推理（HuggingFace，4 模型并行）"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# === 优先：fi-en fr-en Qwen3.5-9B（GPU 0，等待完成）===
echo ""
echo "  [优先] Qwen3.5-9B: fi-en fr-en → GPU 0"
CUDA_VISIBLE_DEVICES=0 python run_all_inference.py \
    --lang-pairs fi-en fr-en \
    --model Qwen3.5-9B \
    --steps 2 $MAX_SAMPLES_FLAG
echo "  [优先] 完成. $(date '+%Y-%m-%d %H:%M:%S')"

# === 全量：4 模型各占 1 GPU 并行（已完成自动跳过）===
echo ""
echo "  [全量] 4 模型并行启动"
gpu_id=0
pids=()
for model in "${MODELS[@]}"; do
    echo "    ${model} → GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES=$gpu_id python run_all_inference.py \
        --lang-pairs "${ALL_LANG_PAIRS[@]}" \
        --model "$model" \
        --steps 2 $MAX_SAMPLES_FLAG &
    pids+=($!)
    gpu_id=$((gpu_id + 1))
done

echo "  等待全量完成..."
for pid in "${pids[@]}"; do wait "$pid"; done

echo ""
echo "============================================================"
echo "  全部完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
