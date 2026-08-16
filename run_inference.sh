#!/bin/bash
# 推理（GPU，在服务器上跑）
#
# 每个模型加载 1 次，跑完所有语言对的 detection/interpretation/translation
# 模型 size 自动从 config.py 的 LARGE_MODELS 判断：
#   - small  → 1 张 GPU，多模型并行
#   - large  → 全部 4 卡 device_map=auto，串行
#
# 用法：
#   bash run_inference.sh                                            # 默认全部模型 + 全部语言对
#   bash run_inference.sh fi-en ko-en                                # 指定语言对
#   MODEL=Llama-3.3-70B-Instruct bash run_inference.sh fi-en         # 指定单个模型
#   MODELS="Qwen3.5-9B Qwen3-8B" bash run_inference.sh               # 指定多个模型（自动按 large/small 分桶）
#   MAX_SAMPLES=10 bash run_inference.sh fi-en                       # 只跑前 10 条（测试）
set -e
mkdir -p results

# ==================== 解析参数 ====================

if [ $# -gt 0 ]; then
    LANG_PAIRS=("$@")
else
    LANG_PAIRS=(fa-en en-fa ko-en fi-en fr-en ja-en)
fi

# 默认模型集合（用户可通过 MODEL / MODELS 覆盖）
DEFAULT_MODELS=(Qwen3.5-4B Qwen3-4B Qwen3.5-9B Qwen3-8B Llama-3.3-70B-Instruct)

# 优先级: MODEL (单个) > MODELS (列表) > DEFAULT_MODELS
if [ -n "${MODEL}" ]; then
    ALL_MODELS=("${MODEL}")
elif [ -n "${MODELS}" ]; then
    ALL_MODELS=(${MODELS})
else
    ALL_MODELS=("${DEFAULT_MODELS[@]}")
fi

# 从 config.py 读取 LARGE_MODELS 集合
LARGE_SET=$(python -c "from config import LARGE_MODELS; print(' '.join(LARGE_MODELS))")

# 自动分桶
SMALL_MODELS=()
BIG_MODELS=()
for m in "${ALL_MODELS[@]}"; do
    is_large=0
    for big in $LARGE_SET; do
        if [ "$m" = "$big" ]; then is_large=1; break; fi
    done
    if [ $is_large -eq 1 ]; then BIG_MODELS+=("$m"); else SMALL_MODELS+=("$m"); fi
done

MAX_SAMPLES_FLAG=""
if [ -n "${MAX_SAMPLES}" ]; then MAX_SAMPLES_FLAG="--max-samples ${MAX_SAMPLES}"; fi

# ==================== 执行 ====================

echo "============================================================"
echo "  推理（模型加载 1 次，跑完所有语言对）"
echo "  语言对 (${#LANG_PAIRS[@]}): ${LANG_PAIRS[*]}"
echo "  小模型 (${#SMALL_MODELS[@]}): ${SMALL_MODELS[*]:-<none>}"
echo "  大模型 (${#BIG_MODELS[@]}): ${BIG_MODELS[*]:-<none>}"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# —— 小模型：并行，每个独占 1 张 GPU ——
if [ ${#SMALL_MODELS[@]} -gt 0 ]; then
    echo "  [小模型] 并行启动: ${SMALL_MODELS[*]}"
    gpu_id=0
    pids=()
    for model in "${SMALL_MODELS[@]}"; do
        echo "    ${model} → GPU ${gpu_id}"
        CUDA_VISIBLE_DEVICES=$gpu_id python run_all_inference.py \
            --lang-pairs "${LANG_PAIRS[@]}" --model "$model" \
            --all-prompts $MAX_SAMPLES_FLAG &
        pids+=($!)
        gpu_id=$((gpu_id + 1))
    done
    echo "  等待小模型完成..."
    for pid in "${pids[@]}"; do wait "$pid"; done
    echo "  小模型完成. $(date '+%Y-%m-%d %H:%M:%S')"
fi

# —— 大模型：串行，独占全部 4 卡 ——
if [ ${#BIG_MODELS[@]} -gt 0 ]; then
    echo "  [大模型] 串行启动（全部 4 卡）: ${BIG_MODELS[*]}"
    for model in "${BIG_MODELS[@]}"; do
        echo "    ${model} → GPU 0,1,2,3"
        CUDA_VISIBLE_DEVICES=0,1,2,3 python run_all_inference.py \
            --lang-pairs "${LANG_PAIRS[@]}" --model "$model" \
            --all-prompts $MAX_SAMPLES_FLAG
        echo "    ${model} 完成. $(date '+%Y-%m-%d %H:%M:%S')"
    done
fi

echo ""
echo "============================================================"
echo "  全部推理完成! $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
