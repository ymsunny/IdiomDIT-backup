#!/usr/bin/env bash
# run_v4_mechanistic.sh — 用 v4+gpt-5.2 的 LTE 标签，重做 probing + ablation（隔离版）
#
# 一切输出进 mechanistic/v4_gpt52/ 和 evaluation/v4gpt52_*，旧的 old-judge 结果一律不碰。
# 阶段: 1) extract(CPU)  2) probing(GPU)  3) ablation 生成(GPU)  4) ablation 评估(API gpt-5.2)+aggregate
#
# 用法:
#   bash run_v4_mechanistic.sh [MODEL] [GPU] [lang1 lang2 ...]
#   bash run_v4_mechanistic.sh Qwen3.5-9B 0 fi-en fr-en
#   nohup bash run_v4_mechanistic.sh Qwen3.5-9B 0 fi-en fr-en > log_v4_mech.txt 2>&1 &
set -u
MODEL=${1:-Qwen3.5-9B}; GPU=${2:-0}; shift 2 2>/dev/null || true
LANGS=("$@"); [ ${#LANGS[@]} -eq 0 ] && LANGS=(fi-en fr-en)
EXP=v4_gpt52
LTE_PREFIX=translation_lte_v4_gpt52      # v4 prompt + gpt-5.2 的 LTE 评估结果
ABL=ablation_Lall_dir_ablate_allprompts__gb_superstrict
echo "=== v4 mechanistic redo | model=$MODEL | gpu=$GPU | dirs=${LANGS[*]} | exp=$EXP ==="

# ---- Stage 1: 抽组 (CPU, 用 v4+gpt-5.2 的 LTE) ----
for lp in "${LANGS[@]}"; do
    echo ""; echo ">>> [1/4 extract] $lp"
    python mechanistic/extract_known_lte_groups.py --lang-pair "$lp" --model "$MODEL" \
        --lte-prefix "$LTE_PREFIX" --exp-name "$EXP"
done

# ---- Stage 2: probing (GPU; 新组要重抽 hidden state) ----
for lp in "${LANGS[@]}"; do
    for gb in group_known_non_lte_strict.json group_known_non_lte_superstrict.json; do
        echo ""; echo ">>> [2/4 probe] $lp / $gb"
        CUDA_VISIBLE_DEVICES=$GPU python mechanistic/probe_ltb_by_hidden_state_balanced.py \
            --lang-pair "$lp" --model "$MODEL" --exp-name "$EXP" --group-b-file "$gb"
    done
done

# ---- Stage 3: ablation 生成 (GPU, super-strict 方向) ----
for lp in "${LANGS[@]}"; do
    echo ""; echo ">>> [3/4 ablation-gen] $lp"
    CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
        --lang-pair "$lp" --model "$MODEL" --ablate-layers all --mode direction_ablate \
        --exp-name "$EXP" --group-b superstrict
done

# ---- Stage 4: ablation 评估 (API, v4+gpt-5.2) + aggregate ----
for lp in "${LANGS[@]}"; do
    EXPDIR="results/$lp/$MODEL/mechanistic/$EXP"
    for v in baseline ablation; do
        echo ""; echo ">>> [4/4 eval] $lp / $v"
        python evaluation/eval_translation_lte_v4.py --lang-pair "$lp" --model "$MODEL" --no-confirm \
            --judge-model gpt-5.2 \
            --translation-input "$EXPDIR/${ABL}__for_eval_${v}.json" \
            --output-prefix "v4gpt52_${ABL}_${v}"
    done
done
echo ""; echo ">>> [aggregate]"
python analysis/aggregate_ablation.py --model "$MODEL" --ablation-name "v4gpt52_${ABL}"

echo ""; echo "=== done ==="
echo "probing 报告: results/*/$MODEL/mechanistic/$EXP/probing_balanced*_report.txt"
echo "ablation 聚合: 见上方 aggregate 输出 (前缀 v4gpt52_)"
