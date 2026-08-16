#!/usr/bin/env bash
# v4 GPT-5.2 PEAK-LAYER ablation — Qwen3.5-9B, mixed prompts (all 4), single peak layer per pair.
#
# 目的: 回应审稿人 "all-layer ablation is coarse; try peak probing layer".
# 保持与 Table 5 (Finding 8) 唯一的差异 = 消融层数(all vs peak),其它不动:
#   - Prompt: all 4 pooled (同 Finding 8)
#   - Group A: know-but-error strict (同 Finding 8)
#   - Judge: v4 gpt-5.2
#   - Peak layer 取 leakage-safe grouped_cv 峰,per pair.
#
# 前置: v4_gpt52 pipeline 的 step0-1 已跑过(group_known_lte.json + probing_balanced_strict.json 就位).
# 本脚本只跑 step2 (peak-layer ablation) + step3 (judge).
#
# 用法(GPU 服务器):
#   cd /sym/IdiomDIT-LTB && git pull
#   bash mechanistic/run_v4_gpt52_peak_qwen9b.sh [GPU_ID]                # 一步到位
#   SKIP_JUDGE=1 bash mechanistic/run_v4_gpt52_peak_qwen9b.sh [GPU_ID]   # 只跑生成,判审留给本地
#
# Peak grouped_cv layers (from probing_balanced_strict.json, 2026-07-11):
#   en-fa=15  fa-en=16  fi-en=17  fr-en=31  ja-en=11  ko-en=11
#
# 输出:
#   results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52/ablation_L<peak>_dir_ablate_allprompts.json
#   results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52/ablation_L<peak>_dir_ablate_allprompts__for_eval_{baseline,ablation}.json
#   results/<pair>/Qwen3.5-9B/evaluation/ablation_v4gpt52_peak_{baseline,ablation}_score.json  (step3 判审后)

MODEL=Qwen3.5-9B
EXP=v4_gpt52
GPU=${1:-0}

# 每方向的 peak grouped_cv layer
declare -A PEAK
PEAK[en-fa]=15
PEAK[fa-en]=16
PEAK[fi-en]=17
PEAK[fr-en]=31
PEAK[ja-en]=11
PEAK[ko-en]=11

PAIRS="en-fa fa-en ja-en fr-en fi-en ko-en"

for LP in $PAIRS; do
  L=${PEAK[$LP]}
  echo ""
  echo "################  $LP  peak_L=$L  ################"

  echo "---- step2: peak-layer direction ablation ($LP L=$L) ----"
  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
    --lang-pair "$LP" --model "$MODEL" --ablate-layers "$L" --group-b strict --exp-name "$EXP"

  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  BASE="$MECH/ablation_L${L}_dir_ablate_allprompts__for_eval_baseline.json"
  ABL="$MECH/ablation_L${L}_dir_ablate_allprompts__for_eval_ablation.json"
  echo "  baseline file: $BASE"
  echo "  ablation file: $ABL"

  if [ -n "${SKIP_JUDGE:-}" ]; then
    echo "---- step3: SKIPPED (SKIP_JUDGE=1); rsync to local and run judge there. ----"
  else
    echo "---- step3: re-LTE with gpt-5.2 ($LP L=$L) ----"
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$BASE" \
      --output-prefix "ablation_v4gpt52_peak_baseline"
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$ABL" \
      --output-prefix "ablation_v4gpt52_peak_ablation"
  fi
done

echo ""
echo "==== DONE. If SKIP_JUDGE=1, sync results/ to local and run mechanistic/run_v4_gpt52_peak_judge_local.sh ===="
