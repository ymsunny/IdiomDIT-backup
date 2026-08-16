#!/usr/bin/env bash
# v4 GPT-5.2 NAIVE-PEAK ablation — Qwen3.5-9B, mixed prompts, single layer per pair.
#
# 回应审稿 W4 "ablate at the peak probing layer":
# 论文自己的 "peak probing layer" 定义 = Table 3 印的 Peak L 列(naive
# balanced_subsample 峰,Figure 5 也用同一定义),所以本实验在这些层消融:
#   en-fa=27  fa-en=8  fr-en=20  fi-en=27  ko-en=16  ja-en=21
#
# 与早前 grouped-CV 峰实验(run_v4_gpt52_peak_qwen9b.sh, L15/16/17/31/11/11)
# 完全独立:六个方向的层号互不相同,生成文件天然不撞名;判审前缀用
# ablation_v4gpt52_naivepeak_* 与 ablation_v4gpt52_peak_* 区分。
#
# 除层数外与 Table 5 (Finding 8) 唯一差异 = 消融层(naive peak vs all):
#   - Prompt: all 4 pooled(同 Finding 8)
#   - Group A: know-but-error strict(同 Finding 8)
#   - 方向: probing_balanced_strict.json 的 direction_balanced(同 Finding 8)
#   - Judge: v4 gpt-5.2
#
# 前置: v4_gpt52 pipeline 的 step0-1 已跑过(group_known_lte.json +
#       probing_balanced_strict.json 就位;上次 peak 实验已验证)。
#
# 用法(GPU 服务器):
#   cd /sym/IdiomDIT-LTB && git pull
#   SKIP_JUDGE=1 nohup bash mechanistic/run_v4_gpt52_naivepeak_qwen9b.sh 0 > log_naivepeak_ltd.txt 2>&1 &
#
# 输出:
#   results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52/ablation_L<naive_peak>_dir_ablate_allprompts.json
#   results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52/ablation_L<naive_peak>_dir_ablate_allprompts__for_eval_{baseline,ablation}.json
#   results/<pair>/Qwen3.5-9B/evaluation/ablation_v4gpt52_naivepeak_{baseline,ablation}_score.json  (step3 判审后)

MODEL=Qwen3.5-9B
EXP=v4_gpt52
GPU=${1:-0}

# 每方向的 naive peak layer(= paper Table 3 "Peak L" 列)
declare -A PEAK
PEAK[en-fa]=27
PEAK[fa-en]=8
PEAK[fi-en]=27
PEAK[fr-en]=20
PEAK[ja-en]=21
PEAK[ko-en]=16

PAIRS="en-fa fa-en ja-en fr-en fi-en ko-en"

for LP in $PAIRS; do
  L=${PEAK[$LP]}
  echo ""
  echo "################  $LP  naive_peak_L=$L  ################"

  echo "---- step2: naive-peak direction ablation ($LP L=$L) ----"
  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
    --lang-pair "$LP" --model "$MODEL" --ablate-layers "$L" --group-b strict --exp-name "$EXP"

  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  BASE="$MECH/ablation_L${L}_dir_ablate_allprompts__for_eval_baseline.json"
  ABL="$MECH/ablation_L${L}_dir_ablate_allprompts__for_eval_ablation.json"
  echo "  baseline file: $BASE"
  echo "  ablation file: $ABL"

  if [ -n "${SKIP_JUDGE:-}" ]; then
    echo "---- step3: SKIPPED (SKIP_JUDGE=1); sync to local and run judge there. ----"
  else
    echo "---- step3: re-LTE with gpt-5.2 ($LP L=$L) ----"
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$BASE" \
      --output-prefix "ablation_v4gpt52_naivepeak_baseline"
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$ABL" \
      --output-prefix "ablation_v4gpt52_naivepeak_ablation"
  fi
done

echo ""
echo "==== DONE. If SKIP_JUDGE=1, sync results/ to local and run mechanistic/run_v4_gpt52_naivepeak_judge_local.sh ===="
