#!/usr/bin/env bash
# v4 GPT-5.2 PEAK-LAYER RANDOM-DIRECTION control — Qwen3.5-9B, mixed prompts, single peak layer per pair.
#
# 配合 run_v4_gpt52_peak_qwen9b.sh, 建立 peak-layer 下的 random-direction null band.
# 同 all-layer random control (Finding 8) 的 3 seed 设置.
#
# 用法(GPU 服务器,3 seed 三 GPU 并行):
#   cd IdiomDIT-LTB && git pull
#   bash mechanistic/run_v4_gpt52_peak_random_qwen9b.sh 42 0 > log_peak_rand_s42.txt 2>&1 &
#   bash mechanistic/run_v4_gpt52_peak_random_qwen9b.sh 43 1 > log_peak_rand_s43.txt 2>&1 &
#   bash mechanistic/run_v4_gpt52_peak_random_qwen9b.sh 44 2 > log_peak_rand_s44.txt 2>&1 &
#   wait
#
# 只想跑生成(判审留本地):
#   SKIP_JUDGE=1 bash mechanistic/run_v4_gpt52_peak_random_qwen9b.sh <SEED> <GPU_ID>
#
# 输出:
#   results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52/ablation_L<peak>_dir_ablate_allprompts__rand_s<S>.json
#   results/<pair>/Qwen3.5-9B/evaluation/ablation_L<peak>_dir_ablate_allprompts__rand_s<S>_ablation_score.json

MODEL=Qwen3.5-9B
EXP=v4_gpt52
S=${1:?usage: run_v4_gpt52_peak_random_qwen9b.sh <SEED> <GPU_ID>}
GPU=${2:-0}

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
  echo "########  $LP  peak_L=$L  seed=$S  gpu=$GPU  (random-direction)  ########"

  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
    --lang-pair "$LP" --model "$MODEL" --ablate-layers "$L" --group-b strict --exp-name "$EXP" \
    --random-direction --seed "$S"

  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  OUT="ablation_L${L}_dir_ablate_allprompts__rand_s${S}"
  ABL="$MECH/${OUT}__for_eval_ablation.json"
  echo "  random-ablation file: $ABL"

  if [ -n "${SKIP_JUDGE:-}" ]; then
    echo "  step3 SKIPPED (SKIP_JUDGE=1)"
  else
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$ABL" \
      --output-prefix "${OUT}_ablation"
  fi
done

echo ""
echo "==== DONE seed=$S (gpu=$GPU). ===="
