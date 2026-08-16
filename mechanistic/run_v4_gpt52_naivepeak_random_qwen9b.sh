#!/usr/bin/env bash
# v4 GPT-5.2 NAIVE-PEAK RANDOM-DIRECTION control — Qwen3.5-9B, mixed prompts, single naive-peak layer per pair.
#
# 配合 run_v4_gpt52_naivepeak_qwen9b.sh,建立 naive-peak(Table 3 Peak L)下的
# random-direction null band。同 all-layer random control (Finding 8) 的 3 seed 设置。
#
# Naive peak layers(= paper Table 3 "Peak L" 列;区别于 grouped-CV 峰 15/16/17/31/11/11):
#   en-fa=27  fa-en=8  fr-en=20  fi-en=27  ko-en=16  ja-en=21
# 层号与 grouped-CV 峰实验互不相同 → 输出文件天然不撞名。
#
# 用法(GPU 服务器,单卡串行 3 seed):
#   cd /sym/IdiomDIT-LTB && git pull
#   SKIP_JUDGE=1 nohup bash -c '
#     bash mechanistic/run_v4_gpt52_naivepeak_random_qwen9b.sh 42 0 > log_naivepeak_rand_s42.txt 2>&1 &&
#     bash mechanistic/run_v4_gpt52_naivepeak_random_qwen9b.sh 43 0 > log_naivepeak_rand_s43.txt 2>&1 &&
#     bash mechanistic/run_v4_gpt52_naivepeak_random_qwen9b.sh 44 0 > log_naivepeak_rand_s44.txt 2>&1
#   ' > log_naivepeak_rand_all.txt 2>&1 &
#
# 输出:
#   results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52/ablation_L<naive_peak>_dir_ablate_allprompts__rand_s<S>*.json
#   results/<pair>/Qwen3.5-9B/evaluation/ablation_L<naive_peak>_dir_ablate_allprompts__rand_s<S>_ablation_score.json

MODEL=Qwen3.5-9B
EXP=v4_gpt52
S=${1:?usage: run_v4_gpt52_naivepeak_random_qwen9b.sh <SEED> <GPU_ID>}
GPU=${2:-0}

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
  echo "########  $LP  naive_peak_L=$L  seed=$S  gpu=$GPU  (random-direction)  ########"

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
