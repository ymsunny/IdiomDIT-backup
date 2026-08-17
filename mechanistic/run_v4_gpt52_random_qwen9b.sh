#!/usr/bin/env bash
# v4 GPT-5.2 RANDOM-DIRECTION control — Qwen3.5-9B, ONE seed on ONE GPU.
# Ablate along a RANDOM unit direction per layer (not the LTD w), regenerate Group-A
# translations for all 6 pairs, then re-LTE the ablated output with gpt-5.2.
# Baseline is identical to the LTD run; the random run writes its own seeded baseline file
# (harmless), and we only re-evaluate the random ablation variant.
#
# Usage:   bash mechanistic/run_v4_gpt52_random_qwen9b.sh <SEED> <GPU_ID>
#
# Run 3 seeds on 3 GPUs in parallel:
#   cd IdiomDIT-LTB && git pull
#   bash mechanistic/run_v4_gpt52_random_qwen9b.sh 42 0 > log_rand_s42.txt 2>&1 &
#   bash mechanistic/run_v4_gpt52_random_qwen9b.sh 43 1 > log_rand_s43.txt 2>&1 &
#   bash mechanistic/run_v4_gpt52_random_qwen9b.sh 44 2 > log_rand_s44.txt 2>&1 &
#   wait

MODEL=Qwen3.5-9B
EXP=v4_gpt52
S=${1:?usage: run_v4_gpt52_random_qwen9b.sh <SEED> <GPU_ID>}
GPU=${2:-0}
PAIRS="en-fa fa-en ja-en fr-en fi-en ko-en"

for LP in $PAIRS; do
  echo ""
  echo "########  $LP  seed=$S  gpu=$GPU  (random-direction)  ########"
  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
    --lang-pair "$LP" --model "$MODEL" --ablate-layers all --group-b strict --exp-name "$EXP" \
    --random-direction --seed "$S"

  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  OUT="ablation_Lall_dir_ablate_allprompts__rand_s${S}"
  ABL=$(ls "$MECH/${OUT}__for_eval_ablation.json" 2>/dev/null | tail -n 1)
  echo "  random-ablation file: $ABL"

  python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
    --judge-model gpt-5.2 --translation-input "$ABL" \
    --output-prefix "${OUT}_ablation"
done

echo ""
echo "==== DONE seed=$S (gpu=$GPU). ===="
