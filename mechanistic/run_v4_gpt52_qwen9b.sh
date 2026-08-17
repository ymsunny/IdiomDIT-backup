#!/usr/bin/env bash
# v4 GPT-5.2 mechanistic pipeline — Qwen3.5-9B ONLY (all 6 language pairs)
#   step0  regroup A/B from v4 GPT-5.2 LTE          (CPU)
#   step1  probing  -> per-layer LTD direction w     (GPU + model)
#   step2  direction ablation generation            (GPU + model)
#   step3  re-LTE the baseline+ablation generations  (API, judge = gpt-5.2)
#
# Run on the GPU server:
#   git pull && bash mechanistic/run_v4_gpt52_qwen9b.sh [GPU_ID]
#
# Outputs land in results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52/ (groups, probe, ablation)
# and results/<pair>/Qwen3.5-9B/evaluation/ablation_v4gpt52_{baseline,ablation}_score.json (re-LTE).

MODEL=Qwen3.5-9B
EXP=v4_gpt52
GPU=${1:-0}
PAIRS="en-fa fa-en ja-en fr-en fi-en ko-en"

echo "==== step0: regroup A/B from v4 GPT-5.2 LTE (all pairs) ===="
python mechanistic/extract_known_lte_groups.py --all --model "$MODEL" \
  --lte-prefix translation_lte_v4_gpt52 --exp-name "$EXP"

for LP in $PAIRS; do
  echo ""
  echo "################  $LP  ################"

  echo "---- step1: probing ($LP) ----"
  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/probe_ltb_by_hidden_state_balanced.py \
    --lang-pair "$LP" --model "$MODEL" --exp-name "$EXP"

  echo "---- step2: direction ablation generation ($LP) ----"
  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
    --lang-pair "$LP" --model "$MODEL" --ablate-layers all --group-b strict --exp-name "$EXP"

  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  BASE=$(ls "$MECH"/ablation_*__for_eval_baseline.json 2>/dev/null | tail -n 1)
  ABL=$(ls "$MECH"/ablation_*__for_eval_ablation.json  2>/dev/null | tail -n 1)
  echo "  baseline file: $BASE"
  echo "  ablation file: $ABL"

  echo "---- step3: re-LTE with gpt-5.2 ($LP) ----"
  python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
    --judge-model gpt-5.2 --translation-input "$BASE" \
    --output-prefix ablation_v4gpt52_baseline
  python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
    --judge-model gpt-5.2 --translation-input "$ABL" \
    --output-prefix ablation_v4gpt52_ablation
done

echo ""
echo "==== DONE. Sync results/ back to local, then aggregate ΔLTE. ===="
