#!/usr/bin/env bash
# Basic-only 版 random-direction control — Qwen3.5-9B, ONE seed on ONE GPU.
# 在 Basic-only 场景下，用同 footprint 的随机方向替换 LTD 做全层消融，验证
# LTD 效应在 Basic-only 上仍然不 direction-specific（对照 run_v4_gpt52_basic_qwen9b.sh 的 LTD 消融）。
#
# 前提：先跑 run_v4_gpt52_basic_qwen9b.sh 生成 group_known_lte.json + probing 缓存
# （random 模式不读 probing 的 direction，但复用同一 Group A 集合，规范流程）
#
# 用法:   bash mechanistic/run_v4_gpt52_basic_random_qwen9b.sh <SEED> <GPU_ID>
#
# 3 seed 并行:
#   bash mechanistic/run_v4_gpt52_basic_random_qwen9b.sh 42 0 > log_basic_rand_s42.txt 2>&1 &
#   bash mechanistic/run_v4_gpt52_basic_random_qwen9b.sh 43 1 > log_basic_rand_s43.txt 2>&1 &
#   bash mechanistic/run_v4_gpt52_basic_random_qwen9b.sh 44 2 > log_basic_rand_s44.txt 2>&1 &
#   wait

set -u
MODEL=Qwen3.5-9B
EXP=v4_gpt52_basic
S=${1:?usage: run_v4_gpt52_basic_random_qwen9b.sh <SEED> <GPU_ID>}
GPU=${2:-0}
PAIRS="ja-en en-fa fa-en fr-en ko-en fi-en"

for LP in $PAIRS; do
  echo ""
  echo "########  $LP  seed=$S  gpu=$GPU  (Basic-only, random-direction)  ########"

  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  if [ ! -f "$MECH/group_known_lte.json" ]; then
    echo "  [SKIP $LP] 缺 $MECH/group_known_lte.json，先跑 run_v4_gpt52_basic_qwen9b.sh"
    continue
  fi
  N_A=$(python -c "import json;print(json.load(open('$MECH/group_known_lte.json',encoding='utf-8'))['count'])")
  if [ "$N_A" -lt 5 ]; then
    echo "  [SKIP $LP] A_Basic=$N_A <5，跳过"
    continue
  fi

  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
    --lang-pair "$LP" --model "$MODEL" --ablate-layers all --group-b strict \
    --exp-name "$EXP" --prompt-type BasicPrompt \
    --random-direction --seed "$S"

  OUT="ablation_Lall_dir_ablate_BasicPrompt__rand_s${S}"
  ABL=$(ls "$MECH/${OUT}__for_eval_ablation.json" 2>/dev/null | tail -n 1)
  echo "  random-ablation file: $ABL"
  if [ -z "$ABL" ]; then
    echo "  [WARN $LP] 缺 $ABL，跳过 re-LTE"
    continue
  fi

  if [ -n "${SKIP_JUDGE:-}" ]; then
    echo "  step3: SKIPPED (SKIP_JUDGE=1)。翻译输出已写盘,本地跑 run_v4_gpt52_basic_judge_local.sh 补判审。"
  else
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$ABL" \
      --output-prefix "${OUT}_ablation"
  fi
done

echo ""
echo "==== DONE seed=$S (gpu=$GPU) Basic-only random-direction. ===="
