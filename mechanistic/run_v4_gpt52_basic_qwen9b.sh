#!/usr/bin/env bash
# Basic-only 版本的 v4 GPT-5.2 mechanistic pipeline — Qwen3.5-9B ONLY
# 回应 reviewer 关于 cross-prompt pooling 的顾虑：所有步骤都只保留 BasicPrompt 条目。
#
#   step0  regroup A/B from v4 GPT-5.2 LTE (--filter-prompt BasicPrompt) (CPU)
#   step1  probing on Basic-only groups → per-layer LTD (GPU + model)
#   step2  direction ablation on Basic-only Group A (GPU + model)
#   step3  re-LTE the baseline+ablation generations with gpt-5.2 (API)
#          → 用 SKIP_JUDGE=1 跳过 step3(GPU-only 模式:服务器判审 API 不稳时用,
#            拉回本地后跑 run_v4_gpt52_basic_judge_local.sh 补判审)
#
# 与 run_v4_gpt52_qwen9b.sh 的差异：
#   EXP=v4_gpt52_basic（隔离目录，不覆盖 pooled 结果）
#   extract 加 --filter-prompt BasicPrompt
#   ablation 加 --prompt-type BasicPrompt（Group A 已只含 Basic，冗余但显式）
#   ja-en 作为 headline 方向；小方向 A_Basic<20 会跑但预期方差大，仅供附录
#
# 用法（在 GPU 服务器 g73 上）:
#   cd /sym/IdiomDIT-LTB && git pull
#   bash mechanistic/run_v4_gpt52_basic_qwen9b.sh [GPU_ID]
#
# 输出:
#   results/<pair>/Qwen3.5-9B/mechanistic/v4_gpt52_basic/
#     ├── group_known_lte.json                        (Basic-only Group A)
#     ├── group_known_non_lte_strict.json             (Basic-only Group B strict)
#     ├── probing_balanced_strict.json                (Basic-only LTD 每层)
#     ├── ablation_Lall_dir_ablate_BasicPrompt.json   (LTD 消融主输出)
#     └── ablation_Lall_dir_ablate_BasicPrompt__for_eval_{baseline,ablation}.json
#   results/<pair>/Qwen3.5-9B/evaluation/
#     ├── ablation_v4gpt52_basic_baseline_score.json
#     └── ablation_v4gpt52_basic_ablation_score.json

set -u
MODEL=Qwen3.5-9B
EXP=v4_gpt52_basic
GPU=${1:-0}
# ja-en 作为 headline 放最前（Basic A=63，规模最好），其它跟着跑
PAIRS="ja-en en-fa fa-en fr-en ko-en fi-en"

echo "==== step0: regroup A/B from v4 GPT-5.2 LTE, Basic-only (all pairs) ===="
python mechanistic/extract_known_lte_groups.py --all --model "$MODEL" \
  --lte-prefix translation_lte_v4_gpt52 --exp-name "$EXP" \
  --filter-prompt BasicPrompt

for LP in $PAIRS; do
  echo ""
  echo "################  $LP  (Basic-only)  ################"

  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  # 检查 Group A 是否够跑（Basic-only 下小方向 A 可能 <10，probing 会 skip 但 ablation 仍可做小 n）
  N_A=$(python -c "import json;print(json.load(open('$MECH/group_known_lte.json',encoding='utf-8'))['count'])")
  echo "  Group A (Basic-only) = $N_A"
  if [ "$N_A" -lt 5 ]; then
    echo "  [SKIP $LP] A<5，样本量太小，跳过 probing 和 ablation"
    continue
  fi

  echo "---- step1: probing on Basic-only groups ($LP) ----"
  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/probe_ltb_by_hidden_state_balanced.py \
    --lang-pair "$LP" --model "$MODEL" --exp-name "$EXP"

  echo "---- step2: direction ablation generation, Basic-only Group A ($LP) ----"
  CUDA_VISIBLE_DEVICES=$GPU python mechanistic/direction_ablation_generation.py \
    --lang-pair "$LP" --model "$MODEL" --ablate-layers all --group-b strict \
    --exp-name "$EXP" --prompt-type BasicPrompt

  BASE=$(ls "$MECH"/ablation_Lall_dir_ablate_BasicPrompt__for_eval_baseline.json 2>/dev/null | tail -n 1)
  ABL=$(ls "$MECH"/ablation_Lall_dir_ablate_BasicPrompt__for_eval_ablation.json  2>/dev/null | tail -n 1)
  echo "  baseline file: $BASE"
  echo "  ablation file: $ABL"
  if [ -z "$BASE" ] || [ -z "$ABL" ]; then
    echo "  [WARN $LP] 没找到 for_eval_{baseline,ablation}.json，跳过 re-LTE"
    continue
  fi

  if [ -n "${SKIP_JUDGE:-}" ]; then
    echo "---- step3: SKIPPED (SKIP_JUDGE=1)。翻译输出已写盘,拉回本地跑 run_v4_gpt52_basic_judge_local.sh 补判审。 ----"
  else
    echo "---- step3: re-LTE with gpt-5.2, Basic-only ($LP) ----"
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$BASE" \
      --output-prefix ablation_v4gpt52_basic_baseline
    python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
      --judge-model gpt-5.2 --translation-input "$ABL" \
      --output-prefix ablation_v4gpt52_basic_ablation
  fi
done

echo ""
if [ -n "${SKIP_JUDGE:-}" ]; then
  echo "==== DONE (GPU-only). 下一步:git push 结果到本地,再跑 run_v4_gpt52_basic_judge_local.sh 补判审。 ===="
else
  echo "==== DONE. Basic-only pipeline complete. 下一步：跑 random-direction control ===="
  echo "==== bash mechanistic/run_v4_gpt52_basic_random_qwen9b.sh <SEED> <GPU_ID> ===="
fi
