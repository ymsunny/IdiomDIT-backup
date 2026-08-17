#!/usr/bin/env bash
# 本地判审脚本 —— 对应服务器的 SKIP_JUDGE=1 GPU-only 流程。
# 服务器只跑 step0-2(分组+probing+LTD 消融生成),翻译输出通过 git 拉回本地,
# 由本地(API 稳定)对所有 for_eval_{baseline,ablation}.json 跑 gpt-5.2 判审。
#
# 覆盖:
#   - LTD 主实验:ablation_Lall_dir_ablate_BasicPrompt__for_eval_{baseline,ablation}.json
#   - Random 对照(3 seed):ablation_Lall_dir_ablate_BasicPrompt__rand_s{42,43,44}__for_eval_ablation.json
#
# 幂等:分数文件已存在 → 调 retry_failed_lte_v4.py 只补 null;不存在 → 全量跑。
#
# 用法(在本地 D:\CodeSpace\IdiomDIT-LTB\ 目录):
#   bash mechanistic/run_v4_gpt52_basic_judge_local.sh
#
# 只跑某方向:
#   PAIRS="ja-en" bash mechanistic/run_v4_gpt52_basic_judge_local.sh
#
# 只判 LTD 主实验,跳过 random 对照:
#   SKIP_RAND=1 bash mechanistic/run_v4_gpt52_basic_judge_local.sh

set -u
MODEL=Qwen3.5-9B
EXP=v4_gpt52_basic
PAIRS=${PAIRS:-"ja-en en-fa fa-en fr-en ko-en fi-en"}
JUDGE=gpt-5.2
RAND_SEEDS=${RAND_SEEDS:-"42 43 44"}

echo "==== 本地 gpt-5.2 判审 Basic-only 消融输出 | model=$MODEL | dirs=$PAIRS ===="

for LP in $PAIRS; do
  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  EV="results/$LP/$MODEL/evaluation"
  if [ ! -d "$MECH" ]; then
    echo "  [SKIP $LP] 无 $MECH,先在服务器跑 GPU pipeline 并 push"
    continue
  fi

  # ---- LTD 主实验:baseline + ablation ----
  for KIND in baseline ablation; do
    INPUT="$MECH/ablation_Lall_dir_ablate_BasicPrompt__for_eval_${KIND}.json"
    PREFIX="ablation_v4gpt52_basic_${KIND}"
    SCORE="$EV/${PREFIX}_score.json"
    if [ ! -f "$INPUT" ]; then
      echo "  [SKIP $LP/$KIND] 缺 $INPUT"; continue
    fi
    echo ""
    if [ -f "$SCORE" ]; then
      echo "--- $LP LTD $KIND: 已有 $PREFIX_score.json,补 null ---"
      python evaluation/retry_failed_lte_v4.py --lang-pair "$LP" --model "$MODEL" \
        --judge-model "$JUDGE" --output-prefix "$PREFIX"
    else
      echo "--- $LP LTD $KIND: 新跑 ---"
      python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
        --judge-model "$JUDGE" --translation-input "$INPUT" \
        --output-prefix "$PREFIX"
    fi
  done

  # ---- Random 对照 3 seed(只判 ablation,baseline 复用 LTD 的) ----
  if [ -z "${SKIP_RAND:-}" ]; then
    for S in $RAND_SEEDS; do
      INPUT="$MECH/ablation_Lall_dir_ablate_BasicPrompt__rand_s${S}__for_eval_ablation.json"
      PREFIX="ablation_Lall_dir_ablate_BasicPrompt__rand_s${S}_ablation"
      SCORE="$EV/${PREFIX}_score.json"
      if [ ! -f "$INPUT" ]; then
        echo "  [SKIP $LP rand s=$S] 缺 $INPUT"; continue
      fi
      echo ""
      if [ -f "$SCORE" ]; then
        echo "--- $LP RAND s=$S: 已有,补 null ---"
        python evaluation/retry_failed_lte_v4.py --lang-pair "$LP" --model "$MODEL" \
          --judge-model "$JUDGE" --output-prefix "$PREFIX"
      else
        echo "--- $LP RAND s=$S: 新跑 ---"
        python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
          --judge-model "$JUDGE" --translation-input "$INPUT" \
          --output-prefix "$PREFIX"
      fi
    done
  fi
done

echo ""
echo "==== 判审完成 ===="
