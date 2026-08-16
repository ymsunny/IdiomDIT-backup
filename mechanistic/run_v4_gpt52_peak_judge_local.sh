#!/usr/bin/env bash
# 本地 gpt-5.2 判审脚本 —— 对应 SKIP_JUDGE=1 的 peak-layer GPU 流程。
# 服务器 GPU 只做 step2(peak-layer 消融生成),翻译输出 git push 回本地,
# 本地跑此脚本对 for_eval_{baseline,ablation}.json 走 gpt-5.2 判审。
#
# 每方向的 peak grouped_cv layer(与 run_v4_gpt52_peak_qwen9b.sh 一致):
#   en-fa=15  fa-en=16  fi-en=17  fr-en=31  ja-en=11  ko-en=11
#
# 覆盖:
#   - LTD peak-layer 主实验:ablation_L<peak>_dir_ablate_allprompts__for_eval_{baseline,ablation}.json
#   - Random peak-layer 对照(3 seed):ablation_L<peak>_dir_ablate_allprompts__rand_s{42,43,44}__for_eval_ablation.json
#
# 幂等:分数文件已存在 → 调 retry_failed_lte_v4.py 只补 null;不存在 → 全量跑。
#
# 用法(本地 D:\CodeSpace\IdiomDIT-LTB\):
#   bash mechanistic/run_v4_gpt52_peak_judge_local.sh
#
# 只跑某方向:
#   PAIRS="ja-en" bash mechanistic/run_v4_gpt52_peak_judge_local.sh
#
# 只判主实验,暂时跳过 random:
#   SKIP_RAND=1 bash mechanistic/run_v4_gpt52_peak_judge_local.sh

set -u
MODEL=Qwen3.5-9B
EXP=v4_gpt52
PAIRS=${PAIRS:-"ja-en en-fa fa-en fr-en ko-en fi-en"}
JUDGE=gpt-5.2
RAND_SEEDS=${RAND_SEEDS:-"42 43 44"}

declare -A PEAK
PEAK[en-fa]=15
PEAK[fa-en]=16
PEAK[fi-en]=17
PEAK[fr-en]=31
PEAK[ja-en]=11
PEAK[ko-en]=11

echo "==== 本地 gpt-5.2 判审 peak-layer 消融输出 | model=$MODEL | dirs=$PAIRS ===="

for LP in $PAIRS; do
  L=${PEAK[$LP]}
  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  EV="results/$LP/$MODEL/evaluation"
  if [ ! -d "$MECH" ]; then
    echo "  [SKIP $LP] 无 $MECH,先在服务器跑 GPU pipeline 并 push"
    continue
  fi

  # ---- LTD peak-layer 主实验 ----
  for KIND in baseline ablation; do
    INPUT="$MECH/ablation_L${L}_dir_ablate_allprompts__for_eval_${KIND}.json"
    PREFIX="ablation_v4gpt52_peak_${KIND}"
    SCORE="$EV/${PREFIX}_score.json"
    if [ ! -f "$INPUT" ]; then
      echo "  [SKIP $LP/$KIND] 缺 $INPUT"; continue
    fi
    echo ""
    if [ -f "$SCORE" ]; then
      echo "--- $LP LTD peak_L=$L $KIND: 已有,补 null ---"
      python evaluation/retry_failed_lte_v4.py --lang-pair "$LP" --model "$MODEL" \
        --judge-model "$JUDGE" --output-prefix "$PREFIX"
    else
      echo "--- $LP LTD peak_L=$L $KIND: 新跑 ---"
      python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
        --judge-model "$JUDGE" --translation-input "$INPUT" \
        --output-prefix "$PREFIX"
    fi
  done

  # ---- Random 对照 3 seed(只判 ablation) ----
  if [ -z "${SKIP_RAND:-}" ]; then
    for S in $RAND_SEEDS; do
      INPUT="$MECH/ablation_L${L}_dir_ablate_allprompts__rand_s${S}__for_eval_ablation.json"
      PREFIX="ablation_L${L}_dir_ablate_allprompts__rand_s${S}_ablation"
      SCORE="$EV/${PREFIX}_score.json"
      if [ ! -f "$INPUT" ]; then
        echo "  [SKIP $LP rand s=$S] 缺 $INPUT"; continue
      fi
      echo ""
      if [ -f "$SCORE" ]; then
        echo "--- $LP RAND peak_L=$L s=$S: 已有,补 null ---"
        python evaluation/retry_failed_lte_v4.py --lang-pair "$LP" --model "$MODEL" \
          --judge-model "$JUDGE" --output-prefix "$PREFIX"
      else
        echo "--- $LP RAND peak_L=$L s=$S: 新跑 ---"
        python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
          --judge-model "$JUDGE" --translation-input "$INPUT" \
          --output-prefix "$PREFIX"
      fi
    done
  fi
done

echo ""
echo "==== 判审完成。跑 analysis/aggregate_peak_ablation.py 出表 ===="
