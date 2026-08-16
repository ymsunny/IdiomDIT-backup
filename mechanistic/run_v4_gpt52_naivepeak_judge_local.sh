#!/usr/bin/env bash
# 本地 gpt-5.2 判审脚本 —— 对应 SKIP_JUDGE=1 的 naive-peak GPU 流程。
# 服务器 GPU 只做 step2(naive-peak 消融生成),翻译输出 git push 回本地,
# 本地跑此脚本对 for_eval_{baseline,ablation}.json 走 gpt-5.2 判审。
#
# 每方向的 naive peak layer(= paper Table 3 "Peak L" 列,
# 与 run_v4_gpt52_naivepeak_qwen9b.sh 一致;区别于 grouped-CV 峰实验):
#   en-fa=27  fa-en=8  fi-en=27  fr-en=20  ja-en=21  ko-en=16
#
# 覆盖:
#   - LTD naive-peak 主实验:ablation_L<naive_peak>_dir_ablate_allprompts__for_eval_{baseline,ablation}.json
#     → 判审前缀 ablation_v4gpt52_naivepeak_{baseline,ablation}(与 grouped-CV 峰的
#       ablation_v4gpt52_peak_* 前缀区分,互不覆盖)
#   - Random naive-peak 对照(3 seed):ablation_L<naive_peak>_..._rand_s{42,43,44}__for_eval_ablation.json
#     → 前缀含层号,与 grouped-CV 峰输出天然区分
#
# 幂等:分数文件已存在 → 调 retry_failed_lte_v4.py 只补 null;不存在 → 全量跑。
#
# 用法(本地 D:\CodeSpace\IdiomDIT-LTB\):
#   bash mechanistic/run_v4_gpt52_naivepeak_judge_local.sh
#
# 只跑某方向:
#   PAIRS="ja-en" bash mechanistic/run_v4_gpt52_naivepeak_judge_local.sh
#
# 只判主实验,暂时跳过 random:
#   SKIP_RAND=1 bash mechanistic/run_v4_gpt52_naivepeak_judge_local.sh

set -u
MODEL=Qwen3.5-9B
EXP=v4_gpt52
PAIRS=${PAIRS:-"ja-en en-fa fa-en fr-en ko-en fi-en"}
JUDGE=gpt-5.2
RAND_SEEDS=${RAND_SEEDS:-"42 43 44"}

declare -A PEAK
PEAK[en-fa]=27
PEAK[fa-en]=8
PEAK[fi-en]=27
PEAK[fr-en]=20
PEAK[ja-en]=21
PEAK[ko-en]=16

echo "==== 本地 gpt-5.2 判审 naive-peak 消融输出 | model=$MODEL | dirs=$PAIRS ===="

for LP in $PAIRS; do
  L=${PEAK[$LP]}
  MECH="results/$LP/$MODEL/mechanistic/$EXP"
  EV="results/$LP/$MODEL/evaluation"
  if [ ! -d "$MECH" ]; then
    echo "  [SKIP $LP] 无 $MECH,先在服务器跑 GPU pipeline 并 push"
    continue
  fi

  # ---- LTD naive-peak 主实验 ----
  for KIND in baseline ablation; do
    INPUT="$MECH/ablation_L${L}_dir_ablate_allprompts__for_eval_${KIND}.json"
    PREFIX="ablation_v4gpt52_naivepeak_${KIND}"
    SCORE="$EV/${PREFIX}_score.json"
    if [ ! -f "$INPUT" ]; then
      echo "  [SKIP $LP/$KIND] 缺 $INPUT"; continue
    fi
    echo ""
    if [ -f "$SCORE" ]; then
      echo "--- $LP LTD naive_peak_L=$L $KIND: 已有,补 null ---"
      python evaluation/retry_failed_lte_v4.py --lang-pair "$LP" --model "$MODEL" \
        --judge-model "$JUDGE" --output-prefix "$PREFIX"
    else
      echo "--- $LP LTD naive_peak_L=$L $KIND: 新跑 ---"
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
        echo "--- $LP RAND naive_peak_L=$L s=$S: 已有,补 null ---"
        python evaluation/retry_failed_lte_v4.py --lang-pair "$LP" --model "$MODEL" \
          --judge-model "$JUDGE" --output-prefix "$PREFIX"
      else
        echo "--- $LP RAND naive_peak_L=$L s=$S: 新跑 ---"
        python evaluation/eval_translation_lte_v4.py --lang-pair "$LP" --model "$MODEL" --no-confirm \
          --judge-model "$JUDGE" --translation-input "$INPUT" \
          --output-prefix "$PREFIX"
      fi
    done
  fi
done

echo ""
echo "==== 判审完成。跑 analysis/aggregate_naivepeak_ablation.py 出表 ===="
