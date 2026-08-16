#!/usr/bin/env bash
# run_v4_eval.sh — v4 LTE 评估，评委外层、方向内层（先把一个评委的所有方向跑完，再下一个评委）
#
# 用法:
#   bash run_v4_eval.sh [MODEL] [lang1 lang2 ...]     # 默认全部 6 个方向, judge=gpt-5.2
#   JUDGES="gpt-5.2 gpt-4o-mini" bash run_v4_eval.sh Qwen3-8B en-fa fa-en ...  # 自定义评委顺序/集合
#
# 评委顺序/集合由 JUDGES 控制（空格分隔；默认只跑论文采用的 "gpt-5.2"）。
# 顺序：外层评委、内层方向 → 多个评委时先把所有方向的第一个评委跑完，再跑下一个。
# 每个方向：{prefix}_score.json 不存在 → 新跑（eval 自带 checkpoint 续跑）；
#           已存在 → 调 retry_failed_lte_v4.py 补 null 失败条（无 null 秒退）。
# 一趟即可同时收尾「没跑的」和「跑了但有 API 失败的」。纯 API；不覆盖 canonical old 结果。
set -u
MODEL=${1:-Qwen3.5-9B}; shift 2>/dev/null || true
LANGS=("$@"); [ ${#LANGS[@]} -eq 0 ] && LANGS=(en-fa fa-en fr-en fi-en ja-en ko-en)
JUDGES_STR=${JUDGES:-"gpt-5.2"}
echo "=== v4 LTE eval | model=$MODEL | dirs=${LANGS[*]} | judges=$JUDGES_STR (judge-outer) ==="
for J in $JUDGES_STR; do
    case "$J" in
        gpt-4o-mini) PREFIX=translation_lte_v4 ;;
        gpt-5.2)     PREFIX=translation_lte_v4_gpt52 ;;
        *)           PREFIX="translation_lte_v4_$(echo "$J" | tr -c 'A-Za-z0-9' '_')" ;;
    esac
    for lp in "${LANGS[@]}"; do
        SCORE="results/$lp/$MODEL/evaluation/${PREFIX}_score.json"
        if [ -e "$SCORE" ]; then
            # 已存在 → 补 null 失败条（无 null 则秒退，不浪费配额）
            echo ""; echo "--- $J / $lp -> $PREFIX (已存在，补 null) ---"
            python evaluation/retry_failed_lte_v4.py --lang-pair "$lp" --model "$MODEL" \
                --judge-model "$J" --output-prefix "$PREFIX"
        else
            echo ""; echo "--- $J / $lp -> $PREFIX (新跑) ---"
            python evaluation/eval_translation_lte_v4.py --lang-pair "$lp" --model "$MODEL" --no-confirm \
                --judge-model "$J" --output-prefix "$PREFIX"
        fi
    done
done
echo ""; echo "=== done ==="
