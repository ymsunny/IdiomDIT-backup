"""
direction_ablation_generation.py — LTD direction ablation (Arditi et al. 2024)

▌目的
  Probing 找到的 ŵ 是 LTE-vs-correct 的线性可分方向。
  在 generate 循环中沿 ŵ 做 direction ablation，验证 ŵ 是否真的因果驱动字面翻译行为。

▌干预模式
  - direction_ablate (Arditi 2024):  h ← h − (h·ŵ)ŵ        （推荐 --ablate-layers all）
  - direction_add (Turner 2023):     h ← h + α·ŵ            （剂量—反应曲线）

▌输入
  results/{lang_pair}/{model}/mechanistic/
    ├── group_known_lte.json              ← Group A
    └── probing_balanced_strict.json      ← probing 保存的每层 ŵ (default strict)

▌输出
  results/{lang_pair}/{model}/mechanistic/[exp_name/]
    ├── ablation_L{tag}_{mode}_{prompt}.json                          主输出
    └── ablation_L{tag}_{mode}_{prompt}__for_eval_{baseline|ablation}.json
        可直接作为 eval_translation_lte_old.py 的 --translation-input

▌用法
  CUDA_VISIBLE_DEVICES=0 python mechanistic/direction_ablation_generation.py \\
    --lang-pair en-fa --model Qwen3.5-9B \\
    --ablate-layers all --mode direction_ablate
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import torch
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, PROMPTS, ALL_TARGETS, is_thinking_model
from model_utils import load_local_model


def build_chat_prompt(sentence, prompt_type, src_lang, tgt_lang, model_name, tokenizer):
    system_prompt = PROMPTS[prompt_type].replace("[SRC]", src_lang).replace("[TGT]", tgt_lang)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sentence},
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if is_thinking_model(model_name):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def load_directions_from_probing(mech_dir, target_layers, probing_filename, balanced=True):
    """读取每层归一化 direction。balanced=True → direction_balanced。"""
    probing_path = os.path.join(mech_dir, probing_filename)
    if not os.path.exists(probing_path):
        raise FileNotFoundError(
            f"未找到 {probing_path}。先跑 probe_ltb_by_hidden_state_balanced.py"
        )
    with open(probing_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = "direction_balanced" if balanced else "direction_imbalanced"
    dirs = {}
    for l in target_layers:
        layer_result = next((r for r in data["results"] if r["layer"] == l), None)
        if layer_result is None:
            raise ValueError(f"{probing_filename} 没有 layer {l} 的结果")
        if key not in layer_result:
            raise KeyError(f"{probing_filename} layer {l} 缺少 {key}")
        w = np.array(layer_result[key], dtype=np.float32)
        w = w / (np.linalg.norm(w) + 1e-12)
        dirs[l] = w
    return dirs


def install_intervention_hooks(model, target_layers, mode, directions, alpha, device):
    """干预 hook 覆盖整个 generate 循环。"""
    hooks = []
    if directions is None:
        raise ValueError(f"{mode} 模式必须提供 directions")

    for i, layer in enumerate(model.model.layers):
        if i not in target_layers:
            continue
        w_np = directions[i]
        w_tensor = torch.tensor(w_np, dtype=torch.float32, device=device)

        if mode == "direction_ablate":
            def make_ablate(w_t):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    w_cast = w_t.to(h.dtype)
                    proj = torch.matmul(h, w_cast)
                    h = h - proj.unsqueeze(-1) * w_cast
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook_fn
            hooks.append(layer.register_forward_hook(make_ablate(w_tensor)))

        elif mode == "direction_add":
            alpha_t = float(alpha)
            def make_add(w_t, a):
                def hook_fn(module, inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    w_cast = w_t.to(h.dtype)
                    h = h + a * w_cast
                    if isinstance(output, tuple):
                        return (h,) + output[1:]
                    return h
                return hook_fn
            hooks.append(layer.register_forward_hook(make_add(w_tensor, alpha_t)))
        else:
            raise ValueError(f"未知 mode: {mode}")

    return hooks


def generate_translation(model, tokenizer, text, max_new_tokens, is_thinking):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][input_len:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if is_thinking:
        import re
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw.strip()


def main():
    parser = argparse.ArgumentParser(description="LTD direction ablation (Arditi 2024)")
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ablate-layers", required=True,
                        help="逗号分隔层号或 'all'（推荐 Arditi 协议）")
    parser.add_argument("--mode", default="direction_ablate",
                        choices=["direction_ablate", "direction_add"])
    parser.add_argument("--alpha", type=float, default=-1.0,
                        help="direction_add 的系数（α<0 推向 non-LTE）")
    parser.add_argument("--group-b", default="strict", choices=["strict", "loose", "superstrict"],
                        help="选择哪个 probing 输出（strict=默认主分析; superstrict=Lit=0,Mean=1; loose）")
    parser.add_argument("--imbalanced", action="store_true",
                        help="用 direction_imbalanced（legacy），默认 balanced")
    parser.add_argument("--random-direction", action="store_true",
                        help="【随机方向基线 concern #4】每层用随机单位向量代替 LTD；建议多 seed 跑取均值")
    parser.add_argument("--direction-from-lang-pair", default=None,
                        help="【跨语言 transfer】从其他语言对加载 direction")
    parser.add_argument("--prompt-type", default=None,
                        help="只对单一 prompt 做 ablation（默认 None = 全部 4 prompt）")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--shuffle", action="store_true",
                        help="--max-samples 截断前先 shuffle Group A")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regen-baseline", action="store_true",
                        help="重新生成 baseline 译文（默认复用 Group A 已有的 model_translation）")
    parser.add_argument("--exp-name", default="",
                        help="实验名隔离输出目录")
    args = parser.parse_args()

    ablate_layers_spec = args.ablate_layers.strip().lower()

    cfg = get_config(args.lang_pair, model=args.model)
    direction = cfg["direction"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    mech_dir = str(cfg["mechanistic_dir"])
    if args.exp_name:
        mech_dir = str(Path(mech_dir) / args.exp_name)

    probing_filename = {
        "strict": "probing_balanced_strict.json",
        "loose": "probing_balanced.json",
        "superstrict": "probing_balanced_superstrict.json",
    }[args.group_b]

    # 1. 读 Group A
    with open(os.path.join(mech_dir, "group_known_lte.json"), encoding="utf-8") as f:
        group_a = json.load(f)["results"]

    if args.prompt_type is None:
        filtered = list(group_a)
        from collections import Counter
        ptc = Counter(x["prompt_type"] for x in group_a)
        print(f"Group A 全量: {len(group_a)} (跨 prompt 聚合)")
        print(f"  prompt 分布: {dict(ptc)}")
    else:
        filtered = [x for x in group_a if x["prompt_type"] == args.prompt_type]
        print(f"Group A 全量: {len(group_a)} | 筛到 {args.prompt_type}: {len(filtered)}")

    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(filtered)
        print(f"  --shuffle on (seed={args.seed})")

    if args.max_samples:
        filtered = filtered[:args.max_samples]
        print(f"  --max-samples={args.max_samples}, 实际处理 {len(filtered)} 条")

    if not filtered:
        print(f"错误: 没有可用的 Group A 样本")
        return

    # 2. 加载模型
    model, tokenizer, device = load_local_model(args.model)
    n_layers = model.config.num_hidden_layers
    is_thinking = is_thinking_model(args.model)

    if ablate_layers_spec == "all":
        target_layers = set(range(n_layers))
        layer_tag = "all"
        print(f"  --ablate-layers all → 所有 {n_layers} 层（Arditi 协议）")
    else:
        target_layers = set(int(x) for x in ablate_layers_spec.split(","))
        layer_tag = "_".join(str(l) for l in sorted(target_layers))

    print(f"模型层数: {n_layers} | mode: {args.mode}")
    if args.mode == "direction_add":
        print(f"  alpha={args.alpha}")

    for l in target_layers:
        if l < 0 or l >= n_layers:
            raise ValueError(f"ablate-layer {l} 越界 (n_layers={n_layers})")

    # 3. 加载 / 构造 direction
    balanced = not args.imbalanced
    if args.random_direction:
        D = model.config.hidden_size
        rng = np.random.RandomState(args.seed)
        directions = {}
        for l in target_layers:
            v = rng.randn(D).astype(np.float32)
            directions[l] = v / (np.linalg.norm(v) + 1e-12)
        print(f"\n[RANDOM-DIR baseline] seed={args.seed}, dim={D}, "
              f"{len(target_layers)} 层随机单位向量（与 LTD 同 footprint）")
    else:
        if args.direction_from_lang_pair:
            donor_cfg = get_config(args.direction_from_lang_pair, model=args.model)
            donor_mech_dir = str(donor_cfg["mechanistic_dir"])
            if args.exp_name:
                donor_mech_dir = str(Path(donor_mech_dir) / args.exp_name)
            print(f"\n[CROSS-LANG TRANSFER] donor = {args.direction_from_lang_pair}")
            direction_source = donor_mech_dir
        else:
            direction_source = mech_dir

        print(f"\nStep 1: 加载 {'balanced' if balanced else 'imbalanced'} direction "
              f"({probing_filename}) for {len(target_layers)} 层 from {direction_source}")
        directions = load_directions_from_probing(direction_source, target_layers,
                                                  probing_filename, balanced=balanced)
    for l in sorted(target_layers)[:4]:
        print(f"  Layer {l}: ||w||={np.linalg.norm(directions[l]):.4f}, dim={len(directions[l])}")
    if len(target_layers) > 4:
        print(f"  ... (共 {len(target_layers)} 层)")

    # 4. 生成 (baseline + ablated)
    results = []
    print(f"\nStep 2: 生成翻译（{len(filtered)} 条 × 2 版本）...")

    for i, item in enumerate(filtered):
        text = build_chat_prompt(item["sentence"], item["prompt_type"],
                                 src_lang, tgt_lang, args.model, tokenizer)

        if args.regen_baseline:
            baseline = generate_translation(model, tokenizer, text,
                                            args.max_new_tokens, is_thinking)
        else:
            baseline = item.get("model_translation", "")

        hooks = install_intervention_hooks(
            model, target_layers=target_layers, mode=args.mode,
            directions=directions, alpha=args.alpha, device=model.device)
        try:
            abl_trans = generate_translation(model, tokenizer, text,
                                             args.max_new_tokens, is_thinking)
        finally:
            for h in hooks:
                h.remove()

        results.append({
            "id": item["id"],
            "direction": direction,
            "prompt_type": item["prompt_type"],
            "idiom": item["idiom"],
            "sentence": item["sentence"],
            "reference_meaning": item.get("reference_meaning", ""),
            "gold_translation": item.get("gold_translation", ""),
            "baseline_translation": baseline,
            "ablation_translation": abl_trans,
            "baseline_was_regenerated": args.regen_baseline,
        })

        if (i + 1) % 5 == 0 or i == len(filtered) - 1:
            print(f"  [{i+1}/{len(filtered)}] id={item['id']} | "
                  f"baseline[:60]={baseline[:60]!r} | ablation[:60]={abl_trans[:60]!r}")

    # 5. 保存主输出
    if args.mode == "direction_add":
        mode_tag = f"dir_add_a{args.alpha:+.2f}".replace("+", "p").replace("-", "m")
    else:
        mode_tag = "dir_ablate"
    prompt_tag = args.prompt_type if args.prompt_type else "allprompts"
    donor_tag = f"__donor_{args.direction_from_lang_pair}" if args.direction_from_lang_pair else ""
    # group-b 标记：strict（默认）不加后缀以保持向后兼容；其余加 __gb_{name} 避免覆盖 strict 结果
    gb_tag = "" if args.group_b == "strict" else f"__gb_{args.group_b}"
    rand_tag = f"__rand_s{args.seed}" if args.random_direction else ""
    out_name = f"ablation_L{layer_tag}_{mode_tag}_{prompt_tag}{donor_tag}{gb_tag}{rand_tag}"

    os.makedirs(mech_dir, exist_ok=True)
    out_path = os.path.join(mech_dir, f"{out_name}.json")
    output = {
        "config": {
            "lang_pair": args.lang_pair,
            "model": args.model,
            "direction": direction,
            "direction_from_lang_pair": args.direction_from_lang_pair,
            "mode": args.mode,
            "alpha": args.alpha if args.mode == "direction_add" else None,
            "imbalanced_direction": args.imbalanced,
            "ablate_layers": sorted(target_layers),
            "intervention_site": "residual_stream",
            "prompt_type": args.prompt_type if args.prompt_type else "all",
            "group_b": args.group_b,
            "random_direction": args.random_direction,
            "seed": args.seed,
            "probing_source": None if args.random_direction else probing_filename,
            "n_samples": len(results),
            "max_new_tokens": args.max_new_tokens,
            "regen_baseline": args.regen_baseline,
        },
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n主输出: {out_path}")

    # 6. 两份 LTE-ready 文件
    eval_files = {}
    for variant in ("baseline", "ablation"):
        eval_items = []
        for r in results:
            eval_items.append({
                "id": r["id"],
                "source_prompt_type": r["prompt_type"],  # 真实 prompt（用于配对）
                "idiom": r["idiom"],
                "sentence": r["sentence"],
                "reference_meaning": r["reference_meaning"],
                "gold_translation": r["gold_translation"],
                "model_translation": r[f"{variant}_translation"],
            })
        fname = f"{out_name}__for_eval_{variant}.json"
        fpath = os.path.join(mech_dir, fname)
        prompt_meta = args.prompt_type if args.prompt_type else "all"
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "model": args.model,
                    "prompt_type": f"{prompt_meta}__ABLATION_{variant}_{mode_tag}_L{layer_tag}",
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "ablation_variant": variant,
                    "ablation_mode": args.mode,
                    "ablation_alpha": args.alpha if args.mode == "direction_add" else None,
                    "ablate_layers": sorted(target_layers),
                },
                "results": eval_items,
            }, f, ensure_ascii=False, indent=2)
        eval_files[variant] = fpath

    print(f"\nLTE 评估用单文件:")
    print(f"  baseline: {eval_files['baseline']}")
    print(f"  ablation: {eval_files['ablation']}")
    print("\n下一步（评估 LTE）:")
    print(f"  python evaluation/eval_translation_lte_old.py --lang-pair {args.lang_pair} --model {args.model} \\")
    print(f"      --translation-input {eval_files['baseline']} \\")
    print(f"      --output-prefix {out_name}_baseline")
    print(f"  python evaluation/eval_translation_lte_old.py --lang-pair {args.lang_pair} --model {args.model} \\")
    print(f"      --translation-input {eval_files['ablation']} \\")
    print(f"      --output-prefix {out_name}_ablation")


if __name__ == "__main__":
    main()
