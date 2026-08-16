"""
model_utils.py — 统一模型加载 + 多卡工具

所有阶段（detection, interpretation, translation）共用此模块加载模型。
"""

import os
import math
from config import resolve_model_path, LARGE_MODELS


def strip_thinking_output(text: str) -> str:
    """Remove Qwen-style thinking blocks, including truncated/unclosed ones."""
    if not text:
        return text

    import re

    # 1) 去掉完整闭合的 <think>...</think>
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2) 若仍残留 </think>（chat template 把开标签放进了 prompt），取最后一个之后的内容
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    # 3) 残留未闭合的 <think>（被截断的思考）→ 丢弃其后所有内容
    start = cleaned.find("<think>")
    if start != -1:
        cleaned = cleaned[:start]

    # 4) "Final Answer:" 类前缀：只匹配「行首」标记。
    #    用 rfind 做子串匹配会命中模型在思考里复述指令的那句
    #    （"...put the final translation after the prefix 'Final Answer:'"），
    #    把整段推理当成答案返回。行首锚定可避免这种误截取。
    final_markers = (
        "Final Answer:", "Final answer:",
        "Final JSON:", "Final json:",
        "Answer:",
    )
    marker_re = re.compile(
        r"(?m)^\s*(?:" + "|".join(re.escape(m) for m in final_markers) + r")\s*")
    hits = list(marker_re.finditer(cleaned))
    if hits:
        return cleaned[hits[-1].end():].strip()

    # 5) 整段仍是思考（截断/循环，没有产出最终答案）→ 返回空，交上游判失败
    if re.match(r"^\s*(Thinking Process|Thought Process|Reasoning|Analysis)\s*:", cleaned, re.IGNORECASE):
        return ""

    return cleaned.strip()


def load_local_model(model_name: str, gpu_id=None):
    """
    统一加载本地模型。

    Args:
        model_name: 模型名称或路径（如 "Qwen3.5-4B" 或绝对路径）
        gpu_id: GPU 编号。None=自动检测（可用则 cuda:0，否则 CPU）。
                大模型（LARGE_MODELS）忽略 gpu_id，始终用 device_map="auto"。

    Returns:
        (model, tokenizer, device)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = resolve_model_path(model_name)
    is_large = any(m in model_name for m in LARGE_MODELS)

    # 确定设备
    if is_large:
        # 大模型：跨所有可见 GPU 自动分配
        device = torch.device("cuda:0")
        dtype = torch.float16
        device_map = "auto"
        print(f"  加载大模型: {model_name} → device_map=auto (path={resolved})")
    elif gpu_id is not None:
        device = torch.device(f"cuda:{gpu_id}")
        dtype = torch.float16
        device_map = {"": device}
        print(f"  加载模型: {model_name} → cuda:{gpu_id} (path={resolved})")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
        device_map = {"": device}
        print(f"  加载模型: {model_name} → cuda:0 (path={resolved})")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        device_map = None
        print(f"  加载模型: {model_name} → cpu (path={resolved})")

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()

    print(f"  模型加载完成, device={model.device}, dtype={model.dtype}")
    return model, tokenizer, device


def load_vllm_model(model_name: str, max_model_len=8192, tensor_parallel_size=None):
    """
    加载 vLLM 模型（离线批量推理）。

    Args:
        model_name: 模型名称或路径
        max_model_len: 最大序列长度

    Returns:
        vllm.LLM 实例
    """
    from vllm import LLM

    resolved = resolve_model_path(model_name)
    print(f"  加载 vLLM 模型: {model_name} (path={resolved})")

    kwargs = {
        "model": resolved,
        "trust_remote_code": True,
        "dtype": "half",
        "max_model_len": max_model_len,
    }
    if tensor_parallel_size:
        kwargs["tensor_parallel_size"] = tensor_parallel_size

    llm = LLM(**kwargs)

    print(f"  vLLM 模型加载完成")
    return llm


def get_num_gpus() -> int:
    """返回可用 GPU 数量"""
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 0


def split_list(lst, n):
    """将列表均分为 n 份"""
    k = math.ceil(len(lst) / n)
    return [lst[i * k:(i + 1) * k] for i in range(n)]


def make_sampling_params(max_tokens: int, enable_thinking: bool = False):
    """
    返回 vLLM SamplingParams。
    thinking 模式用 Qwen3 官方推荐采样参数；非 thinking 用贪心（temperature=0）。
    """
    from vllm import SamplingParams
    if enable_thinking:
        # Qwen3.5 思考模式易陷入 "Wait... 再想想..." 的反复重新考虑循环（官方 issue），
        # 习语翻译这种本身有 字面/意译 歧义的任务尤其严重，模型在候选间无限振荡。
        # presence_penalty 只惩罚"是否出现过"（一次性），对 "Wait" 被刷几十次效果有限；
        # frequency_penalty 按出现次数累加惩罚，专治这种重复。官方文本任务推荐 presence≈1.5。
        # 若仍语言混杂就调低，仍循环就把 frequency 往 1.0 提。
        return SamplingParams(temperature=0.6, top_p=0.95, top_k=20,
                              presence_penalty=1.5, frequency_penalty=0.5,
                              max_tokens=max_tokens)
    return SamplingParams(temperature=0, max_tokens=max_tokens)
