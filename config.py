"""
config.py — 项目统一配置（路径 + 共享常量）

三层架构：
  Layer 1: LANG_PAIRS  — 语言对数据注册表
  Layer 2: TARGET_CONFIG — 实验目标配置（从 LANG_PAIRS 派生）
  Layer 3: get_data_paths() / get_config() — 公共查询接口

新增语言对只需：
  1. LANG_NAMES 加语言代码
  2. LANG_PAIRS 加数据条目
  3. TARGET_CONFIG 加一行 _make_target()
"""

import os
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# 自动加载项目根目录的 .env 文件（.env 优先，覆盖系统环境变量）
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

_BASE = Path(__file__).resolve().parent

# ==================== 自动加载 .env ====================

_env_path = _BASE / ".env"
if _env_path.is_file():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ==================== 语言名称注册 ====================

LANG_NAMES = {
    "en": "English",
    "fa": "Persian",
    "zh": "Chinese",
    "ja": "Japanese",
    "fr": "French",
    "fi": "Finnish",
    "ko": "Korean",
}


def _lang_name(code: str) -> str:
    if code not in LANG_NAMES:
        raise ValueError(f"Unknown language code: {code!r}. Known: {list(LANG_NAMES.keys())}")
    return LANG_NAMES[code]


# ==================== Layer 1: 语言对数据注册表 ====================
# 每个方向存精确文件名（不做推导），因为命名不一致

LANG_PAIRS: Dict[str, Dict[str, Any]] = {
    "fa-en": {
        "dataset_dir": "Fa-En-Idiom",
        "directions": [
            {"src": "en", "tgt": "fa", "csv": "En_Fa.csv", "meaning": "En_Fa_meanings.json"},
            {"src": "fa", "tgt": "en", "csv": "Fa_En.csv", "meaning": "Fa_En_meanings.json"},
        ],
    },
    "ja-en": {
        "dataset_dir": "Ja-En-Idiom",
        "directions": [
            {"src": "ja", "tgt": "en", "csv": "Ja_En_clean.csv", "meaning": "Ja_En_meanings.json"},
        ],
    },
    "fr-en": {
        "dataset_dir": "Fr-En-Idiom",
        "directions": [
            {"src": "fr", "tgt": "en", "csv": "Fr_En_clean.csv", "meaning": "Fr_En_meanings.json"},
        ],
    },
    "fi-en": {
        "dataset_dir": "Fi-En-Idiom",
        "directions": [
            {"src": "fi", "tgt": "en", "csv": "Fi_En_clean.csv", "meaning": "Fi_En_meanings.json"},
        ],
    },
    "ko-en": {
        "dataset_dir": "Ko-En-Idiom",
        "directions": [
            {"src": "ko", "tgt": "en", "csv": "Ko_En_clean.csv", "meaning": "Ko_En_meanings.json"},
        ],
    },
}


def get_data_paths(pair: str, direction: Optional[str] = None) -> Dict[str, Any]:
    """
    按语言对 + 方向查询数据路径。

    Args:
        pair: 语言对代码，如 "fa-en", "ja-en"
        direction: 方向代码，如 "en-fa", "fa-en"。单向语言对可省略。

    Returns:
        dict: data_dir, meaning_dir, csv_file, meaning_file, src_lang, tgt_lang, ...
    """
    if pair not in LANG_PAIRS:
        raise ValueError(f"Unknown pair: {pair!r}. Known: {list(LANG_PAIRS.keys())}")

    lp = LANG_PAIRS[pair]
    dataset_dir = _BASE / "data" / lp["dataset_dir"]
    data_dir = dataset_dir / "ParallelData"
    meaning_dir = dataset_dir / "Meaning"

    directions = lp["directions"]
    if direction is None:
        if len(directions) == 1:
            d = directions[0]
        else:
            avail = [f"{dd['src']}-{dd['tgt']}" for dd in directions]
            raise ValueError(f"Pair {pair!r} is bidirectional. Specify direction: {avail}")
    else:
        parts = direction.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid direction: {direction!r}. Expected 'xx-yy'.")
        d = next((dd for dd in directions if dd["src"] == parts[0] and dd["tgt"] == parts[1]), None)
        if d is None:
            avail = [f"{dd['src']}-{dd['tgt']}" for dd in directions]
            raise ValueError(f"Direction {direction!r} not in pair {pair!r}. Available: {avail}")

    return {
        "pair": pair,
        "data_dir": data_dir,
        "meaning_dir": meaning_dir,
        "csv_file": data_dir / d["csv"],
        "meaning_file": meaning_dir / d["meaning"],
        "src_code": d["src"],
        "tgt_code": d["tgt"],
        "src_lang": _lang_name(d["src"]),
        "tgt_lang": _lang_name(d["tgt"]),
        "direction": f"{_lang_name(d['src'])}_to_{_lang_name(d['tgt'])}",
    }


# ==================== 模型配置 ====================

MODEL_BASE_DIR = "/pretrain-models"
MODEL_NAME = "Qwen3.5-4B"
ENABLE_THINKING = True
MAX_NEW_TOKENS = 512
# thinking 模式下 chain-of-thought 会额外消耗大量 token，需单独设置上限
MAX_NEW_TOKENS_THINKING_DETECTION    = 4096   # detection: Qwen3.5 即使简单任务也过度思考(~2000 tok)，2048 会把答案 JSON 截断；4096 留余量
MAX_NEW_TOKENS_THINKING_INTERP       = 4096   # interpretation: 同上，思考易超 2000 tok
MAX_NEW_TOKENS_THINKING_TRANSLATION  = 4096   # translation: CoT + 实际译文

# 需要全部 GPU 的大模型（单卡放不下）
LARGE_MODELS = {"Llama-3.3-70B-Instruct"}

def is_thinking_model(model_name: str) -> bool:
    """是否支持 enable_thinking 参数（Qwen3 系列全部支持，包括 Qwen3 base 和 Qwen3.5）"""
    n = model_name.lower()
    return "qwen3" in n


def resolve_model_path(model_name: str) -> str:
    """
    解析模型路径：
    - model_name 已是存在的目录 → 直接用
    - MODEL_BASE_DIR / model_name 存在 → 用拼接路径
    - 都不存在 → 返回原始名称（交给 transformers 处理）
    -thinking 后缀会被自动去掉（仅用于区分输出目录，不影响实际权重路径）
    """
    base = model_name[:-len("-thinking")] if model_name.endswith("-thinking") else model_name
    if os.path.isdir(base):
        return base
    full = os.path.join(MODEL_BASE_DIR, base)
    if os.path.isdir(full):
        return full
    return base

# ==================== API 配置（LLM Judge）====================

# 自动加载项目根目录下的 .env（如果存在且环境变量未设置）
_dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_dotenv):
    for _line in open(_dotenv, encoding="utf-8"):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY")
JUDGE_MODEL = "gpt-4o-mini"
API_TIMEOUT = 60
API_TEMPERATURE = 0
API_MAX_TOKENS = 512
API_MAX_RETRIES = 3
API_RETRY_DELAY = 5
API_REQUEST_INTERVAL = 0

# ==================== Prompt 模板 ====================

PROMPTS = {
    "BasicPrompt":
        "Translate the following sentences from [SRC] to [TGT].",
    "Anti-LiteralConstraintPrompt":
        "Translate the following sentences from [SRC] to [TGT], avoiding being a literal translation.",
    "Meaning-FirstTranslationPrompt":
        "Translate the following sentences from [SRC] to [TGT], ensuring that the translation conveys the intended meaning accurately.",
    "SuperLiteralBiasMitigationPrompt":
        "Translate the following sentences from [SRC] to [TGT], ensuring that the translation conveys the intended meaning accurately and avoiding being a literal translation.",
}

ALL_PROMPT_TYPES = list(PROMPTS.keys())

# ==================== 结果目录 ====================

DETECTION_DIR = _BASE / "Results" / "Detection"
INTERPRETATION_DIR = _BASE / "Results" / "Interpretation"

# ==================== 向后兼容（指向 Fa-En，逐步废弃）====================

DATA_DIR = _BASE / "Data" / "Fa-En-Idiom" / "ParallelData"
MEANING_DIR = _BASE / "Data" / "Fa-En-Idiom" / "Meaning"

# ==================== Layer 2: 实验目标配置 ====================


def _subdir(src_code: str, tgt_code: str) -> str:
    """en, fa -> 'En_Fa'"""
    return f"{src_code.capitalize()}_{tgt_code.capitalize()}"


def _make_target(
    src: str, tgt: str,
    data_pair: str, data_direction: str,
    translation_subdir: Optional[str] = None,
    det_subdir: Optional[str] = None,
    interp_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    从 LANG_PAIRS 派生实验目标配置。

    Args:
        src, tgt: 翻译方向（如 "en", "fa"）
        data_pair: 数据来源语言对（如 "fa-en"）
        data_direction: 数据方向（如 "en-fa"）
        translation_subdir: 翻译结果子目录名，默认自动生成
        det_subdir: 检测结果子目录名覆盖（用于跨语言复用）
        interp_subdir: 释义结果子目录名覆盖（用于跨语言复用）
    """
    dp = get_data_paths(data_pair, data_direction)
    t_sub = translation_subdir or _subdir(src, tgt)
    d_sub = det_subdir or _subdir(dp["src_code"], dp["tgt_code"])
    i_sub = interp_subdir or _subdir(dp["src_code"], dp["tgt_code"])

    return {
        # 数据路径
        "data_pair": data_pair,
        "data_direction": data_direction,
        "csv_file": dp["csv_file"],
        "meaning_file": str(dp["meaning_file"]),
        # 翻译方向
        "direction": f"{_lang_name(src)}_to_{_lang_name(tgt)}",
        "src_lang": _lang_name(src),
        "tgt_lang": _lang_name(tgt),
        "src_code": src,
        "tgt_code": tgt,
        # 翻译结果路径
        "translation_input": _BASE / "Results" / "Translation" / t_sub,
        "eval_output": _BASE / "Results" / "Translation" / t_sub / "LTEEvaluation",
        "cascade_output": _BASE / "Results" / "Translation" / t_sub / "CascadeAnalysis",
        # 检测 / 释义（可跨目标复用）
        "interpretation_direction": f"{_lang_name(dp['src_code'])}_to_{_lang_name(dp['tgt_code'])}",
        "interpretation_subdir": i_sub,
        "detection_subdir": d_sub,
    }


TARGET_CONFIG = {
    "en-fa":     _make_target("en", "fa", "fa-en", "en-fa"),
    "fa-en":     _make_target("fa", "en", "fa-en", "fa-en"),
    "ja-en":     _make_target("ja", "en", "ja-en", "ja-en"),
    "fr-en":     _make_target("fr", "en", "fr-en", "fr-en"),
    "fi-en":     _make_target("fi", "en", "fi-en", "fi-en"),
    "ko-en":     _make_target("ko", "en", "ko-en", "ko-en"),
}

ALL_TARGETS = list(TARGET_CONFIG.keys())


# ==================== Layer 3: 公共查询接口 ====================

def get_target_arg(extra_args=None):
    """解析命令行 --lang-pair 参数"""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lang-pair", required=True, choices=ALL_TARGETS,
                        help=f"语言对: {', '.join(ALL_TARGETS)}")
    if extra_args:
        for arg in extra_args:
            parser.add_argument(*arg[0], **arg[1])
    args, _ = parser.parse_known_args()
    return args


def get_config(target: str, model: Optional[str] = None) -> dict:
    """
    获取完整的目标配置（含派生路径）。

    Args:
        target: 目标代码，如 "fi-en"
        model: 模型名称，如 "Qwen3.5-4B"。
               提供时使用新结构: results/{target}/{model}/...
               不提供时使用旧结构: Results/Detection/{subdir}/...（向后兼容）
    """
    if target not in TARGET_CONFIG:
        raise ValueError(f"未知 target: {target!r}，可选: {ALL_TARGETS}")
    cfg = TARGET_CONFIG[target].copy()
    cfg["target"] = target

    if model:
        # 新结构: results/{target}/{model}/
        base = _BASE / "results" / target / model
        cfg["detection_dir"] = base / "detection"
        cfg["interpretation_dir"] = base / "interpretation"
        cfg["translation_input"] = base / "translation"
        cfg["eval_output"] = base / "evaluation"
        cfg["analysis_output"] = base / "analysis"
        cfg["mechanistic_dir"] = base / "mechanistic"
        cfg["intervention_dir"] = base / "intervention"
        cfg["methods_dir"] = base / "methods"
    else:
        # 旧结构（向后兼容）
        cfg["detection_dir"] = DETECTION_DIR / cfg["detection_subdir"]
        cfg["interpretation_dir"] = INTERPRETATION_DIR / cfg["interpretation_subdir"]

    if model:
        # 新结构：评估结果在 evaluation/ 下
        cfg["interpretation_json"] = cfg["eval_output"] / "interpretation_p1_score.json"
        cfg["interpretation_p2_json"] = cfg["eval_output"] / "interpretation_p2_score.json"
    else:
        # 旧结构
        cfg["interpretation_json"] = cfg["interpretation_dir"] / "interpretation_P1_detail.json"
        cfg["interpretation_p2_json"] = cfg["interpretation_dir"] / "interpretation_P2_detail.json"
    return cfg


def setup_logger(name: str, lang_pair: Optional[str] = None,
                 model: Optional[str] = None,
                 log_filename: Optional[str] = None) -> logging.Logger:
    """
    统一日志配置。

    Args:
        name: logger 名称（内部标识，保证唯一）
        lang_pair: 语言对，如 "fi-en"
        model: 模型名，如 "Qwen3.5-4B"
        log_filename: 日志文件名（不含路径），默认 {name}.log

    用法:
        logger = setup_logger("detection", lang_pair="fi-en", model="Qwen3.5-4B")
        → results/fi-en/Qwen3.5-4B/detection.log

        logger = setup_logger("det_fi-en", lang_pair="fi-en", model="Qwen3.5-4B",
                              log_filename="detection.log")
        → results/fi-en/Qwen3.5-4B/detection.log（name 不同但文件名固定）
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if lang_pair and model:
        log_dir = _BASE / "results" / lang_pair / model
        os.makedirs(log_dir, exist_ok=True)
        fname = log_filename or f"{name}.log"
        fh = logging.FileHandler(str(log_dir / fname), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
