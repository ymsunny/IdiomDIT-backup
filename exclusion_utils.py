"""
exclusion_utils.py — 通用工具：加载 exclusion_key.json 并过滤 cascade 分析数据。

用法（在 cascade 分析脚本里）：
    from exclusion_utils import load_exclusion_set, filter_cascade_rows

    excluded = load_exclusion_set(lang_pair, model)   # set of (id, prompt_type)
    rows = filter_cascade_rows(rows, excluded)
"""
import json
from pathlib import Path
from typing import Set, Tuple, Iterable, List

_BASE = Path(__file__).resolve().parent


def load_exclusion_set(lang_pair: str, model: str) -> Set[Tuple[int, str]]:
    """加载 (lang_pair × model) 的 exclusion key，返回 (id, prompt_type) 集合。"""
    path = _BASE / "results" / lang_pair / model / "exclusion_key.json"
    if not path.is_file():
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {(int(e["id"]), e["prompt_type"]) for e in data}


def is_excluded(row_id: int, prompt_type: str, excluded_set: Set[Tuple[int, str]]) -> bool:
    return (int(row_id), prompt_type) in excluded_set


def filter_cascade_rows(rows: Iterable[dict], excluded_set: Set[Tuple[int, str]],
                        id_key: str = "id", prompt_key: str = "prompt_type") -> List[dict]:
    """过滤 cascade rows，排除 (id, prompt_type) 在 excluded_set 里的。"""
    return [r for r in rows if (int(r[id_key]), r[prompt_key]) not in excluded_set]
