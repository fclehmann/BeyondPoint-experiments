from collections import defaultdict
from typing import Any, Callable
import torch


def untensor_dict(d: dict[str, Any]) -> dict[str, Any]:
    r = {}
    for k, v in d.items():
        r[k] = v.item() if isinstance(v, torch.Tensor) else v
    return r

def dicts_mean(dicts: list[dict[str, float]]) -> dict[str, float]:
    assert len(dicts) > 0
    r: dict[str, float] = defaultdict(0.0)  # type: ignore

    for d in dicts:
        for k in dicts[0].keys():
            r[k] += d[k]

    for k in dicts[0].keys():
        r[k] /= len(dicts)

    return r


def modify_dict_keys(d: dict[str, Any], f: Callable[[str], str]) -> dict[str, Any]:
    r = {}
    for k, v in d.items():
        r[f(k)] = v
    return r