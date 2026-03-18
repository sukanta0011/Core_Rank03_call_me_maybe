import numpy as np
from typing import List, Dict, Callable
from src.parser import FnInfo


def show_toke_distribution(data: List[float], bin_spacing: int):
    min_val = min(data)
    max_mal = max(data)
    bins = np.arange(min_val, max_mal, bin_spacing)
    counts, _ = np.histogram(data, bins)
    for b, c in zip(bins, counts):
        print(f"{b}: {c}")


def char_feq(data: Dict) -> Dict[str, int]:
    char_hash: Dict[str, int] = {}
    for key, _ in data.items():
        for c in key:
            if c in char_hash:
                char_hash[c] += 1
            else:
                char_hash[c] = 1
    return char_hash


def initial_prompt_toke(prompt: str, functions: List[FnInfo],
                        encode: Callable) -> List[int]:
    pre_prompt = ""
    pre_prompt += "Available functions with description about the function:\n"
    for fn in functions:
        pre_prompt += f"{fn.fn_name}: {fn.description}\n"
        # pre_prompt += f"{fn.fn_name}("
        # arg_with_types = ""
        # for a, t in fn.args_types.items():
        #     arg_with_types += f"{a}: {t}, "
        # pre_prompt += f"{arg_with_types[:-2]})"
        # pre_prompt += f" -> {fn.return_type}\n\n"

    pre_prompt += (
        "Example: 'Question': 'Greet Sukanta' -> "
        "'fn_name': 'fn_greet', 'args': {'name': 'Sukanta'}\n"
        )

    tokens = encode(pre_prompt)
    return tokens


def is_valid_num(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False
