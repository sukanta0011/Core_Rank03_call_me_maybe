import json
import numpy as np
from typing import List, Dict, Callable
from src.parser import Flags, ResourcePath
from src.parser import FnInfo, Prompts
from src.custom_errors import CLIParsingError, SourceError


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


# def initial_prompt_toke(prompt: str, llm, tokenizer) -> List[int]:
#     json_txt = str(load_json())
#     pre_prompt = "You need to act as function generator\n After reading the " \
#                  "user question, your job will be to provide the function "\
#                  f"name.\nAvailable function names are: {json_txt}\n"

#     question = f"Question: {prompt}\n"
#     combined_prompt = f"{pre_prompt}\n{question}"
#     # print(f"Initial tokes: {combined_prompt}")
#     tokens = tokenizer.encode(combined_prompt)
#     return tokens

def initial_prompt_toke(prompt: str, functions: List[FnInfo],
                        encode: Callable) -> List[int]:
    pre_prompt = ""
    # pre_prompt += "Available function format: fn_name(args: args_type)"\
    #               "-> return_type\nAll functions: \n"
    # pre_prompt += "Available functions with description about the function:\n"
    for fn in functions:
        pre_prompt += f"{fn.fn_name}: {fn.description}\n"
        # pre_prompt += f"{fn.fn_name}("
        # arg_with_types = ""
        # for a, t in fn.args_types.items():
        #     arg_with_types += f"{a}: {t}, "
        # pre_prompt += f"{arg_with_types[:-2]})"
        # pre_prompt += f" -> {fn.return_type}\n\n"

    pre_prompt += "Example: 'Question': 'Greet Sukanta' -> 'fn_name': 'fn_greet', 'args': {'name': 'Sukanta'}\n"
    # pre_prompt += "Example: 'Question': 'sum of -5 and 7?' -> 'fn_name': 'fn_add_numbers', 'args': {'a': -5, 'b': 7}\n"
    # example2 = "Example: 'prompt': 'Substitute the 'digits' in the string 'Hello 34 I'm 233 years old' with NUMBERS' -> 'args': {'source_string': 'Hello 34 I'm 233 years old', regex: r'\\d', 'replacement': 'NUMBER'}\n"
    # example3 = "Example: Replace consonants in 'Programming is fun' with hash -> 'args': {'source_string': 'Programming is fun', regex: '^[a|e|i|o|u]', 'replacement': '#'}\n"

    # print(pre_prompt)
    tokens = encode(pre_prompt)
    return tokens


def is_valid_num(val: str) -> bool:
    if val == ".":
        return True
    try:
        int(val)
        return True
    except ValueError:
        return False
