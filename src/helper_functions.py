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
    pre_prompt += "Available function format: fn_name(args: args_type)"\
                  "-> return_type\nAll functions: \n"
    pre_prompt += "Choose from the following functions "\
                  "to answer user question.\n"
    for fn in functions:
        pre_prompt += f"{fn.fn_name}("
        arg_with_types = ""
        for a, t in fn.args_types.items():
            arg_with_types += f"{a}: {t}, "
        pre_prompt += f"{arg_with_types[:-2]})"
        pre_prompt += f" -> {fn.return_type}\n"
        # pre_prompt += ""

    # example1 = "Example 1: 'prompt': 'Greet Sukanta' -> 'fn_name': 'fn_greet', 'args': {'name': 'Sukanta'}\n"
    # example2 = "Example 2: 'prompt': 'Substitute the r'\d+' in the string 'Hello 34 I'm 233 years old' with NUMBERS' -> 'fn_name': 'fn_substitute_string_with_regex', 'args': {'source_string': 'Hello 34 I'm 233 years old', regex: r'\d+', 'replacement': 'NUMBER'}\n"
    # pre_prompt += example1
    # pre_prompt += example2

    # pre_prompt = ""
    # for fn_name, args in zip(data_str['fn_name'], data_str['args_types']):
    #     pre_prompt += f"{fn_name}, args: {args}\n"
    # print(pre_prompt)
    # json_txt = str(load_json())
    # pre_prompt = f"Allowed functions: {json_txt}"

    # pre_prompt = f"Allowed name: {','.join(func)}"
    # pre_prompt += f"arguments: {str(args)}"
    # print(pre_prompt)
    # pre_prompt = ""

    # question = f"Question: {prompt}\n"
    combined_prompt = f"{pre_prompt}\n"
    # print(f"Initial tokes: {combined_prompt}")
    tokens = encode(combined_prompt)
    # print(tokens)
    return tokens


def tokenize_string(string: str, llm) -> List[int]:
    return llm._encode(string)


def is_valid_num(val: str) -> bool:
    if val == ".":
        return True
    try:
        int(val)
        return True
    except ValueError:
        return False


def split_word(string: str) -> List[str]:
    str_len = len(string)
    tokens = []
    i = 0
    try:
        while (i < str_len):
            sub_str = ""
            if string[i] == "'":
                sub_str += string[i]
                i += 1
                while string[i] != "'" and i < str_len:
                    sub_str += string[i]
                    i += 1
                if i < str_len:
                    sub_str += string[i]
                    i += 1
            elif string[i] == '"':
                sub_str += string[i]
                i += 1
                while string[i] != '"' and i < str_len:
                    sub_str += string[i]
                    i += 1
                if i < str_len:
                    sub_str += string[i]
                    i += 1
            else:
                while i < str_len and string[i].isalpha():
                    sub_str += string[i]
                    i += 1
            if len(sub_str) > 0:
                tokens.append(sub_str)
            else:
                tokens.append(string[i])
                i += 1
        return tokens
    except IndexError as e:
        print(f"Index Error: {e}")
        return tokens
