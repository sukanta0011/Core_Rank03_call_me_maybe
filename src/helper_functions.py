import json
import numpy as np
from typing import List, Dict
from src.tokenizer import Tokenizer


def load_json():
    path = "data/input/functions_definition.json"
    with open(path, "r") as fl:
        data = json.load(fl)
    return data


def show_toke_distribution(data: List[float], bin_spacing: int):
    min_val = min(data)
    max_mal = max(data)
    bins = np.arange(min_val, max_mal, bin_spacing)
    counts, _ = np.histogram(data, bins)
    for b, c in zip(bins, counts):
        print(f"{b}: {c}")


def show_high_probability_tokens(data: List[float], no: int):
    pass


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

def initial_prompt_toke(prompt: str, func: List[str],
                        args: List[str],
                        tokenizer: Tokenizer) -> List[int]:
    # from src.parser import Parser
    # path = "data/input/functions_definition.json"
    # parser = Parser()
    # data_str = parser.load_json(path)
    # # pre_prompt = f"Allowed functions: {data_str['fn_name']}, {data_str['args_types']}"
    # pre_prompt = "Allowed functions[arguments]\n"
    # for fn_name, args, args_type in zip(data_str['fn_name'], data_str['args_names'], data_str['args_types']):
    #     pre_prompt += f"{fn_name}["
    #     arg_with_types = ""
    #     for a, t in zip(args, args_type):
    #         arg_with_types += f"{a}:{t},"
    #     pre_prompt += f"{arg_with_types[:-1]}]\n"

    # pre_prompt = ""
    # for fn_name, args in zip(data_str['fn_name'], data_str['args_types']):
    #     pre_prompt += f"{fn_name}, args: {args}\n"
    # print(pre_prompt)
    json_txt = str(load_json())
    pre_prompt = f"Allowed functions: {json_txt}"

    # pre_prompt = f"Allowed name: {','.join(func)}"
    # pre_prompt += f"arguments: {str(args)}"
    print(pre_prompt)
    # pre_prompt = ""

    question = f"Question: {prompt}\n"
    combined_prompt = f"{pre_prompt}\n{question}"
    # print(f"Initial tokes: {combined_prompt}")
    tokens = tokenizer.encode(combined_prompt)
    return tokens


def tokenize_string(string: str, llm) -> List[int]:
    return llm._encode(string).tolist()[0]


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
