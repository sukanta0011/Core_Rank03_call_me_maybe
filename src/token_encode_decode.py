import time
import json
import sys
from typing import Dict
import numpy as np
from pydantic import BaseModel
from memory_profiler import profile


# class TokenEncodeDecode(BaseModel):
#     path: str
#     decoder = []
#     encoder = {}
#     with open(path, 'r') as fl:
#         data = json.load(fl)
#     for key, val in data.items():
#         token_dict[val] = key
#         token_list.append(key)


def char_feq(data: Dict) -> Dict[str, int]:
    char_hash = {}
    for key, _ in data.items():
        for c in key:
            if c in char_hash:
                char_hash[c] += 1
            else:
                char_hash[c] = 1
    return char_hash


def test():
    start = time.time()
    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model()
    token_path = llm.get_path_to_vocabulary_json()
    # data = []
    with open(token_path, 'r') as fl:
        data = json.load(fl)
    char_dict = char_feq(data)
    # sorted_dict = sorted(char_dict, key=lambda x: x[x.key()])
    print(char_dict)
    print(len(char_dict))
    # token_list = []
    # token_dict = {}
    # for key, val in data.items():
    #     token_dict[val] = key
    #     token_list.append(key)
    # print(f"memory data: {sys.getsizeof(data) // 1024} MB")
    # print(f"memory list: {sys.getsizeof(token_list) // 1024} MB")
    # print(f"memory: {sys.getsizeof(token_dict) // 1024} MB")
    # print(f"Token len: {len(token_list)}, {token_list[-1]}")
    end = time.time()
    print(f"Time taken: {(end - start):.3f}s")


if __name__ == "__main__":
    test()
