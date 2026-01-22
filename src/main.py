import json
import time
import numpy as np
import os
import sys
from typing import List

from helper_functions import load_json

def test_llm():
    # start = time.time()

    from llm_sdk import Small_LLM_Model

    start = time.time()
    llm = Small_LLM_Model()
    # file_path = small_llm.get_path_to_vocabulary_json()
    # print(file_path)
    txt = "What is the addition of 2 and 3?"
    json_txt = str(load_json())
    pre_prompt = "You need to act as function generator\n After reading the " \
                 "user question, your job will be to provide the answer\n"\
                 "Answer format will be: prompt: question asked by the user \n fn_name: applicable function name\n args: required arguments with types"\
                 f"Available function: {json_txt}\n"

    question = f"Question: {txt}\n"
    answer = ""
    func_tokens = llm._encode(json_txt).tolist()

    for i in range(100):
        print(answer)
        prompt = f"{pre_prompt} + {question} + {answer}"
        tokens = llm._encode(prompt).tolist()
        logits = llm.get_logits_from_input_ids(tokens[0])
        logits_dict = {idx: tkn for idx, tkn in enumerate(logits)}

        max_val = -100
        max_key = -1
        for key, val in logits_dict.items():
            if val > max_val:
                max_val = val
                max_key = key
        decoded_token = llm._decode([max_key])
        answer += str(decoded_token)
        if "}" in decoded_token:
            break


    # tokens = ["{", "fn", "_name"]

    # path = "/home/sudas/.cache/huggingface/hub/" \
    #        "models--Qwen--Qwen3-0.6B/snapshots/" \
    #        "c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"
    # with open(path, "r") as fl:
    #     data = json.loads(fl.read())

    # token_list = [data.get(token) for token in tokens if data.get(token) is not None]
    # print(token_list)

    # logits = llm.get_logits_from_input_ids(token_list)
    # print(max(logits), min(logits))
    # print(data)

    end = time.time()
    print(f"Testing the LLM, Time: {(end - start):.3f}")


def extract_probability():
    with open("src/demo.txt", "r") as fl:
        data = fl.read()
    data = data[1: -1].split(",")
    data = [float(d) for d in data]
    return data


if __name__ == "__main__":
    test_llm()
    # extract_probability()
    # data = extract_probability()
    # decode_tokens(data, 10)
