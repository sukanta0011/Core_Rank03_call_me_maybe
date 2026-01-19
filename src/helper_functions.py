import json
import time
import numpy as np
import os
import sys
from typing import List


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


def decode_tokens(tokens: List[float], threshold: int):
    path = "/home/sudas/.cache/huggingface/hub/" \
           "models--Qwen--Qwen3-0.6B/snapshots/" \
           "c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"
    with open(path, "r") as fl:
        data = json.loads(fl.read())
    high_prob_tokens = [idx for idx, token in
                        enumerate(tokens) if token > threshold]
    high_prob = [token for idx, token in
                        enumerate(tokens) if token > threshold]
    print(high_prob_tokens)

    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model()
    decoder = llm._decode(high_prob_tokens)
    for t, p in zip(decoder, high_prob):
        print(f"{t}: {p}")