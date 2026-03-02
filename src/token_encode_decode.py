import time
import json
import sys
from typing import Dict, List
import numpy as np
from pydantic import BaseModel, PrivateAttr
from memory_profiler import profile


class TokenEncodeDecode(BaseModel):
    path: str
    _decoder_list: List[str] = PrivateAttr(default_factory=list)
    _encoder_dict: Dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        """Called automatically after the object is created."""
        self._create_encoder_decoder(self.path)

    def _create_encoder_decoder(self, path: str):
        with open(path, 'r') as fl:
            data = json.load(fl)
        self._decoder_list = [""] * (len(data.values()) + 1)
        for key, val in data.items():
            new_key = key.replace("Ġ", " ")
            self._encoder_dict[new_key] = val
            self._decoder_list[val] = new_key
        del data

    def encoder(self, string: str) -> List[int]:
        tokens: List[int] = []
        str_len = len(string)
        max_tkn_len = len(max(self._decoder_list))
        left_ptr = 0
        if str_len > max_tkn_len:
            right_ptr = left_ptr + max_tkn_len
        else:
            right_ptr = str_len
        print(f"Max token len: {max_tkn_len}, max_token: {max(self._decoder_list)}")
        while left_ptr < right_ptr:
            sub_str = string[left_ptr: right_ptr]
            if sub_str in self._encoder_dict.keys():
                print(f"Match found: {sub_str}, {left_ptr}, {right_ptr}")
                tokens.append(self._encoder_dict[sub_str])
                left_ptr = right_ptr
                if left_ptr < str_len:
                    # right_ptr = str_len
                    if (str_len - left_ptr) < max_tkn_len:
                        right_ptr = str_len
                    else:
                        right_ptr = left_ptr + max_tkn_len
            else:
                right_ptr -= 1
        return tokens

    def decode(self, tokens: List[int]) -> str:
        string = ""
        for token in tokens:
            try:
                string += self._decoder_list[token]
            except IndexError as e:
                string = ""
                raise IndexError(f"Token ID {token}, not found: {e}")
        return string


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
    sorted_dict = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dict)
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


def test_toke_encoder():
    start = time.time()
    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model()
    token_path = llm.get_path_to_vocabulary_json()

    encoder_decoder = TokenEncodeDecode(path=token_path)
    # msg = "This is a classic 'hardware vs. software' version mismatch often seen in older workstations. Your NVIDIA GT 1030 has a Pascal architecture (Compute Capability 6.1), but modern versions of PyTorch (like the 2.9.1 required in your dependencies) have dropped support for anything older than Volta (7.0) in their pre-compiled binaries. The workstation's GPU (GT 1030) has a Compute Capability of 6.1, which is deprecated in PyTorch 2.x. To ensure stability and graceful error handling as required by the subject, I forced the model to CPU mode, which still meets the < 5 minute processing requirement for the test prompts."
    msg = "I love coding."
    start_1 = time.time()
    print(f"Built-in encoder: {llm._encode(msg)}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")
    start_1 = time.time()
    tokens = encoder_decoder.encoder(msg)
    print(f"Custom encoder: {tokens}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")

    start_1 = time.time()
    print(f"Built-in decoder: {llm._decode(tokens)}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")
    start_1 = time.time()
    print(f"Custom decoder: {encoder_decoder.decode(tokens)}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")

    end = time.time()
    print(f"Time taken: {(end - start):.3f}s")


if __name__ == "__main__":
    # test()
    test_toke_encoder()
