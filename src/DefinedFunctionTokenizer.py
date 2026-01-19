import json
import time
# from pydantic import BaseModel
from typing import List, Dict, Set
from src.helper_functions import load_json


def initial_prompt_toke(llm):
    txt = "What is the multiplication of 2 and 8?"
    json_txt = str(load_json())
    pre_prompt = "You need to act as function generator\n After reading the " \
                 "user question, your job will be to provide the function name\n"\
                 f"Available function names are: {json_txt}\n"

    question = f"Question: {txt}\n"
    answer = ""
    prompt = f"{pre_prompt} + {question} + {answer}"
    tokens = llm._encode(prompt).tolist()
    return tokens[0]


def split_word(string: str) -> List[str]:
    str_len = len(string)
    tokens = []
    i = 0
    try:
        while (i < str_len):
            sub_str = ""
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


class DefinedFunctionTokenizer:
    def __init__(self):
        self.data_str = {}
        self.data_token = {}
        self.data_token_by_word = {}

    def tokenize_string(self, string: str, llm) -> List[int]:
        return llm._encode(string).tolist()[0]

    def load_json(self, file_path: str) -> Dict:
        with open(file_path, "r") as fl:
            data = json.load(fl)

        for d in data:
            for key, val in d.items():
                if key not in self.data_str.keys():
                    self.data_str[key] = set()
                if isinstance(val, list):
                    for lst_data in val:
                        self.data_str[key].add(lst_data)
                elif isinstance(val, dict):
                    for _, v in val.items():
                        self.data_str[key].add(v)
                else:
                    self.data_str[key].add(val)

        self.data_str["special_char"] = set()
        for char in str(data):
            if not char.isalpha() and not char.isalnum():
                self.data_str["special_char"].add(char)
        return self.data_str

    def tokenize_json_by_llm(self, llm) -> Dict:
        for k, v in self.data_str.items():
            if k not in self.data_token.keys():
                self.data_token[k] = []
            for word in v:
                tokens = llm._encode(word).tolist()
                # for token in tokens[0]:
                self.data_token[k].append(tokens[0])
        # print(self.data_token)
        return self.data_token

    def tokenize_json_by_word(self, llm) -> Dict:
        for k, v in self.data_str.items():
            if k not in self.data_token_by_word.keys():
                self.data_token_by_word[k] = []
            for word in v:
                tokens = []
                sub_words = split_word(word)
                for sub_word in sub_words:
                    print(f"Sub word: {sub_word}")
                    try:
                        token = llm._encode(sub_word).tolist()
                        print(f"Tokens: {token[0]}")
                        for sub_token in token[0]:
                            tokens.append(token[0][0])
                    except Exception as e:
                        print(f"{e}")
                self.data_token_by_word[k].append(tokens)
                
                    # self.data_token_by_word[k].append(token)
        # print(self.data_token)
        return self.data_token_by_word


class TokenMasking:
    def __init__(self, allowed_token_bank: Dict, prompt_tokens: List[int], llm) -> None:
        self.allowed_toke = allowed_token_bank
        self.prompt_tokens = prompt_tokens
        self.llm = llm

    def generate_function_name(self):
        complete_fn_tokens = []
        token = float("-inf")
        try:
            while True:
                print(llm._decode(complete_fn_tokens))
                logits = llm.get_logits_from_input_ids(self.prompt_tokens)
                next_allowed_tokens = set()
                complete_fn_len = len(complete_fn_tokens)
                for fn in self.allowed_toke['fn_name']:
                    if complete_fn_len == 0:
                        if len(fn) > 0:
                            next_allowed_tokens.add(fn[0])
                    elif token in fn:
                        idx = fn.index(token)
                        if len(fn) > idx + 1:
                            next_allowed_tokens.add(fn[idx + 1])
                    elif self.list_compare(fn, complete_fn_tokens):
                        return complete_fn_tokens
                # print(f"allowed token: {llm._decode(list(next_allowed_tokens))}")
                token = self.get_next_token(logits, next_allowed_tokens)
                complete_fn_tokens.append(token)
                self.prompt_tokens.append(token)
        except Exception as e:
            print(e)
            return

    def get_next_token(self, logits: List[float],
                       allowed_idx: Set[int]) -> int:
        max_prob = float("-inf")
        max_prob_idx = -1
        print()
        for token in allowed_idx:
            print(f"{token}, {self.llm._decode([token])}, {logits[token]}")
            if logits[token] > max_prob:
                max_prob = logits[token]
                max_prob_idx = token
        return max_prob_idx

    def list_compare(self, list1: List[int], list2: List[int]) -> bool:
        if len(list1) != len(list2):
            return False
        else:
            for l1, l2 in zip(list1, list2):
                if l1 != l2:
                    return False
        return True


if __name__ == "__main__":
    start = time.time()
    func_tokenizer = DefinedFunctionTokenizer()
    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model()
    end = time.time()
    print(f"LLM loaded in {(end - start):.3f}s")
    print(f"{llm._encode('_')}")
    tokens = func_tokenizer.tokenize_string("fn_add_sum", llm)
    # print(tokens)
    path = "data/input/functions_definition.json"
    allowed_words = func_tokenizer.load_json(path)
    allowed_tokens = func_tokenizer.tokenize_json_by_word(llm)
    print(allowed_tokens)

    prompt_tokens = initial_prompt_toke(llm)
    # allowed_token = {'fn_name': [[8822, 93054, 32964], [8822, 1889, 3744], [8822, 3062, 39794, 12993], [8822, 2891, 32964], [8822, 43277, 3904], [8822, 6892, 68347], [8822, 5228, 7660, 3904, 6615, 41832]], 'args_names': [[64], [82], [83631], [2427, 3904], [26387], [606], [77], [65]], 'args_types': [[495], [3649], [396]], 'return_type': [[2641], [495], [3649]], 'special_char': [[58], [25], [92], [6], [90], [11], [220], [62], [60]]}

    toke_masking = TokenMasking(allowed_tokens, prompt_tokens, llm)
    final_fn = toke_masking.generate_function_name()
    print(final_fn)
    print(llm._decode(final_fn))
