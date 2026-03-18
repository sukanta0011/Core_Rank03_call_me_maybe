from typing import List, Set, Callable, Tuple
import numpy as np
import re
from src.helper_functions import is_valid_num
from src.parser import FnInfo


class TokenGenerator:
    def __init__(self, llm, token_set, encode: Callable,
                 decode: Callable, token_limit: int = 20) -> None:
        self.prompt_tokens: List[int] = []
        self.llm = llm
        self.encode = encode
        self.decode = decode
        self.tkn_limits = token_limit
        self.token_set = token_set

    def get_prompt(self) -> List[int]:
        return self.prompt_tokens

    def set_token_limit(self, limit: int) -> None:
        self.tkn_limits = limit

    def slice_prompt_tokens(self, start: int, end: int) -> None:
        self.prompt_tokens = self.prompt_tokens[start: end]

    def add_to_prompt(self, tokens: List[int]) -> int:
        self.prompt_tokens.extend(tokens)
        return len(self.prompt_tokens)

    def re_initialize_prompt_token(self):
        self.prompt_tokens = []

    def generate_function_name(self, allowed_token: List[FnInfo]) -> List:
        complete_fn_tokens: List[int] = []
        token = float("-inf")
        terminating_token = self.encode('"')
        # print(f"prompt: {self.tokenizer.decode(self.prompt_tokens)}")

        while token != terminating_token[0] and \
                len(complete_fn_tokens) < self.tkn_limits:
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)

            # all_tokens = terminating_token
            # for tkn in allowed_token:
            #     all_tokens.extend(tkn.fn_name_token)
            # next_allowed_tokens = set(all_tokens)
            next_allowed_tokens = set()
            complete_fn_len = len(complete_fn_tokens)
            for tkn in allowed_token:
                if len(tkn.fn_name_token) > complete_fn_len:
                    next_allowed_tokens.add(tkn.fn_name_token[complete_fn_len])
                else:
                    next_allowed_tokens.add(terminating_token[0])

            token = self.get_next_fn_token(logits, next_allowed_tokens)
            complete_fn_tokens.append(token)
            self.prompt_tokens.append(token)
        return complete_fn_tokens

    def get_matching_word(self, sub_str: str,
                          prompt: str, arg_type: str) -> str | None:
        if len(sub_str.strip()) == 0:
            return None
        num_type = {"float", "int", "num", "number"}
        if arg_type in num_type:
            pattern = re.escape(sub_str) + "[0-9.]*"
        else:
            pattern = r"\b" + re.escape(sub_str) + r"\w*"
        # print(f"sub_str: {sub_str}")
        match = re.findall(pattern, prompt)
        unique_matches = list(set(match))
        if len(unique_matches) == 1:
            # if re.search("^\\s+", sub_str):
            #     return f" {unique_matches[0]}"
            return f"{unique_matches[0]}"
        return None

    def generate_args_val(self, allowed_tokens: List[int], arg_name: str,
                          arg_type: str, prompt: str,
                          soft_bias: int) -> Tuple[List, List]:
        complete_arg_tokens: List[int] = []
        token = float("-inf")
        if arg_type == "float" or arg_type == "int" or arg_type == "number":
            terminating_token = self.encode(',')
        else:
            terminating_token = self.encode('"')
        if arg_type == 'bool':
            allowed_tokens.extend(self.encode('True'))
            allowed_tokens.extend(self.encode('False'))
        # print(f"allowed tokens: {self.tokenizer.decode(allowed_tokens)}")
        allowed_tokens.extend(terminating_token)
        token_counter = 0
        sub_str = ""
        while token != terminating_token[0] and \
                len(complete_arg_tokens) < self.tkn_limits:
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            token_counter += 1
            if arg_type == "float" or arg_type == "int" or \
                    arg_type == "number":
                token = self.get_next_numeric_token(
                    logits, set(allowed_tokens))
                str_val = self.decode(token)
                if token in allowed_tokens:
                    allowed_tokens.pop(allowed_tokens.index(token))
                if " " in str_val:
                    str_val = str_val.strip()
                    token = self.encode(str_val)[0]

                sub_str += str_val
                matching_word = self.get_matching_word(
                    sub_str, prompt, arg_type)
                if matching_word is not None:
                    # print(
                    #     f"Matching word: {matching_word}, sub_str: {sub_str}")
                    for _ in range(token_counter - 1):
                        self.prompt_tokens.pop()
                        complete_arg_tokens.pop()
                    if is_valid_num(matching_word):
                        # print(f"matching word: {matching_word}")
                        self.prompt_tokens.extend(self.encode(matching_word))
                        complete_arg_tokens.extend(self.encode(matching_word))
                        sub_str = ""
                        token_counter = 0
                elif str_val.isdigit() or str_val in {"-", "."}:
                    # print(str_val)
                    complete_arg_tokens.append(token)
                    self.prompt_tokens.append(token)
            else:
                token = self.get_next_str_token(
                    logits, set(allowed_tokens), arg_name, soft_bias)
                str_val = self.decode(token)
                sub_str += str_val
                matching_word = self.get_matching_word(
                    sub_str, prompt, arg_type)
                if matching_word is not None:
                    print(f"Matching word: {matching_word}, "
                          f"sub_str: {sub_str}")
                    for _ in range(token_counter - 1):
                        self.prompt_tokens.pop()
                        complete_arg_tokens.pop()
                    self.prompt_tokens.extend(self.encode(matching_word))
                    complete_arg_tokens.extend(self.encode(matching_word))
                    sub_str = ""
                    token_counter = 0
                else:
                    complete_arg_tokens.append(token)
                    self.prompt_tokens.append(token)
                if '"' in str_val:
                    idx = str_val.index('"')
                    self.prompt_tokens.pop()
                    self.prompt_tokens.extend(self.encode(str_val[:idx + 1]))
                    break

        return (complete_arg_tokens, allowed_tokens)

    def get_next_fn_token(self, logits: List[float],
                          allowed_idx: Set[int]) -> int:
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        mask_idx = list(allowed_idx)
        mask[mask_idx] = 0
        max_prob_token = int(np.argmax(logits_np + mask))
        self.show_top_logits(logits_np, max_prob_token, 5)
        return max_prob_token

    def get_next_str_token(self, logits: List[float],
                           allowed_idx: Set[int], arg_name: str,
                           soft_bias: int = 5) -> int:
        logits_np = np.array(logits)
        mask_idx = list(allowed_idx)
        logits_np[mask_idx] += soft_bias
        if arg_name == 'regex':
            regex_pattern = {'\\', '|', '[', ']', '+', '*', '?', '^', '$', '(', ')', '.', ' ', '-', '\\b', '\\d', '\\s', 'a-z', 'A-Z', '"'}
            # allowed_regex = [idx for idx, val in enumerate(self.token_set) for pattern in regex_pattern if pattern in val]
            allowed_regex = [idx for idx, val in enumerate(self.token_set) for pattern in regex_pattern if val.startswith(pattern)]
            mask_idx = list(allowed_regex)
            logits_np[mask_idx] += 5

        if arg_name == 'replacement':
            symbol_pattern = {'*', '#', ' ', '_', '$', '@'}
            allowed_regex = [idx for idx, val in enumerate(self.token_set) for pattern in symbol_pattern if pattern in val]
            mask_idx = list(allowed_regex)
            logits_np[mask_idx] += 5

        max_prob_token = int(np.argmax(logits_np))
        # self.show_top_logits(logits_np, max_prob_token, 5)
        return max_prob_token

    def get_next_numeric_token(self, logits: List[float],
                               allowed_idx: Set[int],
                               soft_bias: int = 10) -> int:
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        mask_idx = list(allowed_idx)
        mask[mask_idx] = 0
        self.create_token_biasing(mask, mask_idx, "-", soft_bias)
        max_prob_token = int(np.argmax(logits_np + mask))
        # self.show_top_logits(logits_np, max_prob_token, 5)
        return max_prob_token

    def show_top_logits(self, logits_np: List[int],
                        max_logits_idx: int, top: int) -> None:
        sorted_idx = np.argsort(logits_np)
        top_ten = sorted_idx[-top:]
        tokens_with_prob = ""
        for token in top_ten:
            # print(f"{token}, {self.tokenizer.decode([token])}, {logits[token]}")
            tokens_with_prob += f"{
                self.decode([token])}({round(logits_np[token], 2)}),"
        tokens_with_prob += (
            f"\033[92mSelected token: {self.decode([max_logits_idx])}\033[0m")
        print(tokens_with_prob)

    def create_token_biasing(self, mask, allowed_idx: List[int],
                             bias_string: str,
                             soft_bias: int = 5) -> None:
        new_idx: List[int] = []
        # print(allowed_idx)
        for token_idx in allowed_idx:
            token_str = self.decode(token_idx).strip()
            # print(f"{token_str}: {token_str in bias_tokens}")
            if token_str in bias_string and len(token_str) > 0:
                # print(f"Matching toke: {token_str}")
                new_idx.append(token_idx)
        mask[new_idx] += soft_bias
