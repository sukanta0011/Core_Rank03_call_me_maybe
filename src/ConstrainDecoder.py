from typing import List, Set
import numpy as np
from src.helper_functions import is_valid_num
from src.tokenizer import Tokenizer


class ConstrainDecoder:
    def __init__(self, llm, tokenizer) -> None:
        self.prompt_tokens = []
        self.llm = llm
        self.tokenizer = tokenizer

    def get_current_prompt(self) -> List[int]:
        return self.prompt_tokens

    def add_to_prompt(self, tokens: List[int]) -> None:
        # for token in tokens:
        # print(f"tokens: {self.tokenizer.decode(tokens)}")
        self.prompt_tokens.extend(tokens)

    def re_initialize_prompt_token(self):
        self.prompt_tokens = []

    def generate_function_name(self, allowed_token: List[List[int]]) -> List:
        complete_fn_tokens = []
        token = float("-inf")
        terminating_token = self.tokenizer.encode('"')
        # print(f"prompt: {self.tokenizer.decode(self.prompt_tokens)}")

        while token != terminating_token[0] and len(complete_fn_tokens) < 20:
            # print(llm._decode(complete_fn_tokens))
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            next_allowed_tokens = set()
            complete_fn_len = len(complete_fn_tokens)
            for fn in allowed_token:
                if len(fn) > complete_fn_len:
                    next_allowed_tokens.add(fn[complete_fn_len])
                else:
                    next_allowed_tokens.add(terminating_token[0])
                # If two function shares common name upto some extent,
                # this will create problem, need to handle it
                # elif self.list_compare(fn, complete_fn_tokens):
                #     return complete_fn_tokens

            # print(f"allowed token: {self.tokenizer.decode(list(next_allowed_tokens))}")
            # print(f"selected token: {self.tokenizer.decode(complete_fn_tokens)}")
            # print(f"Allowed token: {next_allowed_tokens}")
            token = self.get_next_fn_token(logits, next_allowed_tokens)
            complete_fn_tokens.append(token)
            self.prompt_tokens.append(token)
            # print(f"toke: {token}, terminating_token: {terminating_token}")
            # if token == terminating_token[0]:
            #     print("Terminating token used, generation complete.")
                # return complete_fn_tokens
        return complete_fn_tokens

    def generate_args_val(self, allowed_tokens: List[int],
                          arg_type: str) -> List:
        complete_arg_tokens = []
        token = float("-inf")
        if arg_type == "float" or arg_type == "int":
            terminating_token = self.tokenizer.encode(',')
        else:
            terminating_token = self.tokenizer.encode('"')
        # print(f"allowed tokens: {self.tokenizer.decode(allowed_tokens)}")
        allowed_tokens.extend(terminating_token)

        while token != terminating_token[0] and len(complete_arg_tokens) < 20:
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            if arg_type == "float" or arg_type == "int":
                token = self.get_next_numeric_token(logits, set(allowed_tokens))
                allowed_tokens.pop(allowed_tokens.index(token))
                str_val = self.tokenizer.decode(token)
                str_val = str_val.strip()
                # token = self.tokenizer.encode(str_val)
                if is_valid_num(str_val) or str_val == "-":
                    complete_arg_tokens.append(token)
                    self.prompt_tokens.append(token)
            else:
                token = self.get_next_str_token(logits, set(allowed_tokens), 10)
                # allowed_tokens.pop(allowed_tokens.index(token))
                str_val = self.tokenizer.decode(token)
                # print(f"selected toke: {str_val}")
                complete_arg_tokens.append(token)
                self.prompt_tokens.append(token)
                if '"' in str_val:
                    break

        return complete_arg_tokens

    # def get_next_token(self, logits: List[float],
    #                    allowed_idx: Set[int]) -> int:
    #     max_prob = float("-inf")
    #     max_prob_idx = -1
    #     # print()
    #     for token in allowed_idx:
    #         # print(f"{token}, {self.tokenizer.decode([token])}, {logits[token]}")
    #         if logits[token] > max_prob:
    #             max_prob = logits[token]
    #             max_prob_idx = token
    #     # print(f"Selected token {self.tokenizer.decode([max_prob_idx])},"
    #     #       f" {logits[max_prob_idx]}")
    #     return max_prob_idx

    def get_next_fn_token(self, logits: List[float],
                          allowed_idx: Set[int]) -> int:
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        allowed_idx = list(allowed_idx)
        mask[allowed_idx] = 0
        max_prob_token = int(np.argmax(logits_np + mask))

        # # print()
        tokens_with_prob = ""
        for token in allowed_idx:
            # print(f"{token}, {self.tokenizer.decode([token])}, {logits[token]}")
            tokens_with_prob += f"{self.tokenizer.decode([token])}({round(logits[token], 2)}),"
        tokens_with_prob += f"\033[92mSelected token: {self.tokenizer.decode([max_prob_token])}\033[0m"
        print(tokens_with_prob)
        return max_prob_token

    def get_next_str_token(self, logits: List[float],
                           allowed_idx: Set[int], soft_bias: int = 5) -> int:
        logits_np = np.array(logits)
        allowed_idx = list(allowed_idx)
        logits_np[allowed_idx] += soft_bias
        max_prob_token = int(np.argmax(logits_np))
        return max_prob_token

    def get_next_numeric_token(self, logits: List[float],
                               allowed_idx: Set[int],
                               soft_bias: int = 5) -> int:
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        allowed_idx = list(allowed_idx)
        mask[allowed_idx] = 0
        self.create_toke_biasing(mask, allowed_idx, "-.", soft_bias)
        max_prob_token = int(np.argmax(logits_np + mask))
        return max_prob_token

    def create_toke_biasing(self, mask: np.array, allowed_idx: List[int],
                            bias_tokens: str,
                            soft_bias: int = 5) -> None:
        new_idx: List[int] = []
        # print(allowed_idx)
        for token_idx in allowed_idx:
            token_str = self.tokenizer.decode(token_idx)
            # print(f"{token_str}: {token_str in bias_tokens}")
            if token_str.strip() in bias_tokens:
                # print(f"Matching toke: {token_str}")
                new_idx.append(token_idx)
        mask[new_idx] = soft_bias
