from typing import List, Set, Callable, Tuple
import numpy as np
import re
from src.helper_functions import is_valid_num
from src.parser import FnInfo


class TokenGenerator:
    def __init__(self, llm, encode: Callable,
                 decode: Callable) -> None:
        self.prompt_tokens: List[int] = []
        self.llm = llm
        self.encode = encode
        self.decode = decode

    def get_prompt(self) -> List[int]:
        return self.prompt_tokens

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

        while token != terminating_token[0] and len(complete_fn_tokens) < 20:
            # print(llm._decode(complete_fn_tokens))
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            next_allowed_tokens = set()
            complete_fn_len = len(complete_fn_tokens)
            for tkn in allowed_token:
                if len(tkn.fn_name_token) > complete_fn_len:
                    next_allowed_tokens.add(tkn.fn_name_token[complete_fn_len])
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
        return complete_fn_tokens

    def get_matching_word(self, sub_str: str, prompt: str) -> str | None:
        pattern = r"\b" + re.escape(sub_str) + r"\w*"
        match = re.findall(pattern, prompt)
        unique_matches = list(set(match))
        if len(unique_matches) == 1:
            # if re.search("^\\s+", sub_str):
            #     return f" {unique_matches[0]}"
            return f"{unique_matches[0]}"
        return None

    def generate_args_val(self, allowed_tokens: List[int],
                          arg_type: str, prompt: str, soft_bias: int) -> Tuple[List, List]:
        complete_arg_tokens: List[int] = []
        token = float("-inf")
        if arg_type == "float" or arg_type == "int":
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
        while token != terminating_token[0] and len(complete_arg_tokens) < 20:
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            token_counter += 1
            if arg_type == "float" or arg_type == "int":
                token = self.get_next_numeric_token(logits, set(allowed_tokens))
                allowed_tokens.pop(allowed_tokens.index(token))
                str_val = self.decode(token)
                str_val = str_val.strip()
                # token = self.tokenizer.encode(str_val)
                if is_valid_num(str_val) or str_val == "-":
                    complete_arg_tokens.append(token)
                    self.prompt_tokens.append(token)
            else:
                token = self.get_next_str_token(logits, set(allowed_tokens), soft_bias)
                if token in allowed_tokens:
                    allowed_tokens.pop(allowed_tokens.index(token))
                str_val = self.decode(token)
                # self.remove_token(token, allowed_tokens)
                sub_str += str_val
                matching_word = self.get_matching_word(sub_str, prompt)
                if matching_word is not None:
                    # print(f"Matching word: {matching_word}, sub_str: {sub_str}")
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
                    break

        return (complete_arg_tokens, allowed_tokens)
    
    def remove_token(self, token: int, allowed_token: List[int]) -> List[int]:
        allowed_token_str = [self.decode(tkn) for tkn in allowed_token]
        token_str = self.decode(token)[0]
        new_allowed_token = [sub_str for sub_str in allowed_token_str if sub_str.strip() != token_str]
        print(new_allowed_token)

    def get_next_fn_token(self, logits: List[float],
                          allowed_idx: Set[int]) -> int:
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        mask_idx = list(allowed_idx)
        mask[mask_idx] = 0
        max_prob_token = int(np.argmax(logits_np + mask))

        # # print()
        # tokens_with_prob = ""
        # for token in allowed_idx:
        #     # print(f"{token}, {self.tokenizer.decode([token])}, {logits[token]}")
        #     tokens_with_prob += f"{self.tokenizer.decode([token])}({round(logits[token], 2)}),"
        # tokens_with_prob += f"\033[92mSelected token: {self.tokenizer.decode([max_prob_token])}\033[0m"
        # print(tokens_with_prob)
        return max_prob_token

    def get_next_str_token(self, logits: List[float],
                           allowed_idx: Set[int], soft_bias: int = 5) -> int:
        logits_np = np.array(logits)
        mask_idx = list(allowed_idx)
        logits_np[mask_idx] += soft_bias
        max_prob_token = int(np.argmax(logits_np))

        # extract top 10 tokens
        # sorted_idx  = np.argsort(logits_np)
        # top_ten = sorted_idx[-5:]
        # # top_ten = logits_np[sorted_idx][-10:].tolist()
        # tokens_with_prob = ""
        # for token in top_ten:
        #     # print(f"{token}, {self.tokenizer.decode([token])}, {logits[token]}")
        #     tokens_with_prob += f"{self.decode([token])}({round(logits[token], 2)}),"
        # tokens_with_prob += f"\033[92mSelected token: {self.decode([max_prob_token])}\033[0m"
        # print(tokens_with_prob)
        return max_prob_token

    def get_next_numeric_token(self, logits: List[float],
                               allowed_idx: Set[int],
                               soft_bias: int = 5) -> int:
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        mask_idx = list(allowed_idx)
        mask[mask_idx] = 0
        self.create_toke_biasing(mask, mask_idx, "-.", soft_bias)
        max_prob_token = int(np.argmax(logits_np + mask))
        return max_prob_token

    def create_toke_biasing(self, mask, allowed_idx: List[int],
                            bias_string: str,
                            soft_bias: int = 5) -> None:
        new_idx: List[int] = []
        # print(allowed_idx)
        for token_idx in allowed_idx:
            token_str = self.decode(token_idx)
            # print(f"{token_str}: {token_str in bias_tokens}")
            if token_str.strip() in bias_string:
                # print(f"Matching toke: {token_str}")
                new_idx.append(token_idx)
        mask[new_idx] += soft_bias

