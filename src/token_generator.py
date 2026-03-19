from typing import List, Set, Callable, Tuple, Dict, Any
import numpy as np
import re
from pydantic import BaseModel, Json
from dataclasses import dataclass
from llm_sdk import Small_LLM_Model
from .helper_functions import is_valid_num
from .parser import FnInfo


@dataclass
class ConstantPrams:
    regex_pattern = {
        '\\', '|', '[', ']', '+', '*', '?', '^', '$',
        '(', ')', '.', ' ', '-', '\\b', '\\d', '\\s',
        'a-z', 'A-Z', '"',
    }
    symbol_pattern = {
        '*', '#', ' ', '_', '$', '@'
        }
    regex_bias = 5
    symbol_bias = 5
    prompt_bias = 5
    terminator_bias = 5
    str_terminator = '"'
    num_terminator = ','
    negative_sign_bias = 5
    token_limit = 20


@dataclass
class Cost:
    token_used = 0
    time_taken = 0.0
    avg_time = 0.0


class Output(BaseModel):
    prompt: str = ""
    name: str = ""
    parameters: Dict[str, Any] = {}

    def get_json_str(self) -> Json:
        return self.model_dump_json(indent=2)


class TokenGenerator:
    def __init__(self, llm: Small_LLM_Model,
                 token_set: List[str],
                 encode: Callable,
                 decode: Callable,
                 token_limit: int = ConstantPrams.token_limit) -> None:
        self.prompt_tokens: List[int] = []
        self.llm = llm
        self.encode = encode
        self.decode = decode
        self.tkn_limits = token_limit
        self.token_set = token_set
        self.tokens_spend = 0

    def get_prompt(self) -> List[int]:
        return self.prompt_tokens

    def set_token_limit(self, limit: int) -> None:
        self.tkn_limits = limit

    def slice_prompt_tokens(self, start: int, end: int) -> None:
        self.prompt_tokens = self.prompt_tokens[start: end]

    def add_to_prompt(self, tokens: List[int]) -> int:
        self.prompt_tokens.extend(tokens)
        return len(self.prompt_tokens)

    def re_initialize_prompt_token(self) -> None:
        self.prompt_tokens = []
        self.tokens_spend = 0

    def get_total_token_spend(self) -> int:
        return self.tokens_spend

    def get_str_to_matching_tokens(self, str_srt: Set) -> List[int]:
        tokens = [idx for idx, val in enumerate(self.token_set)
                  for pattern in str_srt if val.startswith(pattern)]
        return tokens

    def generate_function_name(self, allowed_token: List[FnInfo]) -> List:
        complete_fn_tokens: List[int] = []
        token = float("-inf")
        terminating_token = self.encode(ConstantPrams.str_terminator)
        # print(f"prompt: {self.tokenizer.decode(self.prompt_tokens)}")

        # Create the list of all allowed function names
        all_tokens = terminating_token
        for tkn in allowed_token:
            all_tokens.extend(tkn.fn_name_token)
        next_allowed_tokens = set(all_tokens)

        while token != terminating_token[0] and \
                len(complete_fn_tokens) < self.tkn_limits:
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)

            token = self.get_next_fn_token(logits, next_allowed_tokens)
            complete_fn_tokens.append(token)
            self.prompt_tokens.append(token)
            self.tokens_spend += 1
        return complete_fn_tokens

    def get_matching_word(
            self, sub_str: str,
            prompt: str, arg_type: str
            ) -> str | None:
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
        allowed_tokens.extend(terminating_token)

        if arg_name == 'regex':
            self.regex = self.get_str_to_matching_tokens(
                ConstantPrams.regex_pattern)
        if arg_name == 'replacement':
            self.symbols = self.get_str_to_matching_tokens(
                ConstantPrams.symbol_pattern)

        token_counter = 0
        sub_str = ""
        while token != terminating_token[0] and \
                len(complete_arg_tokens) < self.tkn_limits:
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            self.tokens_spend += 1
            token_counter += 1
            if arg_type == "float" or arg_type == "int" or \
                    arg_type == "number":
                token = self.get_next_numeric_token(
                    logits, set(allowed_tokens))
                str_val = self.decode(token)

                # Force the model not to predict same number twice
                # if token in allowed_tokens:
                #     allowed_tokens.pop(allowed_tokens.index(token))

                if " " in str_val:
                    str_val = str_val.strip()
                    token = self.encode(str_val)[0]

                sub_str += str_val
                matching_word = self.get_matching_word(
                    sub_str, prompt, arg_type)
                if matching_word is not None:
                    # print(
                    #     f"Matching word: {matching_word},"
                    #     f" sub_str: {sub_str}")
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
                    # print(f"Matching word: {matching_word}, "
                    #       f"sub_str: {sub_str}")
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
        # show_top_logits(self.decode, logits_np, max_prob_token, 5)
        return max_prob_token

    def get_next_str_token(self, logits: List[float],
                           allowed_idx: Set[int], arg_name: str,
                           soft_bias: int = 5) -> int:
        logits_np = np.array(logits)
        mask_idx = list(allowed_idx)
        logits_np[mask_idx] += soft_bias
        if arg_name == 'regex':
            mask_idx = self.regex
            logits_np[mask_idx] += ConstantPrams.regex_bias

        if arg_name == 'replacement':
            mask_idx = self.symbols
            logits_np[mask_idx] += ConstantPrams.symbol_bias

        max_prob_token = int(np.argmax(logits_np))
        # show_top_logits(self.decode, logits_np, max_prob_token, 5)
        return max_prob_token

    def get_next_numeric_token(self, logits: List[float],
                               allowed_idx: Set[int]) -> int:
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        mask_idx = list(allowed_idx)
        mask[mask_idx] = 0
        self.create_token_biasing(
            mask, mask_idx, "-", ConstantPrams.negative_sign_bias)
        max_prob_token = int(np.argmax(logits_np + mask))
        # show_top_logits(self.decode, logits_np, max_prob_token, 5)
        return max_prob_token

    def create_token_biasing(
            self, mask: np.array,
            allowed_idx: List[int],
            bias_string: str,
            soft_bias: int = 5) -> None:
        new_idx: List[int] = []
        for token_idx in allowed_idx:
            token_str = self.decode(token_idx).strip()
            if token_str in bias_string and len(token_str) > 0:
                new_idx.append(token_idx)
        mask[new_idx] += soft_bias
