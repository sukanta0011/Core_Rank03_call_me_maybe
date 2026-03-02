from typing import List, Set
from src.helper_functions import is_valid_num


class ConstrainDecoder:
    def __init__(self, llm) -> None:
        self.prompt_tokens = []
        self.llm = llm

    def get_current_prompt(self) -> List[int]:
        return self.prompt_tokens

    def add_to_prompt(self, tokens: List[int]) -> None:
        for token in tokens:
            self.prompt_tokens.append(token)

    def re_initialize_prompt_token(self):
        self.prompt_tokens = []

    def generate_function_name(self, allowed_token: List[List[int]]) -> List:
        complete_fn_tokens = []
        token = float("-inf")
        terminating_token = self.llm._encode('"').tolist()[0][0]

        while token != terminating_token:
            # print(llm._decode(complete_fn_tokens))
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            next_allowed_tokens = set()
            next_allowed_tokens.add(terminating_token)
            complete_fn_len = len(complete_fn_tokens)
            for fn in allowed_token:
                if len(fn) > complete_fn_len:
                    next_allowed_tokens.add(fn[complete_fn_len])
                # If two function shares common name upto some extent,
                # this will create problem, need to handle it
                # elif self.list_compare(fn, complete_fn_tokens):
                #     return complete_fn_tokens

            # print(f"allowed token: {llm._decode(list(next_allowed_tokens))}")
            # print(f"Allowed token: {next_allowed_tokens}")
            token = self.get_next_token(logits, next_allowed_tokens)
            complete_fn_tokens.append(token)
            self.prompt_tokens.append(token)
            # if token == terminating_token:
            #     print("Terminating token used, generation complete.")
            #     return complete_fn_tokens
        return complete_fn_tokens

    def generate_args_val(self, allowed_tokens: List[int],
                          arg_type: str) -> List:
        complete_fn_tokens = []
        token = float("-inf")
        if arg_type == "float" or arg_type == "int":
            terminating_token = self.llm._encode(',').tolist()[0][0]
        else:
            terminating_token = self.llm._encode('"').tolist()[0][0]
        # double_quote = self.llm._encode('"').tolist()[0][0]
        # allowed_tokens.append(double_quote)
        allowed_tokens.append(terminating_token)

        while token != terminating_token:
            # print(llm._decode(complete_fn_tokens))
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            # print(f"allowed token: {llm._decode(list(next_allowed_tokens))}")
            # print(f"Allowed args: {allowed_tokens}")
            token = self.get_next_token(logits, set(allowed_tokens))
            allowed_tokens.pop(allowed_tokens.index(token))
            if arg_type == "float" or arg_type == "int":
                str_val = self.llm._decode(token)
                if is_valid_num(str_val):
                    complete_fn_tokens.append(token)
                    self.prompt_tokens.append(token)
            else:
                complete_fn_tokens.append(token)
                self.prompt_tokens.append(token)
        return complete_fn_tokens

    def get_next_token(self, logits: List[float],
                       allowed_idx: Set[int]) -> int:
        max_prob = float("-inf")
        max_prob_idx = -1
        # print()
        for token in allowed_idx:
            # print(f"{token}, {self.llm._decode([token])}, {logits[token]}")
            if logits[token] > max_prob:
                max_prob = logits[token]
                max_prob_idx = token
        # print(f"Selected token {self.llm._decode([max_prob_idx])},"
        #       f" {logits[max_prob_idx]}")
        return max_prob_idx

    def list_compare(self, list1: List[int], list2: List[int]) -> bool:
        # print(f"List1: {self.llm._decode(list1)}")
        # print(f"List2: {self.llm._decode(list2)}")
        if len(list1) != len(list2):
            return False
        else:
            # print("Same length list found")
            for l1, l2 in zip(list1, list2):
                # print(self.llm._decode([l1]))
                # print(self.llm._decode([l2]))
                if l1 != l2:
                    return False
        return True
