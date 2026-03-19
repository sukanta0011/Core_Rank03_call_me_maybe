import time
from typing import List, Callable, Any
import re
from llm_sdk import Small_LLM_Model
from src.token_generator import TokenGenerator, Output
from src.helper_functions import initial_prompt_toke
from src.parser import FnInfo, Prompts


class ConstrainDecoder:
    def __init__(self, llm: Small_LLM_Model,
                 functions: List[FnInfo],
                 token_set: List[str],
                 encode: Callable, decode: Callable) -> None:
        self.llm = llm
        self.functions = functions
        self.tkn_generator = TokenGenerator(llm, token_set, encode, decode)
        self.output: List[Output] = []
        self.encode = encode
        self.decode = decode

    def add_str_to_prompt(self, string: str) -> int:
        str_patch = string
        token_patch = self.encode(str_patch)
        return self.tkn_generator.add_to_prompt(token_patch)

    def generate(self, prompt: str, out: Output) -> None:
        starting_prompt = f'Question: "{prompt}",\n'
        self.add_str_to_prompt(starting_prompt)

        self.add_str_to_prompt('{"name": "fn')
        # print(f"prompt: {self.tokenizer.decode(
        #     self.constrain_decoder.prompt_tokens)}")
        final_fn = self.tkn_generator.\
            generate_function_name(self.functions)

        final_fn_name = "fn" + self.decode(final_fn[: -1])
        # self.tkn_generator.slice_prompt_tokens(0, before_fn_pos)
        for fn in self.functions:
            if final_fn_name == fn.fn_name:
                out.name = final_fn_name
                self.handle_arguments(prompt, fn, out)
        self.add_str_to_prompt("},\n},")

    def get_all_allowed_token(self, prompt: str) -> List[int]:
        tokens = self.encode(prompt)
        words = re.findall(r"\'[^\']+\'|\b\w+\b", prompt)
        # print(words)
        for word in words:
            tokens.extend(self.encode(word))
        unique_tokens = list(set(tokens))
        # print([self.decode(tkn) for tkn in unique_tokens])
        return unique_tokens

    def handle_arguments(self, prompt: str,
                         fn: FnInfo, out: Output) -> None:
        original_prompt = prompt

        prompt_tokens = self.get_all_allowed_token(prompt)
        initial_arg_token = ',\n"parameters": {'
        self.add_str_to_prompt(initial_arg_token)
        total_args = len(fn.args_names)

        for i, arg in enumerate(fn.args_names):
            arg_type = fn.args_types[arg]
            if (arg_type == "float" or arg_type == "int" or
               arg_type == "number"):
                self.add_str_to_prompt(f'"{arg}": ')
            else:
                self.add_str_to_prompt(f'"{arg}": "')
            arg_val_token, prompt_tokens = self.tkn_generator.\
                generate_args_val(
                    prompt_tokens, arg, arg_type, original_prompt, 2)
            arg_val_str = self.decode(arg_val_token)
            self._store_arguments(arg, arg_val_str, arg_type, out)

            if i < total_args - 1:
                self.add_str_to_prompt(", ")

    def _store_arguments(self, key: str, val: Any,
                         arg_type: Any, out: Output) -> None:
        if arg_type == 'number':
            try:
                if "." in val:
                    out.parameters[key] = float(val)
                else:
                    out.parameters[key] = int(val)
            except ValueError:
                out.parameters[key] = ""
                print(f"{key} has not numeric value: {val}")
        elif arg_type == 'bool':
            if 'right' in val.lower() or 'true' in val.lower():
                out.parameters[key] = True
            elif 'wrong' in val.lower() or 'false' in val.lower():
                out.parameters[key] = False
        else:
            str_in_args = re.findall("[^\",}]", val)
            clean_val = "".join(str_in_args)
            if len(clean_val) > 0 and len(clean_val.strip()) == 0:
                clean_val = " "
            else:
                clean_val = clean_val.strip()
            # if key == "replacement":
            #     symbol_prompt = (
            #         "\nWhat is the symbolic representation of"
            #         f" the string '{clean_val}'?\nAnswer: \"")
            #     self.add_str_to_prompt(symbol_prompt)
            #     # print(regex_prompt)
            #     self.tkn_generator.generate_args_val(
            #         [], key, val, symbol_prompt, 0)
            out.parameters[key] = clean_val

    def generate_for_all_prompts(self, prompts: List[Prompts]) -> List[Output]:
        for prompt in prompts:
            start = time.time()

            out = Output()
            out.prompt = prompt.prompt
            self.tkn_generator.re_initialize_prompt_token()
            initial_token = initial_prompt_toke(
                prompt.prompt, self.functions, self.encode)
            # print(f"prompt: {self.tokenizer.decode(initial_token)}")
            self.tkn_generator.add_to_prompt(initial_token)
            self.generate(prompt.prompt, out)
            final_token = self.tkn_generator.get_prompt()
            print(self.decode(final_token[len(initial_token):]))
            self.output.append(out)
            end = time.time()
            time_taken = round((end - start), 3)
            tokens_spend = self.tkn_generator.get_total_token_spend()
            avg_cost = round(time_taken / tokens_spend, 3)
            print(f"\033[92mTokens Used: {tokens_spend},"
                  f" Time: {time_taken}s, Cost/Token: {avg_cost}s\033[0m")
        return self.output
