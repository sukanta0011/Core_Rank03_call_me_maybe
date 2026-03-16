import time
from typing import List, Callable, Dict, Any
import re
from pydantic import BaseModel, Json, Field
from llm_sdk import Small_LLM_Model
from src.token_generator import TokenGenerator
from src.helper_functions import initial_prompt_toke
from src.parser import FnInfo, Prompts


class Output(BaseModel):
    prompt: str = ""
    fn_name: str = ""
    fn_args: Dict[str, Any] = {}

    def get_json_str(self) -> Json:
        return self.model_dump_json(indent=2)


class ConstrainDecoder:
    def __init__(self, llm: Small_LLM_Model,
                 functions: List[FnInfo],
                 encode: Callable, decode: Callable) -> None:
        self.llm = llm
        self.functions = functions
        self.tkn_generator = TokenGenerator(llm, encode, decode)
        self.output: List[Output] = []
        self.encode = encode
        self.decode = decode

    def add_str_to_prompt(self, string: str) -> int:
        str_patch = string
        token_patch = self.encode(str_patch)
        return self.tkn_generator.add_to_prompt(token_patch)

    def modify_prompt_for_regex(self, prompt: str) -> str:
        patterns = {
            "vowels": "[aeiouAEIOU]",
            "consonants": "[^aeiouAEIOU]",
            "asterisks": "*",
            "digits": "r'\\d+'"
        }

        # for key, val in patterns.items():
        #     prompt = prompt.replace(key, val)
        # print(f"Prompt: {prompt}")
        prompt += f"\nExample of regex:\n {patterns}\n"
        # prompt += "Important: When extracting a regex for digits, use \d+. For vowels, use [aeiou]."
        return prompt

    def generate(self, prompt: str, out: Output) -> None:
        starting_prompt = '{\n"Prompt": ' + '"' + prompt + '"'
        self.add_str_to_prompt(starting_prompt)

        self.add_str_to_prompt(',\n"fn_name": "')
        # print(f"prompt: {self.tokenizer.decode(self.constrain_decoder.prompt_tokens)}")
        final_fn = self.tkn_generator.\
            generate_function_name(self.functions)

        final_fn_name = self.decode(final_fn[: -1])
        # self.tkn_generator.slice_prompt_tokens(0, before_fn_pos)
        for fn in self.functions:
            if final_fn_name == fn.fn_name:
                out.fn_name = final_fn_name
                self.handle_arguments(prompt, fn, out)
        self.add_str_to_prompt("},\n},")

    def get_all_allowed_token(self, prompt: str) -> List[int]:
        tokens = self.encode(prompt)
        words = prompt.split(" ")
        for word in words:
            tokens.extend(self.encode(word))
        return tokens

    def handle_arguments(self, prompt: str,
                         fn: FnInfo, out: Output) -> None:
        original_prompt = prompt
        if "regex" in fn.fn_name:
            prompt = self.modify_prompt_for_regex(prompt)
            # print(prompt)
        # if "numbers" in final_fn_name:
        #     prompt += "\nExample: add -2 and 3, a: -2, b: 3"

        prompt_tokens = self.get_all_allowed_token(prompt)
        initial_arg_token = ',\n"args": {'
        self.add_str_to_prompt(initial_arg_token)
        total_args = len(fn.args_names)

        for i, arg in enumerate(fn.args_names):
            arg_type = fn.args_types[arg]
            if arg_type == "float" or arg_type == "int":
                self.add_str_to_prompt(f'"{arg}": ')
            else:
                # self.add_str_to_prompt(f"If the arg in the prompt:'{prompt}' do not have clear bounder, try to extract it properly\nss")
                self.add_str_to_prompt(f'"{arg}": "')
            arg_val_token, prompt_tokens = self.tkn_generator.\
                generate_args_val(prompt_tokens, arg_type,original_prompt)
            arg_val_str = self.decode(arg_val_token)

            # Store th arg value
            if arg_type == 'int':
                try:
                    out.fn_args[arg] = int(arg_val_str)
                except ValueError:
                    out.fn_args[arg] = ""
                    print(f"{arg} has not numeric value: {arg_val_str}")
            elif arg_type == 'float':
                try:
                    out.fn_args[arg] = float(arg_val_str)
                except ValueError:
                    out.fn_args[arg] = ""
                    print(f"{arg} has not numeric value: {arg_val_str}")
            else:
                str_in_args = re.findall("[^\",}]", arg_val_str)
                out.fn_args[arg] = "".join(str_in_args).strip()

            if i < total_args - 1:
                self.add_str_to_prompt(", ")

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
            print("\033[92mToken generation time: "
                  f"{(end - start):.3f}s\033[0m")
        return self.output
