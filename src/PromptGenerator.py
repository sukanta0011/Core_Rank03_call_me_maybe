import time
from typing import List
from src.ConstrainDecoder import ConstrainDecoder
from src.tokenizer import Tokenizer
from src.helper_functions import tokenize_string, initial_prompt_toke


class PromptGenerator:
    def __init__(self, llm, fn_tokens: List[List[int]],
                 args_list: List[List[str]], fn_names: List[str],
                 args_types: List[List[str]], tokenizer: Tokenizer) -> None:
        self.llm = llm
        self.fn_tokens = fn_tokens
        self.fn_names = fn_names
        self.args_list = args_list
        self.args_types = args_types
        self.constrain_decoder = ConstrainDecoder(llm, tokenizer)
        self.tokenizer = tokenizer

    def add_str_to_prompt(self, string: str) -> None:
        str_patch = string
        token_patch = self.tokenizer.encode(str_patch)
        self.constrain_decoder.add_to_prompt(token_patch)

    # def tokenize_prompt(self, prompt: str) -> List[int]:
    #     splitted_prompt = prompt.split(" ")
    #     prompt_tokens = self.tokenizer.encode(prompt)
    #     for word in splitted_prompt:
    #         prompt_tokens.append(self.llm._encode(word).tolist()[0][0])
    #     # print(f"Prompt: {prompt_tokens}")
    #     return prompt_tokens

    def generate(self, prompt: str):
        # prompt_tokens = self.tokenize_prompt(prompt)

        # prompt_2D = self.tokenize_prompt(prompt)
        # print(f"Prompt token 2D: {prompt_2D}")

        starting_prompt = '\t{\n\t\t"Prompt": ' + '"' + prompt + '"'
        # print(f"prompt: {starting_prompt}")
        self.add_str_to_prompt(starting_prompt)
        # final_prompt_token = self.constrain_decoder.generate_function_name([prompt_tokens])
        # print(llm._decode(final_prompt_token))

        self.add_str_to_prompt(',\n\t\t"fn_name": "')
        # print(f"prompt: {self.tokenizer.decode(self.constrain_decoder.prompt_tokens)}")
        final_fn = self.constrain_decoder.\
            generate_function_name(self.fn_tokens)
        # print(self.tokenizer.decode(final_fn[: -1]))

        final_fn_name = self.tokenizer.decode(final_fn[: -1])
        if final_fn_name in self.fn_names:
            self.handle_arguments(prompt, final_fn_name)
        self.add_str_to_prompt("},\n\t},")

        # final_token = self.constrain_decoder.get_current_prompt()
        # print(llm._decode(final_token))

    def modify_prompt_for_regex(self, prompt: str) -> str:
        patterns = {
            "vowels": "[aeiouAEIOU]",
            "asterisks": "*",
            "digits": "r'\d+'",
        }
        for key, val in patterns.items():
            prompt = prompt.replace(key, val)
        # prompt += "\nExample: replace 'r\d+' in Hello123 by Number, source_string: Hello123, regex: r\'d+', replacement: Number"
        return prompt

    def handle_arguments(self, prompt: str, final_fn_name):
        if "regex" in final_fn_name:
            prompt = self.modify_prompt_for_regex(prompt)
        # if "numbers" in final_fn_name:
        #     prompt += "\nExample: add -2 and 3, a: -2, b: 3"
        prompt_tokens = self.tokenizer.encode(prompt)
        fn_idx = self.fn_names.index(final_fn_name)
        # print(f"args: {self.args_list[fn_idx]}")
        initial_arg_token = ',\n\t\t"args": {'
        self.add_str_to_prompt(initial_arg_token)
        total_args = len(self.args_list[fn_idx])

        for i, arg in enumerate(self.args_list[fn_idx]):
            arg_type = self.args_types[fn_idx][i]
            if arg_type == "float" or arg_type == "int":
                self.add_str_to_prompt(f'"{arg}": ')
            else:
                self.add_str_to_prompt(f'"{arg}": "')
            arg_val_token = self.constrain_decoder.\
                generate_args_val(prompt_tokens, arg_type)
            arg_val_str = self.tokenizer.decode(arg_val_token)
            if arg_type == 'float' and '.' not in arg_val_str and len(arg_val_str) > 0:
                self.add_str_to_prompt('.0')
            # print(f"arg_val: {arg_val_str}")
            # initial_arg_token += f"{arg_val_str}, "arg_val_str
            if i < total_args - 1:
                self.add_str_to_prompt(", ")

    def generate_for_all_prompts(self, prompts: List[str]):
        for prompt in prompts:
            start = time.time()
            self.constrain_decoder.re_initialize_prompt_token()
            initial_token = initial_prompt_toke(
                prompt, self.fn_names, self.args_list, self.tokenizer)
            # print(f"prompt: {self.tokenizer.decode(initial_token)}")
            self.constrain_decoder.add_to_prompt(initial_token)
            self.generate(prompt)
            final_token = self.constrain_decoder.get_current_prompt()
            print(self.tokenizer.decode(final_token[len(initial_token):]))
            end = time.time()
            print(f"Token generation time: {(end - start):.3f}s")
