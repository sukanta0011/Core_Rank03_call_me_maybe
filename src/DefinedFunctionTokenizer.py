import json
import time
# from pydantic import BaseModel
from typing import List, Dict, Set
from src.helper_functions import load_json


def initial_prompt_toke(prompt: str, llm) -> List[int]:
    json_txt = str(load_json())
    pre_prompt = "You need to act as function generator\n After reading the " \
                 "user question, your job will be to provide the function name\n"\
                 f"Available function names are: {json_txt}\n"

    question = f"Question: {prompt}\n"
    combined_prompt = f"{pre_prompt} + {question}"
    tokens = llm._encode(combined_prompt).tolist()
    return tokens[0]


def tokenize_string(string: str, llm) -> List[int]:
    return llm._encode(string).tolist()[0]


def is_valid_num(val: str) -> bool:
    if val == ".":
        return True
    try:
        int(val)
        return True
    except ValueError:
        return False


def split_word(string: str) -> List[str]:
    str_len = len(string)
    tokens = []
    i = 0
    try:
        while (i < str_len):
            sub_str = ""
            if string[i] == "'":
                sub_str += string[i]
                i += 1
                while string[i] != "'" and i < str_len:
                    sub_str += string[i]
                    i += 1
                if i < str_len:
                    sub_str += string[i]
                    i += 1
            elif string[i] == '"':
                sub_str += string[i]
                i += 1
                while string[i] != '"' and i < str_len:
                    sub_str += string[i]
                    i += 1
                if i < str_len:
                    sub_str += string[i]
                    i += 1
            else:
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

    def load_json(self, file_path: str) -> Dict:
        with open(file_path, "r") as fl:
            data = json.load(fl)

        for d in data:
            for key, val in d.items():
                if key not in self.data_str.keys():
                    self.data_str[key] = []
                if isinstance(val, list):
                    self.data_str[key].append(val)
                    # for lst_data in val:
                    #     self.data_str[key].append(lst_data)
                elif isinstance(val, dict):
                    # for _, v in val.items():
                    self.data_str[key].append(list(val.values()))
                else:
                    self.data_str[key].append(val)

        # self.data_str["special_char"] = set()
        # for char in str(data):
        #     if not char.isalpha() and not char.isalnum():
        #         self.data_str["special_char"].add(char)
        return self.data_str

    def tokenize_json_using_llm(self, llm) -> Dict:
        for k, v in self.data_str.items():
            if k not in self.data_token.keys():
                self.data_token[k] = []
            for word in v:
                if isinstance(word, str):
                    tokens = tokenize_string(word, llm)
                    self.data_token[k].append(tokens)
                elif isinstance(word, list):
                    # for token in tokens[0]:
                    tokens = []
                    for sub_word in word:
                        token = tokenize_string(sub_word, llm)
                        tokens.append(token[0])
                self.data_token[k].append(tokens)
        # print(self.data_token)
        return self.data_token

    def tokenize_json_manually(self, llm) -> Dict:
        for k, v in self.data_str.items():
            if k not in self.data_token_by_word.keys():
                self.data_token_by_word[k] = []
            for word in v:
                tokens = []
                sub_words = split_word(word)
                for sub_word in sub_words:
                    # print(f"Sub word: {sub_word}")
                    try:
                        token = tokenize_string(word, llm)
                        # print(f"Tokens: {token[0]}")
                        for sub_token in token:
                            tokens.append(sub_token)
                    except Exception as e:
                        print(f"{e}")
                self.data_token_by_word[k].append(tokens)

                    # self.data_token_by_word[k].append(token)
        # print(self.data_token)
        return self.data_token_by_word


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
            logits = llm.get_logits_from_input_ids(self.prompt_tokens)
            next_allowed_tokens = set()
            next_allowed_tokens.add(terminating_token)
            complete_fn_len = len(complete_fn_tokens)
            for fn in allowed_token:
                if len(fn) > complete_fn_len:
                    next_allowed_tokens.add(fn[complete_fn_len])
                ## If two function shares common name upto some extent, this will create problem, need to handle it
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

    def generate_args_val(self, allowed_tokens: List[int], arg_type: str) -> List:
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
            logits = llm.get_logits_from_input_ids(self.prompt_tokens)
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
                       allowed_idx: set[int]) -> int:
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


class PromptGenerator:
    def __init__(self, llm, fn_tokens: List[List[int]],
                 args_list: List[List[str]], fn_names: List[str],
                 args_types: List[List[str]]) -> None:
        self.llm = llm
        self.fn_tokens = fn_tokens
        self.fn_names = fn_names
        self.args_list = args_list
        self.args_types = args_types
        self.constrain_decoder = ConstrainDecoder(llm)

    def add_str_to_prompt(self, string: str) -> None:
        str_patch = string
        token_patch = tokenize_string(str_patch, self.llm)
        self.constrain_decoder.add_to_prompt(token_patch)

    def tokenize_prompt(self, prompt: str) -> List[int]:
        splitted_prompt = prompt.split(" ")
        prompt_tokens = tokenize_string(prompt, self.llm)
        for word in splitted_prompt:
            prompt_tokens.append(self.llm._encode(word).tolist()[0][0])
        # print(f"Prompt: {prompt_tokens}")
        return prompt_tokens

    def generate(self, prompt: str):
        prompt_tokens = tokenize_string(prompt, self.llm)
        # prompt_tokens = self.tokenize_prompt(prompt)

        # prompt_2D = self.tokenize_prompt(prompt)
        # print(f"Prompt token 2D: {prompt_2D}")

        starting_prompt = '\t{\n\t\t"Prompt": ' + '"' + prompt + '"'
        # print(f"{starting_prompt}")
        self.add_str_to_prompt(starting_prompt)
        # final_prompt_token = self.constrain_decoder.generate_function_name([prompt_tokens])
        # print(llm._decode(final_prompt_token))

        self.add_str_to_prompt(',\n\t\t"fn_name": "')
        final_fn = self.constrain_decoder.generate_function_name(self.fn_tokens)
        # print(final_fn)

        final_fn_name = self.llm._decode(final_fn[: -1])
        if final_fn_name in self.fn_names:
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
                arg_val_token = self.constrain_decoder.generate_args_val(prompt_tokens, arg_type)
                arg_val_str = self.llm._decode(arg_val_token)
                if arg_type == 'float' and '.' not in arg_val_str:
                    self.add_str_to_prompt('.0')
                # print(f"arg_val: {arg_val_str}")
                # initial_arg_token += f"{arg_val_str}, "arg_val_str
                if i < total_args - 1:
                    self.add_str_to_prompt(", ")
        self.add_str_to_prompt("},\n\t},")

        # final_token = self.constrain_decoder.get_current_prompt()
        # print(llm._decode(final_token))

    def generate_for_all_prompts(self, prompts: List[str]):
        for prompt in prompts:
            start = time.time()
            self.constrain_decoder.re_initialize_prompt_token()
            initial_token = initial_prompt_toke(prompt, self.llm)
            self.constrain_decoder.add_to_prompt(initial_token)
            self.generate(prompt)
            final_token = self.constrain_decoder.get_current_prompt()
            print(llm._decode(final_token[len(initial_token):]))
            end = time.time()
            print(f"Token generation time: {(end - start):.3f}s")


if __name__ == "__main__":
    try:
        prompts = [
            "How do I calculate my age difference if I was born in year 1998 and "
            "now the year is 2025",
            "What you have generated 'fn_add_two_numbers' is wrong, i am asking about the difference of numbers, not addition of numbers. please correct it and generate it again. Question: How do I calculate my age difference if I was born in year 1998 and "
            "now the year is 2025"
        ]
        # prompt_loc = "data/input/function_calling_tests.json"
        # with open(prompt_loc, 'r') as fl:
        #     data = json.load(fl)
        # prompts = [key["prompt"] for key in data]
        # # print(prompts)

        start = time.time()
        func_tokenizer = DefinedFunctionTokenizer()
        from llm_sdk import Small_LLM_Model
        llm = Small_LLM_Model(device='cpu')
        mid = time.time()
        print(f"LLM loaded in {(mid - start):.3f}s")
        path = "data/input/functions_definition.json"
        allowed_words = func_tokenizer.load_json(path)
        # print(allowed_words)
        # allowed_tokens = func_tokenizer.tokenize_json_manually(llm)
        allowed_tokens = func_tokenizer.tokenize_json_using_llm(llm)
        # print(allowed_tokens)

        # print(f"Arg name: {allowed_words['args_names']}")
        prompt_generator = PromptGenerator(llm, allowed_tokens['fn_name'],
                                           allowed_words['args_names'],
                                           allowed_words['fn_name'],
                                           allowed_words['args_types'])
        prompt_generator.generate_for_all_prompts(prompts)
        end = time.time()
        print(f"function generation time {(end - mid):.3f}s")
    except Exception as e:
        print(f"Error: {e}")
