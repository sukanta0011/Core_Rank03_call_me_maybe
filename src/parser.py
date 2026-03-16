import json
from pathlib import Path
from functools import singledispatchmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, List, Callable, Any
from pydantic import BaseModel, Field, model_validator
# from src.helper_functions import tokenize_string, split_word
# from src.tokenizer import Tokenizer
from src.custom_errors import CLIParsingError, SourceError


class Flags(StrEnum):
    INPUT: str = "--input"
    FUNCTION: str = "--functions_definition"
    OUTPUT: str = "--output"


@dataclass
class ResourcePath:
    function_def: str = "data/input/functions_definition.json"
    # inputs: str = "data/input/function_calling_tests.json"
    inputs: str = "data/input/test_prompts.json"
    outputs: str = "data/output/function_calling_results.json"


class FnInfo(BaseModel):
    fn_name: str = Field(min_length=2, alias="fn_name")
    args_names: List[str] = Field(alias="args_names")
    args_types: Dict[str, str] = Field(alias="args_types")
    return_type: str = Field(alias="return_type")
    fn_name_token: List[int]

    @model_validator(mode='after')
    def function_name_validator(self) -> 'FnInfo':
        if not self.fn_name.startswith('fn'):
            raise ValueError(
                "Function name should start with 'fn")
        return self

    @model_validator(mode='after')
    def args_checker(self) -> 'FnInfo':
        if len(self.args_names) != len(self.args_types):
            raise ValueError(
                "args_name and args_types dimensions are not matching")
        return self


class Prompts(BaseModel):
    prompt: str = Field(min_length=1, alias="prompt")


class Parser():
    def __init__(self):
        self.functions: List[FnInfo] = []
        self.prompts: List[Prompts] = []

    def load_functions(self, path: str, encode: Callable) -> List[FnInfo]:
        with open(path, "r") as fl:
            data = json.load(fl)
        for info in data:
            info['fn_name_token'] = encode(info['fn_name'])
            self.functions.append(FnInfo.model_validate(info))

        return self.functions

    @singledispatchmethod
    def load_prompts(self, args: Any) -> List[Prompts]:
        raise NotImplementedError("No prompts/json file provided")

    @load_prompts.register
    def _(self, args: str) -> List[Prompts]:
        with open(args, "r") as fl:
            data = json.load(fl)
        for info in data:
            self.prompts.append(Prompts.model_validate(info))
        return self.prompts

    @load_prompts.register
    def _(self, args: list) -> List[Prompts]:
        for info in args:
            self.prompts.append(Prompts(prompt=info))
        return self.prompts

    # def tokenize_json_using_custom_tokenizer(self, encode: Callable) -> Dict:
    #     for k, v in self.data_str.items():
    #         if k not in self.data_token.keys():
    #             self.data_token[k] = []
    #         for word in v:
    #             if isinstance(word, str):
    #                 tokens = encode(word)
    #                 self.data_token[k].append(tokens)
    #             elif isinstance(word, list):
    #                 # for token in tokens[0]:
    #                 tokens = []
    #                 for sub_word in word:
    #                     token = encode(sub_word)
    #                     tokens.append(token)
    #             self.data_token[k].append(tokens)
    #     # print(self.data_token)
    #     return self.data_token

    # def tokenize_json_manually(self, llm) -> Dict:
    #     for k, v in self.data_str.items():
    #         if k not in self.data_token_by_word.keys():
    #             self.data_token_by_word[k] = []
    #         for word in v:
    #             tokens = []
    #             sub_words = split_word(word)
    #             for sub_word in sub_words:
    #                 # print(f"Sub word: {sub_word}")
    #                 try:
    #                     token = tokenize_string(word, llm)
    #                     # print(f"Tokens: {token[0]}")
    #                     for sub_token in token:
    #                         tokens.append(sub_token)
    #                 except Exception as e:
    #                     print(f"{e}")
    #             self.data_token_by_word[k].append(tokens)

    #                 # self.data_token_by_word[k].append(token)
    #     # print(self.data_token)
    #     return self.data_token_by_word

    @staticmethod
    def validate_resources() -> None:
        path_obj = Path(ResourcePath.function_def)
        if not path_obj.exists():
            raise SourceError("function_definition path is wrong")
        if path_obj.suffix != '.json':
            raise SourceError("function_definition need to be in 'json' format")

        path_obj = Path(ResourcePath.inputs)
        if not path_obj.exists():
            raise SourceError("Input path is wrong")
        if path_obj.suffix != '.json':
            raise SourceError("Input need to be in 'json' format")

        path_obj = Path(ResourcePath.outputs)
        if not path_obj.exists():
            try:
                dir_path = Path(path_obj.parent)
                dir_path.mkdir(parents=True, exist_ok=True)
                f = open(ResourcePath.outputs, 'x')
                f.close()
            except Exception as e:
                raise SourceError(e)

    @staticmethod
    def parse_cli_arguments(args: List[str]) -> None:
        if len(args) == 1:
            return

        for idx in range(1, len(args) - 1, 2):
            try:
                if args[idx] == Flags.FUNCTION:
                    ResourcePath.function_def = args[idx + 1]
                elif args[idx] == Flags.INPUT:
                    ResourcePath.inputs = args[idx + 1]
                elif args[idx] == Flags.OUTPUT:
                    ResourcePath.outputs = args[idx + 1]
                else:
                    raise CLIParsingError(f"Flag '{args[idx]}' is unknown.")
            except IndexError:
                raise CLIParsingError(
                    f"{args[idx]} missing Path."
                )


if __name__ == "__main__":
    pass
