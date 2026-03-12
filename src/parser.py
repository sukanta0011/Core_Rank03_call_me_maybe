import json
# from abc import ABC, abstractmethod
# from pydantic import BaseModel
from typing import Dict, Callable
from src.helper_functions import tokenize_string, split_word
from src.tokenizer import Tokenizer


class Parser:
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
                if isinstance(val, dict):
                    self.data_str[key].append(list(val.values()))
                else:
                    self.data_str[key].append(val)

        return self.data_str

    def tokenize_json_using_custom_tokenizer(self, encode: Callable) -> Dict:
        for k, v in self.data_str.items():
            if k not in self.data_token.keys():
                self.data_token[k] = []
            for word in v:
                if isinstance(word, str):
                    tokens = encode(word)
                    self.data_token[k].append(tokens)
                elif isinstance(word, list):
                    # for token in tokens[0]:
                    tokens = []
                    for sub_word in word:
                        token = encode(sub_word)
                        tokens.append(token)
                self.data_token[k].append(tokens)
        # print(self.data_token)
        return self.data_token

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


if __name__ == "__main__":
    pass
