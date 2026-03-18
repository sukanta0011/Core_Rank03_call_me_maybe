import time
import json
from typing import Dict, List
# from memory_profiler import profile


class Tokenizer:
    def __init__(self, path: str) -> None:
        self.path: str = path
        self.__decoder_list: List[str] = []
        self.__encoder_dict: Dict[str, int] = {}
        self.unicode_map = {
            "Ġ": " ",
            "Ċ": "\n",
            "ĉ": "\t"
        }
        self._create_encoder_decoder(path)

    def _refine_key(self, key: str) -> str:
        for k, v in self.unicode_map.items():
            key = key.replace(k, v)
        # new_key = key.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        return key

    def _create_encoder_decoder(self, path: str):
        with open(path, 'r') as fl:
            data = json.load(fl)
        self._decoder_list = [""] * (len(data.values()) + 1)
        for key, val in data.items():
            new_key = self._refine_key(key)
            self.__encoder_dict[new_key] = val
            # self.__decoder_list[val] = new_key
            self.__decoder_list.append(new_key)

        del data

    @staticmethod
    def get_longest_str_in_list(str_list: List[str]) -> int:
        max_len: int = 0
        for str in str_list:
            if len(str) > max_len:
                max_len = len(str)
        return max_len
    
    def get_all_tokes(self) -> List[str]:
        return self.__decoder_list

    def encode(self, string: str) -> List[int]:
        # print(f"Received string: {string}")
        tokens: List[int] = []
        str_len = len(string)
        max_tkn_len = self.get_longest_str_in_list(self.__decoder_list)
        left_ptr = 0
        if str_len > max_tkn_len:
            right_ptr = left_ptr + max_tkn_len
        else:
            right_ptr = str_len
        while left_ptr < right_ptr:
            sub_str = string[left_ptr: right_ptr]
            token = self.__encoder_dict.get(sub_str)
            if token is not None:
                # print(f"Match found: {sub_str}, {left_ptr}, {right_ptr}")
                tokens.append(token)
                left_ptr = right_ptr
                if left_ptr < str_len:
                    if (str_len - left_ptr) < max_tkn_len:
                        right_ptr = str_len
                    else:
                        right_ptr = left_ptr + max_tkn_len
            else:
                right_ptr -= 1
        return tokens

    def decode(self, tokens: List[int]) -> str:
        string = ""
        if isinstance(tokens, List):
            for token in tokens:
                try:
                    string += self.__decoder_list[token]
                except IndexError as e:
                    string = ""
                    raise IndexError(f"Token ID {token}, not found: {e}")
        elif isinstance(tokens, int):
            try:
                string += self.__decoder_list[tokens]
            except IndexError as e:
                string = ""
                raise IndexError(f"Token ID {token}, not found: {e}")
        else:
            print(f"Error: {tokens} can not be decoded")
        return string


def test_toke_encoder():
    start = time.time()
    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model()
    token_path = llm.get_path_to_vocabulary_json()

    tokenizer = Tokenizer(path=token_path)
    # print(len(encoder_decoder._decoder_list))
    # msg = "This is a classic 'hardware vs. software' version mismatch often seen in older workstations. Your NVIDIA GT 1030 has a Pascal architecture (Compute Capability 6.1), but modern versions of PyTorch (like the 2.9.1 required in your dependencies) have dropped support for anything older than Volta (7.0) in their pre-compiled binaries. The workstation's GPU (GT 1030) has a Compute Capability of 6.1, which is deprecated in PyTorch 2.x. To ensure stability and graceful error handling as required by the subject, I forced the model to CPU mode, which still meets the < 5 minute processing requirement for the test prompts."
    msg = "Greet shrek"
    start_1 = time.time()
    tokens1 = llm._encode(msg).tolist()[0]
    print(f"Built-in encoder: {tokens1}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")
    start_1 = time.time()
    tokens2 = tokenizer.encode(msg)
    print(f"Custom encoder: {tokens2}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")

    start_1 = time.time()
    print(f"Built-in decoder: {[llm._decode(token) for token in tokens1]}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")
    start_1 = time.time()
    print(f"Custom decoder: {[tokenizer.decode(token) for token in tokens2]}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")

    end = time.time()
    print(f"Time taken: {(end - start):.3f}s")


if __name__ == "__main__":
    # test()
    test_toke_encoder()
