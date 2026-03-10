import time
import json

from src.helper_functions import load_json
# from src.ConstrainDecoder import ConstrainDecoder
from src.parser import Parser
from src.prompt_generator import PromptGenerator
from src.tokenizer import Tokenizer


def test_llm():
    # prompts = [
    #     # "How do I calculate my age difference if I was born in year 1996 "
    #     # "and now the year is 2025",
    #     # "Substitute the digits in the string 'Hello 34 I'm 233 years old' with 'NUMBERS'",
    #     # "Replace all vowels in 'Programming is fun' with asterisks",
    #     # "Hello Sukanta",
    #     # "what is the sum of -2 and 3?",
    #     # "what is the total-sum of 2 and 3?",
    #     # "Greet shrek",
    #     # "Greet Shrek",

    #     "do some random function",
    # ]
    prompt_loc = "data/input/function_calling_tests.json"
    with open(prompt_loc, 'r') as fl:
        data = json.load(fl)
    prompts = [key["prompt"] for key in data]
    print(prompts)

    start = time.time()
    func_tokenizer = Parser()
    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model()
    token_path = llm.get_path_to_vocabulary_json()
    tokenizer = Tokenizer(path=token_path)
    mid = time.time()
    print(f"LLM loaded in {(mid - start):.3f}s")
    path = "data/input/functions_definition.json"
    allowed_words = func_tokenizer.load_json(path)
    # print(allowed_words)
    # allowed_tokens = func_tokenizer.tokenize_json_manually(llm)
    allowed_tokens = func_tokenizer.tokenize_json_using_custom_tokenizer(llm, tokenizer)
    # print(allowed_tokens)

    # print(f"Arg name: {allowed_words['args_names']}")
    prompt_generator = PromptGenerator(llm, allowed_tokens['fn_name'],
                                       allowed_words['args_names'],
                                       allowed_words['fn_name'],
                                       allowed_words['args_types'],
                                       tokenizer)
    prompt_generator.generate_for_all_prompts(prompts)
    end = time.time()
    print(f"function generation time {(end - mid):.3f}s")
    # except Exception as e:
    #     print(f"Error: {e.__class__}, {e}")



def extract_probability():
    with open("src/demo.txt", "r") as fl:
        data = fl.read()
    data = data[1: -1].split(",")
    data = [float(d) for d in data]
    return data


if __name__ == "__main__":
    test_llm()
    