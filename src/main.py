import time
from src.helper_functions import load_json
# from src.ConstrainDecoder import ConstrainDecoder
from src.parser import Parser
from src.PromptGenerator import PromptGenerator
from src.tokenizer import Tokenizer


def test_llm():
    # start = time.time()

    from llm_sdk import Small_LLM_Model

    start = time.time()
    llm = Small_LLM_Model()
    token_path = llm.get_path_to_vocabulary_json()
    tokenizer = Tokenizer(path=token_path)

    txt = "What is the addition of 2 and 3?"
    json_txt = str(load_json())
    pre_prompt = "You need to act as function generator\n After reading the " \
                 "user question, your job will be to provide the answer\n"\
                 "Answer format will be: prompt: question asked by the user \n fn_name: applicable function name\n args: required arguments with types"\
                 f"Available function: {json_txt}\n"

    question = f"Question: {txt}\n"
    answer = ""
    func_tokens = tokenizer.encode(json_txt)

    for i in range(100):
        print(answer)
        prompt = f"{pre_prompt} + {question} + {answer}"
        tokens = tokenizer.encode(prompt)
        logits = llm.get_logits_from_input_ids(tokens[0])
        logits_dict = {idx: tkn for idx, tkn in enumerate(logits)}

        max_val = -100
        max_key = -1
        for key, val in logits_dict.items():
            if val > max_val:
                max_val = val
                max_key = key
        decoded_token = encoder_decoder.decode([max_key])
        answer += str(decoded_token)
        if "}" in decoded_token:
            break


    # tokens = ["{", "fn", "_name"]

    # path = "/home/sudas/.cache/huggingface/hub/" \
    #        "models--Qwen--Qwen3-0.6B/snapshots/" \
    #        "c1899de289a04d12100db370d81485cdf75e47ca/vocab.json"
    # with open(path, "r") as fl:
    #     data = json.loads(fl.read())

    # token_list = [data.get(token) for token in tokens if data.get(token) is not None]
    # print(token_list)

    # logits = llm.get_logits_from_input_ids(token_list)
    # print(max(logits), min(logits))
    # print(data)

    end = time.time()
    print(f"Testing the LLM, Time: {(end - start):.3f}")


def extract_probability():
    with open("src/demo.txt", "r") as fl:
        data = fl.read()
    data = data[1: -1].split(",")
    data = [float(d) for d in data]
    return data


if __name__ == "__main__":
    # try:
    import json
    prompts = [
        # "How do I calculate my age difference if I was born in year 1996 "
        # "and now the year is 2025",
        "Substitute the digits in the string 'Hello 34 I'm 233 years old' with 'NUMBERS'",
        # "Replace all vowels in 'Programming is fun' with asterisks",
        # "Greet 'Sukanta'",
        # "what is the sum of -2 and 3?",
        # "what is the total - sum of 2 and 3?"
    ]
    # prompt_loc = "data/input/function_calling_tests.json"
    # with open(prompt_loc, 'r') as fl:
    #     data = json.load(fl)
    # prompts = [key["prompt"] for key in data]
    # print(prompts)

    start = time.time()
    func_tokenizer = Parser()
    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model(device='cpu')
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
