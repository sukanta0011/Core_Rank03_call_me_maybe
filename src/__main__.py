import time
import sys
# from src.helper_functions import load_json
# from src.ConstrainDecoder import ConstrainDecoder
from typing import List
from pydantic import TypeAdapter
from llm_sdk import Small_LLM_Model
from src.parser import (
    Parser, ResourcePath)
from src.constrain_decoder import ConstrainDecoder, Output
from src.tokenizer import Tokenizer


def main() -> None:
    try:
        my_prompts = [
            # "How do I calculate my age difference if I was born in year 1996 "
            # "and now the year is 2025",
            # "Replace all alphabets in 'hi1245 assuu152' with digit 0",
            # "Replace all the 'alphabet' in 'hi1245 assuu152' with digit 0",
            # "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
            # "Substitute the digits in the string 'Hello 34 I'm 233 years old' with 'NUMBERS'",
            # "Substitute all alphabet between a-z in the string 'Hello 34 I'm 233 years old' with digits",
            # "Replace all '[aeiou]' in 'Programming is fun' with '*'",
            # "Replace all vowels in 'Programming is fun' with asterisks",
            # "Replace consonants in 'Programming is fun' with hash",
            # "Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'",
            # "Replace the 'space' in 'Programming is fun' with 'underscore'",
            # "Replace the spaces in 'Programming is fun' with underscore",
            # "Replace all spaces in 'Programming   is fun' with underscore",
            # "replace the user in email id user@email.com with user123",
            # "replace the 'user' in email id 'user@email.com' with 'user123'",
            # "replace all a in the word hella warld with o",
            # "replace all 'a' in the word 'hella warld' with 'o'",

            # "what is the sum of -20.25 and 30.57?",
            "what is the total sum three numbers 5, 2 and 3?",

            # "reverse sukanta das",
            # "reverse 'sukanta das'",
            # "'reverse' 'sukanta'",
            # "reverse sukanta",
            # "write sukanta from end to start",
            # "write 'sukanta' from end to start",

            # "Greet sukanta das",
            # "Greet 'sukanta das'",
            # "Greet mr. das",
            # "hi mr. unnamed",
            # "'Greet mr. unnamed",
            # "Greet ram",
            # "Greet shrek",
            # "Greet Shrek",
            # "Greet 'Shrek 1234'",
            # "'hello' 'Shrek'",
            # "'Greet John'",
            # "'Welcome John'",
            # "'Hello' 'John'",
            # "let us all welcome shreak to the party",
            # "what is your name? ",

            # "'Mango is a fruit' is a true statement",
            # "'Mango is a fruit' is a correct statement",
            # "'Mango is a fruit' is a right answer",
            # "'Mango is a vegetable' is a wrong answer",
            # "Mango is not a vegetable",
            # "Mango is a vegetable. Wrong",
        ]

        start = time.time()
        llm = Small_LLM_Model(device='cpu')
        token_path = llm.get_path_to_vocab_file()
        tokenizer = Tokenizer(path=token_path)
        encode = tokenizer.encode
        decode = tokenizer.decode
        token_set = tokenizer.get_all_tokes()
        # encode = llm._encode
        # decode = llm._decode
        mid = time.time()
        print(f"LLM loaded in {(mid - start):.3f}s")

        parser = Parser()
        Parser.parse_cli_arguments(sys.argv)
        Parser.validate_resources()
        functions = parser.load_functions(ResourcePath.function_def, encode)
        # print(functions)
        # for fun in functions:
        #     print(fun.fn_name)

        # prompts = parser.load_prompts(ResourcePath.inputs)
        prompts = parser.load_prompts(my_prompts)

        prompt_generator = ConstrainDecoder(
            llm, functions, token_set, encode, decode)
        outputs = prompt_generator.generate_for_all_prompts(prompts)
        end = time.time()
        print(f"function generation time {(end - mid):.3f}s")

        # Saving the outputs
        adapter = TypeAdapter(List[Output])
        json_data = adapter.dump_json(outputs, indent=4).decode('utf-8')
        with open(ResourcePath.outputs, 'w') as fl:
            fl.write(json_data)

    except Exception as e:
        print(e)


def extract_probability():
    with open("src/demo.txt", "r") as fl:
        data = fl.read()
    data = data[1: -1].split(",")
    data = [float(d) for d in data]
    return data


if __name__ == "__main__":
    # test_llm()
    main()
