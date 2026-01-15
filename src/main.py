import json
import time

start = time.time()

from llm_sdk import Small_LLM_Model

llm = Small_LLM_Model(device="cpu")
# file_path = small_llm.get_path_to_vocabulary_json()
txt = "What is the sum of 2 and 3?"
tokens = llm._encode(txt)
# token_list = [token for token in tokens[0]]
print(tokens)
logits = llm.get_logits_from_input_ids(tokens)
print(logits)
# decoded_logits = llm._decode()

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
