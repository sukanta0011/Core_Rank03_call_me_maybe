import json
from llm_sdk import Small_LLM_Model

small_llm = Small_LLM_Model()
file_path = small_llm.get_path_to_vocabulary_json()
print(file_path)
print("Testing the uv venv")
