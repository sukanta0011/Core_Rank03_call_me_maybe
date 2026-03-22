import os
import threading
import streamlit as st
from pathlib import Path
from typing import Tuple, List
from llm_sdk import Small_LLM_Model
from src.tokenizer import Tokenizer
from src.parser import FnInfo, ResourcePaths
from src.constrain_decoder import ConstrainDecoder
from src.__main__ import function_generator


# Tell HuggingFace where to cache the model
# /data persists between restarts on HuggingFace Spaces
os.environ["HF_HOME"] = "/data/huggingface"


@st.cache_resource
def load_shared_resources() -> Tuple:
    """Loads once per container lifetime."""

    # Set paths directly — no CLI parsing needed
    ResourcePaths.function_def = \
        Path("data/input/functions_definition.json")
    ResourcePaths.inputs = \
        Path("data/input/function_calling_tests.json")
    ResourcePaths.outputs = \
        Path("data/output/function_calling_results.json")

    with st.spinner("Loading Qwen3-0.6B... first run takes 2-3 minutes"):
        llm = Small_LLM_Model(device='cuda')
        token_path = llm.get_path_to_vocab_file()
        tokenizer = Tokenizer(path=token_path)
        token_set = tokenizer.get_all_tokes()

        from src.parser import FunctionLoader
        functions = FunctionLoader.load_json(
            ResourcePaths.function_def, tokenizer.encode)
        token_set = tokenizer.get_all_tokes()

        lock = threading.Lock()

    return llm, functions, token_set, tokenizer, lock


def get_session_decoder() -> ConstrainDecoder:
    if "decoder" not in st.session_state:
        llm, functions, token_set, tokenizer, lock =\
            load_shared_resources()
        st.session_state.decoder = ConstrainDecoder(
            llm, functions, token_set,
            tokenizer.encode, tokenizer.decode,
            interface_lock=lock)
    return st.session_state.decoder


st.set_page_config(
    page_title="Call Me Maybe",
    page_icon="📞",
    layout="wide"
)

st.title("🤙 Call Me Maybe")
st.caption(
    "Natural language → structured function calls "
    "using constrained decoding  |  Model: Qwen/Qwen3-0.6B"
)
# st.info(
#     "⏳ First load takes 2-3 minutes while the model downloads (~2GB). "
#     "Subsequent requests are faster."
# )

# Load pipeline once
_, functions, _, _, _ = load_shared_resources()
decoder = get_session_decoder()

col1, col2, col3 = st.columns(
    3, gap="small")

with col1:
    prompt_to_use = st.text_area(
        "Your prompt",
        value="What is the sum of 4250 and 2560?")
    run = st.button("Generate Function Call")

with col2:
    # Show the example prompts
    examples = (
        "What is the of 3 and 5?",
        "Greet john",
        "Reverse the string 'hello'",
        "What is the square root of 16?",
        "Replace all vowels in 'Programming is fun' with asterisks",
    )
    with st.expander("Example Prompts"):
        for prompt in examples:
            st.code(prompt, language=None)


with col3:
    # Show available functions so user knows what to ask
    with st.expander("Available Functions"):
        for fn in functions:
            args = ", ".join(
                f"{k}: {v}" for k, v in fn.args_types.items())
            st.code(f"{fn.fn_name} ({args})")

if run and prompt_to_use:
    st.divider()
    st.markdown("#### 🔄 Live Generation")
    # Create placeholders that update in place
    status_box = st.empty()
    stream_box = st.empty()
    # result_box = st.empty()

    # Accumulate streamed text
    st.session_state.stream_buffer = ""

    def on_token(token_str: str) -> None:
        """Called by TokenGenerator on every new token."""
        st.session_state.stream_buffer += token_str
        stream_box.code(
            st.session_state.stream_buffer,
            language="json"
        )

    # Attach callback before generation
    decoder.set_callback(on_token)

    status_box.caption("⚙️ Generating...")
    try:
        result, cost = function_generator(prompt_to_use, decoder)

        # Clear streaming display
        decoder.set_callback(None)
        stream_box.empty()
        status_box.empty()

        # Show final clean result
        st.success(
            f"Done — {cost.token_used} tokens "
            f"in {cost.time_taken_seconds}s"
        )
        st.subheader("Result")
        st.json({
            "prompt": result.prompt,
            "name": result.name,
            "parameters": result.parameters
        })
        st.markdown(
            f"Tokens used: :green[**{cost.token_used}**] &nbsp;|&nbsp; "
            f"Time: :green[**{cost.time_taken_seconds}s**] &nbsp;|&nbsp; "
            f"Per token: :green[**{cost.avg_time}s**]"
        )
    except Exception as e:
        decoder.set_callback(None)
        st.error(f"Generation failed: {e}")
