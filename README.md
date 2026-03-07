# Core_Rank03_call_me_maybe

## setting cache dir
export UV_CACHE_DIR=/home/sudas/MyProjects/uv_cache
uv cache dir

## Encoding Decoding
storing the text for efficient look up and use

## Resources
- [uv environment manager](https://realpython.com/python-uv/)
- [memory info](https://code.tutsplus.com/understand-how-much-memory-your-python-objects-use--cms-25609t)

## step to use 42 computer
**Create your folder in goinfre:
mkdir -p /goinfre/$USER/uv_cache

**Open your configuration file:
nano ~/.zshrc
**Add this line at the very bottom:
export UV_CACHE_DIR="/goinfre/$USER/uv_cache"
**Refresh your current terminal:
source ~/.zshrc
or 
export VIRTUAL_ENV="/goinfre/$USER/envs/call_me_maybe_env"

**Check the variable:
Run echo $UV_CACHE_DIR

**Create a directory for all your environments in goinfre
mkdir -p /goinfre/$USER/envs

**Create the specific venv for your current project
uv venv /goinfre/$USER/envs/call_me_maybe

Important: copy your project also inside the goinfre.

**set this for importing cache from hugging face
nano ~/.zshrc
export HF_HOME="/goinfre/$USER/huggingface"
source ~/.zshrc