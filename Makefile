# FastContext serving & exploration helpers.
#
#   make serve            # start the vLLM server (CUDA, fp8, 32k context)
#   make serve-mlx        # start the MLX server (Apple Silicon)
#   make explore Q="..."  # run an exploration query against the running server
#   make verify           # check the server is up and tool-calling works
#   make print-env        # print the export lines to wire your own shell
#
# Override any variable inline, e.g.:
#   make serve MAX_MODEL_LEN=140000        # bigger GPU
#   make explore Q="find the auth middleware" PORT=8001

# ---- Configuration (override on the command line) -------------------------
HOST          ?= 0.0.0.0
PORT          ?= 8000
SERVED_NAME   ?= qwen3-fastcontext-4b-rl
MODEL_PATH    ?= microsoft/FastContext-1.0-4B-RL

# vLLM (CUDA) defaults: fp8 weights + fp8 KV, 32k context (fits an 8 GB card;
# raise on bigger GPUs, e.g. MAX_MODEL_LEN=140000 on 16 GB).
MAX_MODEL_LEN ?= 32768
GPU_MEM_UTIL  ?= 0.9
KV_CACHE_DTYPE ?= fp8
QUANTIZATION  ?= fp8
TOOL_PARSER   ?= hermes

# MLX (Apple Silicon) defaults.
Q_BITS        ?= 8

# Client connection (used by `explore` / `verify`). BASE_URL points at localhost
# by default; set MODEL to the served name (must contain "qwen").
BASE_URL      ?= http://localhost:$(PORT)/v1
MODEL         ?= $(SERVED_NAME)
API_KEY       ?= dummy

# Exploration query and flags for `make explore`.
Q             ?= Find where the agent loop dispatches tool calls
FLAGS         ?= --citation --verbose

# Export so the recipe-invoked scripts/CLI inherit them.
export HOST PORT SERVED_NAME MODEL_PATH MAX_MODEL_LEN GPU_MEM_UTIL \
       KV_CACHE_DTYPE QUANTIZATION TOOL_PARSER Q_BITS HF_PATH \
       BASE_URL MODEL API_KEY

.PHONY: help serve serve-mlx explore verify print-env

help:
	@grep -E '^#   make ' $(MAKEFILE_LIST) | sed 's/^#  //'

## Start the vLLM server (CUDA).
serve:
	./serving/serve_vllm.sh

## Start the MLX server (Apple Silicon). Converts to MLX on first run.
serve-mlx:
	HF_PATH=$(MODEL_PATH) ./serving/serve_mlx.sh

## Run an exploration query against the running server.
explore:
	fastcontext -q "$(Q)" $(FLAGS)

## Verify the server is up and tool-calling is parsed.
verify:
	@echo "==> Models advertised at $(BASE_URL):"
	@curl -sf $(BASE_URL)/models | python3 -m json.tool || \
	  { echo "Server not reachable at $(BASE_URL)"; exit 1; }
	@echo "\n==> Tool-call sanity (expect a 'tool_calls' field, not <tool_call> in content):"
	@curl -sf $(BASE_URL)/chat/completions \
	  -H 'Content-Type: application/json' \
	  -d '{"model":"$(MODEL)","messages":[{"role":"user","content":"List files matching *.py"}],"tools":[{"type":"function","function":{"name":"glob","description":"find files by glob","parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}}],"tool_choice":"auto"}' \
	  | python3 -m json.tool

## Print export lines to wire your own shell: eval "$(make print-env)"
print-env:
	@echo 'export BASE_URL="$(BASE_URL)"'
	@echo 'export MODEL="$(MODEL)"'
	@echo 'export API_KEY="$(API_KEY)"'
