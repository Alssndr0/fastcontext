#!/usr/bin/env bash
#
# Serve microsoft/FastContext-1.0-4B-RL with vLLM (NVIDIA / CUDA).
#
# FastContext-1.0-4B-RL is a vanilla Qwen3-4B (Qwen3ForCausalLM), so vLLM runs it
# natively -- no --trust-remote-code needed. The two flags that matter for the
# FastContext client are native tool-calling and prefix caching (the system prompt
# and growing transcript are reused on every exploration turn).
#
# Defaults: fp8 weights + fp8 KV cache, 32768-token context (fits an 8 GB card).
#
# Usage:
#   ./serving/serve_vllm.sh
#   MAX_MODEL_LEN=140000 ./serving/serve_vllm.sh       # bigger GPU (e.g. 16 GB)
#   QUANTIZATION= ./serving/serve_vllm.sh              # bf16 weights (older Ampere)
#
# Then point the CLI at it (see serving/README.md):
#   export BASE_URL="http://localhost:${PORT:-8000}/v1"
#   export MODEL="qwen3-fastcontext-4b-rl"   # must match SERVED_NAME below
#   export API_KEY="dummy"
#
set -euo pipefail

# Use the repo's .venv vllm if present, so this works run directly (not just via make).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[[ -d "${REPO_ROOT}/.venv/bin" ]] && PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# Machine-local overrides (gitignored): export MAX_MODEL_LEN, GPU_MEM_UTIL, etc.
# so per-box tuning applies whether you run this script directly or via `make serve`.
LOCAL_ENV="$(dirname "$0")/serve.local.env"
# shellcheck disable=SC1090
[[ -f "${LOCAL_ENV}" ]] && source "${LOCAL_ENV}"

MODEL_PATH="${MODEL_PATH:-microsoft/FastContext-1.0-4B-RL}"

# IMPORTANT: the served name must contain "qwen" so the FastContext client
# (src/fastcontext/agent/llm.py) sends top_k=20 and enable_thinking=false. The
# RL explorer is meant to run with Qwen3 "thinking" OFF.
SERVED_NAME="${SERVED_NAME:-qwen3-fastcontext-4b-rl}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Context window. The model supports 262144. KV cache for Qwen3-4B is small
# (GQA, 8 KV heads): ~144 KiB/token bf16, ~72 KiB/token fp8 (~14.5k tokens/GB).
# Default 32768 fits an 8 GB card (fp8 weights ~4 GB leave ~2.6 GB KV ≈ 36-40k
# tokens) and is ample for repo exploration. NOTE: vLLM hard-caps max-model-len
# to what fits the KV pool at launch and errors if it can't -- raise this on
# bigger GPUs (e.g. ~140000 on 16 GB, ~240000 on 24 GB).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# Qwen3 tool-call parser. Confirm against your installed vLLM:
#   vllm serve --help | grep -A3 tool-call-parser
# "hermes" is the standard Qwen3 parser; Qwen3-Coder variants use "qwen3_coder".
TOOL_PARSER="${TOOL_PARSER:-hermes}"

# fp8 by default (set to "" to disable):
#   KV_CACHE_DTYPE=fp8  -> ~2x context per GB (~14.5k tokens/GB). Works on Ampere+.
#   QUANTIZATION=fp8    -> fp8 weights, ~halve weight memory + faster. Native on
#                          Ada/Hopper/Blackwell (SM>=8.9); on older Ampere (e.g.
#                          A100) set QUANTIZATION= to disable if vLLM errors on load.
# Use ${VAR-fp8} (no colon) so an explicitly empty value is honored: `QUANTIZATION=`
# disables the flag (bf16 weights), while an unset variable still defaults to fp8.
# With `:-` an empty value would wrongly fall back to fp8.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE-fp8}"
QUANTIZATION="${QUANTIZATION-fp8}"

# NOTE: we deliberately do NOT pass --reasoning-parser. Thinking is disabled by
# the client, and enabling the reasoning parser can break Qwen3 tool-call
# parsing in vLLM (vllm-project/vllm#19513).
args=(
  serve "${MODEL_PATH}"
  --served-model-name "${SERVED_NAME}"
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_PARSER}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEM_UTIL}"
  --enable-prefix-caching
  --host "${HOST}"
  --port "${PORT}"
)

[[ -n "${KV_CACHE_DTYPE}" ]] && args+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
[[ -n "${QUANTIZATION}" ]] && args+=(--quantization "${QUANTIZATION}")

# Append any extra raw vLLM flags (e.g. EXTRA_ARGS="--enforce-eager" to free the
# CUDA-graph capture buffers for more KV cache on small/WSL GPUs).
[[ -n "${EXTRA_ARGS:-}" ]] && args+=(${EXTRA_ARGS})

# Preflight: parser names occasionally change between vLLM releases, and the
# FastContext client depends on working tool-calling. Warn (don't abort) if the
# requested parser isn't listed in this vLLM. Set SKIP_PARSER_CHECK=1 to skip
# (the --help call imports vLLM and can take several seconds).
if [[ -z "${SKIP_PARSER_CHECK:-}" ]]; then
  if help_out="$(vllm serve --help 2>/dev/null)"; then
    if ! grep -qw -- "${TOOL_PARSER}" <<<"${help_out}"; then
      echo "WARNING: tool-call parser '${TOOL_PARSER}' is not listed by 'vllm serve --help'." >&2
      echo "         FastContext requires working tool-calling. List parsers with:" >&2
      echo "           vllm serve --help | grep -A3 tool-call-parser" >&2
      echo "         then re-run with TOOL_PARSER=<name> (or SKIP_PARSER_CHECK=1 to silence)." >&2
    fi
  else
    echo "NOTE: could not run 'vllm serve --help' for the parser preflight; skipping." >&2
  fi
fi

echo "Launching: vllm ${args[*]}"
exec vllm "${args[@]}"
