#!/usr/bin/env bash
#
# Serve microsoft/FastContext-1.0-4B-RL on Apple Silicon with MLX.
#
# Uses mlx-openai-server (https://github.com/cubist38/mlx-openai-server), which
# provides reliable OpenAI-compatible tool-calling for Qwen3 -- unlike the stock
# `mlx_lm.server`, whose tool-call support is not reliably available.
#
# There is no published MLX build of the RL model (only an SFT 4-bit), so we
# convert it once with mlx_lm.convert and cache it locally.
#
# Setup (once):
#   uv pip install mlx-lm mlx-openai-server   # or: pip install ...
#
# Usage:
#   ./serving/serve_mlx.sh
#   Q_BITS=4 ./serving/serve_mlx.sh           # smaller/faster, lower fidelity
#
# Then point the CLI at it (see serving/README.md):
#   export BASE_URL="http://localhost:${PORT:-8000}/v1"
#   export MODEL="qwen3-fastcontext-4b-rl"
#   export API_KEY="not-needed"
#
set -euo pipefail

HF_PATH="${HF_PATH:-microsoft/FastContext-1.0-4B-RL}"
Q_BITS="${Q_BITS:-8}"   # 8-bit is the quality/speed sweet spot for a 4B explorer
MLX_PATH="${MLX_PATH:-./FastContext-4B-RL-mlx-${Q_BITS}bit}"

# Must contain "qwen" so the FastContext client disables thinking + sets top_k.
SERVED_NAME="${SERVED_NAME:-qwen3-fastcontext-4b-rl}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# MLX grows the KV cache lazily per request (no upfront preallocation like vLLM),
# so this is just a ceiling. 32768 matches the vLLM default; raise as needed.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"

if [[ ! -d "${MLX_PATH}" ]]; then
  echo "Converting ${HF_PATH} -> ${MLX_PATH} (${Q_BITS}-bit MLX)..."
  mlx_lm.convert --hf-path "${HF_PATH}" -q --q-bits "${Q_BITS}" --mlx-path "${MLX_PATH}"
fi

# `launch` subcommand + --model-type lm are required by mlx-openai-server.
# --enable-auto-tool-choice is required (with --tool-call-parser) for the server
# to return structured tool_calls, which the FastContext client depends on.
echo "Launching mlx-openai-server on ${HOST}:${PORT} (model=${MLX_PATH})"
exec mlx-openai-server launch \
  --model-type lm \
  --model-path "${MLX_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3 \
  --context-length "${CONTEXT_LENGTH}" \
  --host "${HOST}" \
  --port "${PORT}"
