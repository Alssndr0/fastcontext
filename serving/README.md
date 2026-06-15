# Serving FastContext models locally

FastContext is a client: it talks to any **OpenAI-compatible** chat-completions endpoint
via `BASE_URL` / `MODEL` / `API_KEY` (`src/fastcontext/agent/agent_factory.py`). This
directory has two scripts to stand up a fast local server for
[`microsoft/FastContext-1.0-4B-RL`](https://huggingface.co/microsoft/FastContext-1.0-4B-RL).

The published model is a **vanilla Qwen3-4B** (`Qwen3ForCausalLM`, 262K context), so every
major engine supports it natively.

| Platform | Script | Engine | Why |
| --- | --- | --- | --- |
| NVIDIA / CUDA | `serve_vllm.sh` | [vLLM](https://docs.vllm.ai) | Mature, top-tier Qwen3 support, prefix caching, robust tool-calling. |
| Apple Silicon | `serve_mlx.sh` | [mlx-openai-server](https://github.com/cubist38/mlx-openai-server) | Native MLX with reliable Qwen3 tool-call parsing. |

Both expose the same OpenAI API, so the FastContext CLI is identical against either.

## Quick start (Makefile)

From the repo root:

```bash
make serve                       # vLLM, CUDA, fp8, 32k context (override: MAX_MODEL_LEN=140000)
make serve-mlx                   # MLX, Apple Silicon
make verify                      # server up? tool-calling parsed?
make explore Q="find the auth middleware"
eval "$(make print-env)"         # export BASE_URL/MODEL/API_KEY into your shell
```

The sections below explain what those targets run and how to size them.

## Two things that matter

1. **Tool-calling is required.** The client sends `tools=` and reads back structured
   `message.tool_calls` (`src/fastcontext/agent/llm.py`). The server must parse Qwen3 tool
   calls into OpenAI format — hence `--tool-call-parser` (`hermes` for vLLM, `qwen3` for
   MLX). If you see raw `<tool_call>` text in the model's `content`, the parser is wrong.

2. **The served model name must contain `qwen`.** The client only sends Qwen3's
   `enable_thinking=false` + `top_k=20` when `"qwen"` is in the model name. The RL explorer
   is meant to run with thinking **off**, so the scripts serve it as
   `qwen3-fastcontext-4b-rl`. (The client also matches `fastcontext` in the name as a
   fallback, so the raw HF id works too — but the `qwen` name is the canonical setup.)

## CUDA (vLLM)

```bash
uv pip install vllm           # or: pip install vllm
make serve                    # == ./serving/serve_vllm.sh
```

**Defaults:** fp8 weights (`QUANTIZATION=fp8`) + fp8 KV cache (`KV_CACHE_DTYPE=fp8`) +
`MAX_MODEL_LEN=32768` (fits an 8 GB card). 32k is ample for repo exploration — the agent
reads narrow windows over a handful of turns, so context is driven by accumulated tool
output, not repo size. Raise it on bigger GPUs.

**Sizing.** Qwen3-4B is GQA (8 KV heads), so its KV cache is small: **~144 KiB/token bf16,
~72 KiB/token fp8** (≈ **14.5k tokens per GB** of fp8 KV). fp8 weights are ~4 GB. The 32k
default needs only ~2.3 GB KV + ~4 GB weights. Comfortable, actually fillable
`MAX_MODEL_LEN` per card (fp8 weights + fp8 KV, ~0.6 GB overhead, 0.9 util):

| GPU VRAM | comfortable MAX_MODEL_LEN |
| --- | --- |
| 8 GB (e.g. 8 GB Blackwell) | ~32K–40K |
| 12 GB | ~85K |
| 16 GB | ~140K |
| 24 GB (4090) | ~240K (≈ full) |
| ≥32 GB | 262144 (full) |

> **vLLM vs sglang on the context cap.** vLLM checks at startup that a single sequence of
> `--max-model-len` fits the KV pool and **errors** if it can't — so on an 8 GB card you
> must keep `MAX_MODEL_LEN` near ~32K. sglang treats `--context-length` as a *ceiling* and
> sizes its KV pool from `--mem-fraction-static`, so it lets you set 100K+ without
> preallocating it (it only fails if a run actually grows that large). The numbers above are
> *fillable* tokens; a true 100K on 8 GB isn't possible for this model regardless of engine
> (100K of fp8 KV alone is ~7 GB).

**fp8 weights** run natively on Ada/Hopper/Blackwell (SM ≥ 8.9). On older Ampere (e.g. A100)
they may fail to load — disable with `make serve QUANTIZATION=` (bf16 weights, ~8 GB) and the
same fp8 KV cache still applies.

> **sglang alternative** (near-equal performance):
> ```bash
> python -m sglang.launch_server --model-path microsoft/FastContext-1.0-4B-RL \
>   --served-model-name qwen3-fastcontext-4b-rl \
>   --tool-call-parser qwen3 --context-length 32768 --port 8000
> ```

## Apple Silicon (MLX)

```bash
uv pip install mlx-lm mlx-openai-server
./serving/serve_mlx.sh        # converts to 8-bit MLX on first run, then serves
```

## Use it from the CLI

```bash
export BASE_URL="http://localhost:8000/v1"
export MODEL="qwen3-fastcontext-4b-rl"   # match --served-model-name
export API_KEY="dummy"                    # any non-empty string

fastcontext -q "Find where the agent loop dispatches tool calls" --citation --verbose
```

## Verify end-to-end

```bash
# 1. Server is up and advertises the model
curl -s http://localhost:8000/v1/models | python -m json.tool

# 2. Tool-calls are parsed (look for a "tool_calls" field, not <tool_call> in content)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-fastcontext-4b-rl",
    "messages": [{"role": "user", "content": "List files matching *.py"}],
    "tools": [{"type": "function", "function": {
      "name": "glob", "description": "find files by glob",
      "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}}}],
    "tool_choice": "auto"
  }' | python -m json.tool

# 3. Full CLI run from inside a repo
fastcontext -q "Locate the request validation logic" --citation --verbose
```

A second CLI run should show a faster first token thanks to prefix caching on the shared
system prompt.
