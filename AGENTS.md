# AGENTS.md — Run FastContext locally, end to end

This file is a complete, reproducible runbook for serving the
[`microsoft/FastContext-1.0-4B-RL`](https://huggingface.co/microsoft/FastContext-1.0-4B-RL)
model and driving it with the `fastcontext` CLI — on **macOS (Apple Silicon / MLX)** and on
**NVIDIA (CUDA / vLLM)**. Follow the path for your platform. Everything here has been verified
end to end.

## 0. What you're setting up

- **`fastcontext`** is a CLI repo-exploration agent. It is a *client*: it talks to any
  **OpenAI-compatible** chat-completions endpoint set via `BASE_URL` / `MODEL` / `API_KEY`.
- **The model** `microsoft/FastContext-1.0-4B-RL` is a **vanilla Qwen3-4B** (`Qwen3ForCausalLM`,
  262K context, bf16). No `trust_remote_code` needed; every major engine runs it natively.
- **Goal:** stand up a fast local server, point the CLI at it, and get accurate `file:line`
  citations back.

Two non-negotiable requirements the whole setup hinges on:

1. **The server must do native tool-calling** — the client sends `tools=` and reads back
   structured `message.tool_calls`. Engines need an explicit tool-call parser *and*
   auto-tool-choice enabled.
2. **The served model name must contain `qwen`** (the scripts use `qwen3-fastcontext-4b-rl`).
   The client only sends Qwen3's `enable_thinking=false` + `top_k=20` when the model name
   contains `qwen` (it also matches `fastcontext` as a fallback). The RL explorer must run with
   thinking **off**.

The repo ships helper scripts so you don't type these by hand:
- `serving/serve_vllm.sh` — CUDA / vLLM
- `serving/serve_mlx.sh` — Apple Silicon / MLX
- `Makefile` — `make serve`, `make serve-mlx`, `make verify`, `make explore`, `make print-env`

---

## 1. Common prerequisites (BOTH platforms)

- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)**.
- **Git** + this repository checked out.
- **ripgrep (`rg`) MUST be on `PATH`.** The `Grep` tool shells out to `rg`; without it the agent
  gets no data and **hallucinates** answers (this is the #1 failure mode — see Gotchas).
  - macOS: `brew install ripgrep`
  - Debian/Ubuntu: `sudo apt-get install -y ripgrep`
  - Verify: `rg --version` and `command -v rg`
- **Hugging Face access** to download the model (`huggingface.co` reachable). For gated/rate
  limits, `huggingface-cli login` or set `HF_TOKEN`.

Install the CLI (from the repo root):

```bash
uv tool install .          # provides the `fastcontext` command
fastcontext --help         # sanity check
```

> If you change the source later, see the **uv cache gotcha** in §6 — `uv tool install . --force`
> can silently reuse a cached wheel.

---

## 2. macOS (Apple Silicon / MLX)

Verified on: arm64 macOS, 48 GB unified memory, Python 3.13. A 4B model in 8-bit needs ~4.5 GB;
any 16 GB+ Apple Silicon Mac is fine.

### 2.1 Install the MLX serving stack

```bash
uv tool install mlx-lm                # provides mlx_lm.convert, etc.
uv tool install mlx-openai-server     # OpenAI-compatible server with Qwen3 tool-calling
brew install ripgrep                  # if not already installed
```

We use **mlx-openai-server**, not the stock `mlx_lm.server`: it has reliable Qwen3 tool-call
parsing (`--tool-call-parser qwen3`) and an explicit `--enable-auto-tool-choice` flag.

### 2.2 Convert the model to MLX (one time)

There is **no published MLX build of the RL model** (only an SFT 4-bit), so convert it. This
downloads ~8 GB of bf16 weights, then writes an 8-bit MLX copy (~4 GB). Put it outside the repo
to keep git clean:

```bash
mlx_lm.convert --hf-path microsoft/FastContext-1.0-4B-RL -q --q-bits 8 \
  --mlx-path "$HOME/models/FastContext-4B-RL-mlx-8bit"
```

(8-bit is the quality/speed sweet spot for a 4B explorer; use `--q-bits 4` for a smaller, lower
fidelity build. With 32 GB+ RAM you can skip quantization and serve bf16 directly.)

### 2.3 Launch the server

```bash
MLX_PATH="$HOME/models/FastContext-4B-RL-mlx-8bit" ./serving/serve_mlx.sh
# or, to also run the conversion automatically on first run:
make serve-mlx
```

Under the hood this runs:

```bash
mlx-openai-server launch \
  --model-type lm \
  --model-path "$HOME/models/FastContext-4B-RL-mlx-8bit" \
  --served-model-name qwen3-fastcontext-4b-rl \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3 \
  --context-length 32768 \
  --host 0.0.0.0 --port 8000
```

MLX grows the KV cache lazily per request (no upfront preallocation), so `--context-length` is
just a ceiling — raise it freely. Go to §4 to point the CLI at it.

---

## 3. NVIDIA (CUDA / vLLM)

vLLM is the recommended engine: mature, top-tier Qwen3 support, prefix caching, robust
tool-calling. (sglang is a near-equal alternative — see the note at the end of this section.)

### 3.1 Install

```bash
uv pip install vllm        # in a venv, or: pip install vllm
sudo apt-get install -y ripgrep   # the Grep tool needs it
```

### 3.2 Serve (no conversion needed — vLLM loads from HF directly)

```bash
./serving/serve_vllm.sh
# or
make serve
```

Defaults: **fp8 weights + fp8 KV cache + `MAX_MODEL_LEN=32768`** (fits an 8 GB card). Under the
hood:

```bash
vllm serve microsoft/FastContext-1.0-4B-RL \
  --served-model-name qwen3-fastcontext-4b-rl \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 \
  --kv-cache-dtype fp8 \
  --quantization fp8 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000
```

Notes:
- We deliberately do **not** pass `--reasoning-parser`: thinking is disabled client-side, and
  enabling the reasoning parser can break Qwen3 tool-call parsing (vllm-project/vllm#19513).
- The script runs a preflight that warns if `hermes` isn't a valid parser in your vLLM version.
  Confirm with `vllm serve --help | grep -A3 tool-call-parser`; set `TOOL_PARSER=<name>` to
  override. Set `SKIP_PARSER_CHECK=1` to skip the (slow) preflight.

### 3.3 Context window vs VRAM

Qwen3-4B is GQA (8 KV heads), so its KV cache is small: **~144 KiB/token bf16, ~72 KiB/token
fp8** (≈ 14.5k tokens/GB fp8). fp8 weights ≈ 4 GB. **vLLM hard-caps `--max-model-len` to what
fits the KV pool at launch and errors if it can't.** Comfortable, *actually fillable*
`MAX_MODEL_LEN` (fp8 weights + fp8 KV):

| GPU VRAM | comfortable MAX_MODEL_LEN |
| --- | --- |
| 8 GB (e.g. 8 GB Blackwell) | ~32K (use the 32768 default) |
| 12 GB | ~85K |
| 16 GB | ~140K |
| 24 GB | ~240K |
| ≥32 GB | 262144 (full) |

Override per card: `make serve MAX_MODEL_LEN=140000`. 32k is ample for repo exploration anyway —
context is driven by accumulated tool output over a handful of turns, not repo size.

- **fp8 weights** run natively on Ada/Hopper/Blackwell (SM ≥ 8.9). On older Ampere (e.g. A100)
  they may fail to load — disable with `make serve QUANTIZATION=` (bf16 weights, ~8 GB); the fp8
  KV cache still applies.
- **sglang alternative:**
  ```bash
  python -m sglang.launch_server --model-path microsoft/FastContext-1.0-4B-RL \
    --served-model-name qwen3-fastcontext-4b-rl \
    --tool-call-parser qwen3 --context-length 32768 --port 8000
  ```
  sglang treats `--context-length` as a *ceiling* (KV pool sized from `--mem-fraction-static`),
  so it accepts large values without preallocating — unlike vLLM's strict launch check.

---

## 4. Point the CLI at the server (BOTH platforms)

```bash
export BASE_URL="http://localhost:8000/v1"
export MODEL="qwen3-fastcontext-4b-rl"   # must match --served-model-name (and contain "qwen")
export API_KEY="dummy"                    # any non-empty string; servers ignore it locally
```

Or let the Makefile emit these: `eval "$(make print-env)"`.

Run an exploration from inside the repo you want to explore:

```bash
fastcontext -q "Find where the agent loop dispatches tool calls" --max-turns 6 --citation --verbose
# or
make explore Q="find the request validation logic"
```

You should get a `<final_answer>` block of real `path:line-range` citations.

---

## 5. Verify it actually works (don't trust a clean-looking answer)

```bash
# 1) Server is up and advertises the model
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# 2) Tool-calling is parsed into structured tool_calls (NOT raw <tool_call> text in content)
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-fastcontext-4b-rl",
       "messages":[{"role":"user","content":"List the Python files in src using the glob tool."}],
       "tools":[{"type":"function","function":{"name":"glob","description":"find files by glob",
         "parameters":{"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"]}}}],
       "tool_choice":"auto"}' \
  | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('tool_calls:', m.get('tool_calls')); print('content:', m.get('content'))"
# Expect: tool_calls populated, content None.

# 3) Full CLI run + inspect the trajectory to confirm tools returned REAL data
fastcontext -q "Where is the Read tool implemented?" --max-turns 6 --citation --traj /tmp/fc.jsonl
python3 -c "import json; [print(r.get('role'), (r.get('content') or '')[:80]) for r in map(json.loads, open('/tmp/fc.jsonl'))]"
# Red flag: tool messages containing 'No such file or directory: rg' -> ripgrep not found (see Gotchas).
```

`make verify` runs checks 1 and 2 for you.

Observed performance: a 6-turn exploration on the 8-bit MLX model (Apple Silicon) completes in
~14–26 s and returns accurate citations.

---

## 6. Gotchas we hit (read this — it will save you an hour)

1. **ripgrep not on PATH → silent hallucination.** `grep.py` resolves `rg` via
   `os.environ["FASTCONTEXT_RG"]` → `shutil.which("rg")` → `/usr/bin/rg`. If none resolve, every
   `Grep` fails and the model invents plausible-but-wrong citations (e.g. files/frameworks that
   don't exist). Always `brew install ripgrep` / `apt-get install ripgrep` first and confirm with
   `rg --version`. Override the binary with `FASTCONTEXT_RG=/path/to/rg`.
   *(Historically `grep.py` hardcoded `/usr/bin/rg`, which broke on Homebrew macOS — now fixed.)*

2. **`uv tool install . --force` can reuse a cached wheel.** Because the package version is
   `0.1.0`, uv may serve a cached build and silently ignore your source edits. If a code change
   "doesn't take", run `uv cache clean fastcontext && uv tool install . --force`, or just use
   `uv run fastcontext …` from the repo to run live source.

3. **Tool-calling needs BOTH flags.** `--tool-call-parser` alone is not enough; you also need
   `--enable-auto-tool-choice` (true for vLLM *and* mlx-openai-server). Without it the server may
   return tool syntax inside `content` instead of structured `tool_calls`, and the client gets
   nothing.

4. **Served-model-name must contain `qwen`.** Otherwise the client leaves Qwen3 "thinking" on and
   skips `top_k`, changing explorer behavior. Keep `qwen3-fastcontext-4b-rl` (the client also
   matches `fastcontext` as a fallback).

5. **vLLM vs sglang context cap.** vLLM errors at launch if `--max-model-len` exceeds the KV pool;
   sglang accepts a high `--context-length` ceiling without preallocating. A *true* 100K window on
   an 8 GB card is impossible for this model regardless (100K of fp8 KV ≈ 7 GB); the cap you can
   *set* in sglang is not the same as what you can *fill*.

6. **Don't pass `--reasoning-parser` on vLLM** for this setup — thinking is off and it can break
   tool-call parsing (vllm-project/vllm#19513).

---

## 7. One-shot quick reference

```bash
# ---- macOS (MLX) ----
brew install ripgrep
uv tool install . && uv tool install mlx-lm && uv tool install mlx-openai-server
mlx_lm.convert --hf-path microsoft/FastContext-1.0-4B-RL -q --q-bits 8 \
  --mlx-path "$HOME/models/FastContext-4B-RL-mlx-8bit"
MLX_PATH="$HOME/models/FastContext-4B-RL-mlx-8bit" ./serving/serve_mlx.sh   # terminal 1
eval "$(make print-env)"; make verify; make explore Q="find the agent loop" # terminal 2

# ---- NVIDIA (vLLM) ----
sudo apt-get install -y ripgrep
uv tool install . && uv pip install vllm
./serving/serve_vllm.sh                  # terminal 1 (8 GB card: default 32768 is right)
eval "$(make print-env)"; make verify; make explore Q="find the agent loop"  # terminal 2
```
