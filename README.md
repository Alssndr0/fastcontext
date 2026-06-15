# FastContext: A Lightweight Repository Explorer for Coding Agents

<p align="center">
  <a href="https://arxiv.org/abs/2606.14066"><img src="https://img.shields.io/badge/arXiv-2606.14066-b31b1b.svg" alt="arXiv"></a>
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  <a href="#overview">🔎 Overview</a> |
  <a href="#installation">📦 Installation</a> |
  <a href="#local-serving">🖥️ Local Serving</a> |
  <a href="#quick-start">⚡ Quick Start</a> |
  <a href="#integrating-into-your-coding-agent">🔌 Integration</a> |
  <a href="#citation">📚 Citation</a>
</p>

FastContext is a lightweight repository-exploration subagent for coding agents. Instead of
letting the main coding agent spend its own context window on broad file reads and code
searches, the main agent delegates a natural-language context query to FastContext.
FastContext explores the repository with read-only tools, issues independent tool calls in
parallel, and returns compact file-line citations as focused evidence for the main agent.

> 📄 **Paper:** This work is based on *FastContext: Training Efficient Repository Explorer for
> Coding Agents* — read it at [arxiv.org/html/2606.14066v1](https://arxiv.org/html/2606.14066v1).

## Overview

Modern coding agents often use the same model to explore a repository and solve the task. This
makes exploration expensive: exploratory reads and searches consume tokens, stay in the
solver's history, and can pollute later reasoning with irrelevant snippets.

FastContext separates repository exploration from solving:

- 🧭 **Delegated exploration**: the main agent asks FastContext for repository context before editing or answering.
- 🔒 **Read-only tools**: FastContext uses `Read`, `Glob`, and `Grep`; it does not modify files.
- ⚙️ **Parallel tool calling**: independent reads and searches can be issued in the same exploration turn.
- 📌 **Compact evidence**: the final response is a short `<final_answer>` block with file paths and line ranges.

The intended contract is simple: FastContext finds the relevant code; the main coding agent
uses that focused evidence to edit, test, or answer.

```text
<final_answer>
/path/to/repo/src/router.py:42-58
/path/to/repo/tests/test_router.py:101-119
</final_answer>
```

FastContext is powered by a small, fast model (the released
[`microsoft/FastContext-1.0-4B-RL`](https://huggingface.co/microsoft/FastContext-1.0-4B-RL), a
Qwen3-4B) served behind an OpenAI-compatible endpoint. A larger coding agent — for example
Claude or GPT — calls the `fastcontext` CLI as a subagent and consumes its citations.

## Installation

FastContext requires Python 3.12 or newer. The repository uses [`uv`](https://docs.astral.sh/uv/) for package
and environment management.

Install the CLI from the repository root:

```bash
uv tool install .
```

For development:

```bash
uv sync --all-groups
```

Build a local wheel:

```bash
uv build
```

The built wheel is written under `dist/`, for example:

```text
dist/fastcontext-0.1.0-py3-none-any.whl
```

## Model Configuration

FastContext expects an OpenAI-compatible chat completions endpoint. Configure:

```bash
export BASE_URL="https://your-endpoint.example/v1"
export MODEL="your-model-name"
export API_KEY="your-api-key"
```

### Local Serving

The released `microsoft/FastContext-1.0-4B-RL` is a standard Qwen3-4B, so it runs on every major
engine. A `Makefile` wraps serving and exploration — pick the target for your platform:

```bash
# NVIDIA / CUDA (vLLM) — fp8 weights + fp8 KV cache, 32k context by default
uv pip install vllm
make serve

# Apple Silicon (MLX)
uv pip install mlx-lm mlx-openai-server
make serve-mlx
```

With the server running, wire your shell and explore:

```bash
eval "$(make print-env)"     # exports BASE_URL / MODEL / API_KEY
make verify                  # confirm the server is up and tool-calling works
make explore Q="find the request validation logic"
```

The 32k default context fits an 8 GB card; raise it on bigger GPUs (e.g.
`make serve MAX_MODEL_LEN=140000` on 16 GB). See [`serving/README.md`](serving/README.md)
for tool-calling requirements, the served-name convention, VRAM/context sizing (and the vLLM-vs-sglang
context-cap difference), and verification details. [`AGENTS.md`](AGENTS.md) is a full,
verified end-to-end serving runbook for both platforms.

## Quick Start

Run FastContext from the repository you want to explore:

```bash
fastcontext \
  --query "Find the files that implement authentication and explain where to make a change" \
  --max-turns 20 \
  --traj .fastcontext/trajectory.jsonl
```

Return only the machine-readable citation block:

```bash
fastcontext \
  --query "Locate the request validation logic" \
  --citation
```

Useful CLI options:

| Option | Description |
| --- | --- |
| `--query`, `-q` | Natural-language exploration request. |
| `--traj`, `-t` | JSONL trajectory output path. |
| `--max-turns` | Maximum exploration turns before forcing a final answer (default 20). |
| `--verbose` | Print intermediate messages and runtime information. |
| `--citation` | Return only the `<final_answer>` block when present. |

## Programmatic Use

```python
import asyncio

from fastcontext.agent.agent_factory import make_fastcontext_agent


async def main() -> None:
    agent = make_fastcontext_agent(
        trajectory_file=".fastcontext/trajectory.jsonl",
        work_dir="/path/to/repo",
    )
    answer = await agent.run(
        prompt="Find where database migrations are defined",
        max_turns=20,
        citation=True,
    )
    print(answer)


asyncio.run(main())
```

## Integrating into your coding agent

FastContext is designed to be driven by a larger coding agent (such as Claude or GPT) as a
read-only exploration subagent. [`docs/INTEGRATION.md`](docs/INTEGRATION.md) explains when to
call it, how to write good queries, how to consume its citations, and includes a copy-pasteable
system-prompt snippet you can drop into your own agent.

## Repository Layout

```text
src/fastcontext/
  cli.py                         Command-line entry point
  agent/
    agent.py                     Agent loop
    agent_factory.py             Default FastContext agent construction
    context.py                   Conversation and trajectory storage
    llm.py                       OpenAI-compatible LLM wrapper
    system.md                    Explorer system prompt
    tool/
      read.py                    Read tool
      glob.py                    Glob tool
      grep.py                    Grep tool
      tool.py                    Tool base classes and ToolSet

serving/                         Example serving scripts and API checks (vLLM, MLX)
docs/                            Integration guide for coding agents
tests/                           Unit and integration-style tests
```

## Development

Run linting:

```bash
uv run ruff check .
```

Run tests:

```bash
uv run pytest -q
```

Build the package:

```bash
uv build
```

## Notes

- FastContext is intended for repository exploration, not code modification.
- Tool outputs are capped to keep interactions responsive.
- The default CLI records trajectories under `.fastcontext/` unless `--traj` is provided.
- For best results, write specific exploration queries that name the behavior, subsystem, error, or files you are trying to locate.

## Citation

FastContext builds on the research described in the following paper. If you find it useful,
please cite:

```bibtex
@misc{zhang2026fastcontexttrainingefficientrepository,
      title={FastContext: Training Efficient Repository Explorer for Coding Agents},
      author={Shaoqiu Zhang and Maoquan Wang and Yuling Shi and Yuhang Wang and Xiaodong Gu and Yongqiang Yao and Rao Fu and Shengyu Fu},
      year={2026},
      eprint={2606.14066},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2606.14066},
}
```

## Acknowledgements

FastContext builds on open research infrastructure for coding agents, including SWE-bench,
Mini-SWE-Agent, and the open model and serving ecosystems.
