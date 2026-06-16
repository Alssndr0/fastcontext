import os

from fastcontext.agent.agent import Agent
from fastcontext.agent.llm import LLM
from fastcontext.agent.tool.tool import ToolSet

from fastcontext.agent.utils import load_system_prompt


def make_fastcontext_agent(
    work_dir: str,
    trajectory_file: str | None = None,
    **kwargs,
) -> Agent:
    name = "FastContext"
    system_prompt = kwargs.get("system_prompt", None)
    if system_prompt is None:
        system_prompt = load_system_prompt(work_dir)

    # Default to the local OpenAI-compatible server so the CLI is zero-config
    # against a `make serve` instance; env vars override for remote endpoints.
    llm = LLM(
        model=os.getenv("MODEL", "qwen3-fastcontext-4b-rl"),
        api_key=os.getenv("API_KEY", "dummy"),
        base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
    )

    from fastcontext.agent.tool.glob import GlobTool
    from fastcontext.agent.tool.grep import GrepTool
    from fastcontext.agent.tool.read import ReadTool

    toolset = ToolSet([ReadTool(), GlobTool(), GrepTool()], work_dir=work_dir)
    return Agent(
        name=name,
        system_prompt=system_prompt,
        llm=llm,
        toolset=toolset,
        trajectory_file=trajectory_file,
        work_dir=work_dir,
    )
