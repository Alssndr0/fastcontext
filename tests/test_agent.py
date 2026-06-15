import os

import pytest

from fastcontext.agent.agent import Agent
from fastcontext.agent.llm import LLM
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool

live_server_required = pytest.mark.skipif(
    not os.getenv("BASE_URL"),
    reason="requires a live OpenAI-compatible endpoint (set BASE_URL / MODEL / API_KEY)",
)


@live_server_required
async def test_agent():
    llm = LLM(
        model=os.getenv("MODEL"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    work_dir = os.getcwd()
    toolset = ToolSet(tools=[ReadTool()], work_dir=work_dir)

    agent = Agent(
        name="TestAgent",
        system_prompt="You are a helpful coding assistant.",
        llm=llm,
        toolset=toolset,
        trajectory_file=".fastcontext/test_trajectory.jsonl",
        work_dir=work_dir,
    )

    result = await agent.run(
        "Summarize the content of ./README.md in one sentence.",
        max_turns=20,
        verbose=True,
    )
    assert isinstance(result, str) and result


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_agent())
