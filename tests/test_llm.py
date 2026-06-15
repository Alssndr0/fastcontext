import os

import pytest

from fastcontext.agent.llm import LLM

live_server_required = pytest.mark.skipif(
    not os.getenv("BASE_URL"),
    reason="requires a live OpenAI-compatible endpoint (set BASE_URL / MODEL / API_KEY)",
)


@live_server_required
async def test_llm():
    llm = LLM(
        model=os.getenv("MODEL"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    msg = await llm.acall(messages=messages, tools=None)
    assert msg.content is not None


@live_server_required
async def test_llm_tools():
    from fastcontext.agent.tool.read import ReadTool

    llm = LLM(
        model=os.getenv("MODEL"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.0,
        max_tokens=1024,
    )
    messages = [
        {"role": "system", "content": "You are a powerful AI agent."},
        {"role": "user", "content": "read file content from ./README.md"},
    ]
    msg = await llm.acall(messages=messages, tools=[ReadTool().schema()])
    assert msg.tool_calls


@live_server_required
async def test_llm_tools_result():
    llm = LLM(
        model=os.getenv("MODEL"),
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.0,
        max_tokens=1024,
    )
    messages = [
        {"role": "system", "content": "You are a powerful AI agent."},
        {"role": "user", "content": "please show me the current time"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_0",
                    "function": {"arguments": '{"command": "date"}', "name": "bash"},
                    "type": "function",
                },
            ],
        },
        {
            "role": "tool",
            "content": "Thu Aug 21 17:42:44 CST 2025",
            "tool_call_id": "call_0",
        },
    ]
    msg = await llm.acall(messages=messages, tools=None)
    assert msg.content is not None


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_llm())
