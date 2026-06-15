from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool


async def test_toolset():
    toolset = ToolSet(tools=[ReadTool()], work_dir=".")
    schema_list = toolset.schema_list()
    assert len(schema_list) == 1
    assert schema_list[0]["function"]["name"] == "Read"

    tool_call_msg = Message(
        role="assistant",
        content=None,
        tool_call_id="call_1",
        tool_calls=[
            FunctionCall(
                id="call_1_1",
                name="Read",
                arguments='{"path": "./README.md", "offset": 1, "limit": 20}',
            ),
            FunctionCall(
                id="call_1_2",
                name="Read",
                arguments='{"path": "./pyproject.toml", "offset": 1, "limit": 20}',
            ),
        ],
    )
    tools_result_messages = await toolset.call(tool_call_msg)
    assert len(tools_result_messages) == 2
    for msg in tools_result_messages:
        assert msg.role == "tool"
        assert msg.content


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_toolset())
