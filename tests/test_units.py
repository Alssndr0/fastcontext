"""Offline unit tests — no server or external tools required."""

from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool
from fastcontext.agent.utils import get_final_answer


def test_get_final_answer_extracts_block():
    text = "Summary of findings.\n<final_answer>\n/a/b.py:1-5\n</final_answer>"
    assert get_final_answer(text) == "<final_answer>\n/a/b.py:1-5\n</final_answer>"


def test_get_final_answer_returns_text_when_no_block():
    text = "no final answer tags here"
    assert get_final_answer(text) == text


def test_function_call_serialization():
    fc = FunctionCall(id="call_1", name="Read", arguments='{"path": "x"}')
    data = fc.model_dump()
    assert data["type"] == "function"
    assert data["id"] == "call_1"
    assert data["function"]["name"] == "Read"
    assert data["function"]["arguments"] == '{"path": "x"}'


def test_message_to_dict_excludes_none():
    msg = Message(role="user", content="hi")
    assert msg.to_dict() == {"role": "user", "content": "hi"}


def test_toolset_schema_list():
    toolset = ToolSet(tools=[ReadTool()], work_dir=".")
    schemas = toolset.schema_list()
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "Read"
