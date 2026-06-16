"""Offline unit tests — no server or external tools required."""

from pathlib import Path

from fastcontext.agent.llm import FunctionCall, Message
from fastcontext.agent.tool import ToolSet
from fastcontext.agent.tool.read import ReadTool
from fastcontext.agent.tool.tool import resolve_in_workspace
from fastcontext.agent.utils import find_repo_root, get_final_answer


def test_get_final_answer_extracts_block():
    text = "Summary of findings.\n<final_answer>\n/a/b.py:1-5\n</final_answer>"
    assert get_final_answer(text) == "/a/b.py:1-5"


def test_get_final_answer_returns_text_when_no_block():
    text = "no final answer tags here"
    assert get_final_answer(text) == text


def test_get_final_answer_strips_work_dir_prefix():
    text = "<final_answer>\n/home/me/repo/src/a.py:1-5 (entry)\n/home/me/repo/tests/t.py:9-12\n</final_answer>"
    assert get_final_answer(text, "/home/me/repo") == "src/a.py:1-5 (entry)\ntests/t.py:9-12"


def test_get_final_answer_anchors_to_git_root_from_subdir(tmp_path):
    (tmp_path / ".git").mkdir()
    subdir = tmp_path / "src" / "pkg"
    subdir.mkdir(parents=True)
    text = f"<final_answer>\n{tmp_path}/src/a.py:1-5\n</final_answer>"
    # Invoked from a subdirectory, citations stay relative to the repo root.
    assert get_final_answer(text, str(subdir)) == "src/a.py:1-5"


def test_find_repo_root_falls_back_to_path_when_no_git(tmp_path):
    assert find_repo_root(str(tmp_path)) == str(tmp_path.resolve())


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


def test_resolve_in_workspace_accepts_absolute_inside(tmp_path):
    (tmp_path / "src").mkdir()
    assert resolve_in_workspace(str(tmp_path / "src"), str(tmp_path)) == (tmp_path / "src")


def test_resolve_in_workspace_accepts_relative(tmp_path):
    (tmp_path / "src").mkdir()
    # Relative paths anchor to the workspace, not the process CWD.
    assert resolve_in_workspace("src", str(tmp_path)) == (tmp_path / "src")


def test_resolve_in_workspace_rebases_dropped_prefix(tmp_path):
    # The explorer model drops the parent prefix: `/<work.name>/src` for a
    # workspace at `<tmp_path>/<work.name>`. This is the bug seen in the logs.
    work = tmp_path / "fastcontext"
    (work / "src").mkdir(parents=True)
    rebased = resolve_in_workspace(f"/{work.name}/src", str(work))
    assert rebased == (work / "src")
    assert resolve_in_workspace(f"/{work.name}", str(work)) == work


def test_resolve_in_workspace_rejects_escape(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    assert resolve_in_workspace("/etc/passwd", str(work)) is None
    # `..` traversal that leaves the workspace is rejected.
    assert resolve_in_workspace("../secret", str(work)) is None
