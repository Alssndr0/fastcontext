import asyncio
import json
from asyncio import Future
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from fastcontext.agent.llm import Message

MAX_TOOLRUN_TIMEOUT = 10


def resolve_in_workspace(path: str, work_dir: str) -> Path | None:
    """Resolve a model-supplied path to an absolute path inside the workspace.

    The explorer is sandboxed to ``work_dir``. This normalises the path the model
    passed and returns it as an absolute ``Path`` when it lands inside the
    workspace, or ``None`` when it genuinely escapes (so callers raise a
    permission error). It is lenient about two things the small explorer model
    routinely gets wrong:

    - Relative paths are resolved against the workspace root, not the process
      CWD (which ``Path.resolve()`` would otherwise use).
    - The model often drops the parent prefix and passes a workspace-root-relative
      absolute path -- e.g. ``/fastcontext/src`` for a workspace at
      ``/home/user/fastcontext``. We rebase a leading ``/<workspace-name>/...``
      onto the real workspace before giving up.

    ``..`` traversal that escapes the workspace is still rejected.
    """
    work = Path(work_dir).resolve()
    raw = Path(path)
    candidate = (raw if raw.is_absolute() else work / raw).resolve()
    if candidate.is_relative_to(work):
        return candidate

    # Model dropped the parent prefix: rebase `/<work.name>/...` onto the workspace.
    parts = candidate.parts[1:]  # strip the leading "/"
    if parts and parts[0] == work.name:
        rebased = (work.parent / Path(*parts)).resolve()
        if rebased.is_relative_to(work):
            return rebased

    return None


class ToolResult(BaseModel):
    tool_call_id: str
    output: str
    failed: bool


ToolResultFuture = Future[ToolResult]

type ToolOutput = ToolResult | ToolResultFuture


class Tool:
    name: str
    description: str
    parameters: dict[str, Any]

    async def call(self, parameters: str, **kwargs) -> str:
        raise NotImplementedError("Tool.call must be implemented by subclasses.")

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @staticmethod
    def load_desc(path: str) -> str:
        desc = Path(path).read_text(encoding="utf-8")
        return desc


class ToolSet:
    _tool_dict: dict[str, Tool] = {}

    def __init__(self, tools: list[Tool], work_dir: str):
        self._tool_dict = {tool.name: tool for tool in tools}
        self.work_dir = work_dir

    def schema_list(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tool_dict.values()]

    async def _single_tool_call(self, tool_name: str, parameters: str, toll_call_id: str) -> ToolOutput:
        if tool_name not in self._tool_dict:
            return ToolResult(
                tool_call_id=toll_call_id,
                failed=True,
                output=f"Tool `{tool_name}` not found.",
            )

        tool = self._tool_dict[tool_name]
        try:
            json.loads(parameters or "{}")
        except json.JSONDecodeError:
            return ToolResult(
                tool_call_id=toll_call_id,
                failed=True,
                output=f"Tool `{tool_name}` arguments are invalid.",
            )

        async def _call():
            try:
                output = await tool.call(parameters, cwd=self.work_dir)
                return ToolResult(tool_call_id=toll_call_id, failed=False, output=output)
            except Exception as e:
                return ToolResult(tool_call_id=toll_call_id, failed=True, output=str(e))

        # return asyncio.create_task(_call())
        return await _call()

    async def call(self, msg: Message) -> list[Message]:
        if not msg.tool_calls:
            return []

        tool_results: list[ToolResult] = []
        for c in msg.tool_calls:
            try:
                result = await asyncio.wait_for(
                    self._single_tool_call(c.name, c.arguments, c.id), timeout=MAX_TOOLRUN_TIMEOUT
                )
            except TimeoutError:
                result = ToolResult(
                    tool_call_id=c.id, failed=True, output=f"Tool `{c.name}` timed out after {MAX_TOOLRUN_TIMEOUT}s."
                )
            tool_results.append(result)

        tools_result_messages = []
        for tr in tool_results:
            tools_result_messages.append(
                Message(
                    role="tool",
                    content=tr.output,
                    tool_call_id=tr.tool_call_id,
                )
            )
        return tools_result_messages
