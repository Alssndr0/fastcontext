import json
from pathlib import Path

import aiofiles

from .tool import Tool, resolve_in_workspace

MAX_LINE = 2000
MAX_LINE_LENGTH = 2000


class ReadTool(Tool):
    name = "Read"
    description: str = Tool.load_desc(Path(__file__).parent / "read.md")
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The path of the file to read -- absolute, or relative to the workspace root.",
            },
            "offset": {
                "type": "integer",
                "description": "The line number to start reading from. Positive values are 1-indexed from the start of the file. Negative values count backwards from the end (e.g. -1 is the last line). Only provide if the file is too large to read at once.",
            },
            "limit": {
                "type": "integer",
                "description": "The number of lines to read. Only provide if the file is too large to read at once.",
            },
        },
        "required": ["path"],
    }

    async def call(self, parameters: str, **kwargs) -> str:
        params: dict = json.loads(parameters)
        file_path = params.get("path")
        offset = params.get("offset")
        limit = params.get("limit")

        if not file_path:
            return "Read Tool: file path is required."

        cwd = kwargs.get("cwd", Path.cwd().as_posix())
        resolved = resolve_in_workspace(file_path, cwd)
        if resolved is None:
            return f"Permission error: `{file_path}` is not within the working directory `{cwd}`."
        if resolved.is_dir():
            return f"Read Tool: `{file_path}` is a directory, not a file."
        if not resolved.exists():
            return f"Read Tool: file {file_path} does not exist."
        file_path = str(resolved)

        async with aiofiles.open(file_path, mode="r") as f:
            raw_lines = await f.readlines()

        if len(raw_lines) == 0:
            return "File is empty."

        end_line = -1
        if offset is None or offset < 0:
            offset = 1
        if limit is not None:
            end_line = offset + limit - 1
        if end_line == -1 or end_line > len(raw_lines):
            end_line = len(raw_lines)

        lines = []
        total_read_lines = end_line - offset + 1
        if total_read_lines > MAX_LINE:
            end_line = offset + MAX_LINE - 1
        for i in range(offset - 1, end_line):
            if len(raw_lines[i]) > MAX_LINE_LENGTH:
                line = raw_lines[i][:MAX_LINE_LENGTH] + "...\n"
            else:
                line = raw_lines[i]
            prefixed_line = f"{i+1}|{line}"
            lines.append(prefixed_line)
        if total_read_lines > MAX_LINE:
            lines.append("...")
        content = "".join(lines)
        output = f"```{file_path}:{offset}-{end_line}\n{content}\n```"
        return output
