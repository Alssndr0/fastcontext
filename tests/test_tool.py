import asyncio
import json
import os
import shutil

import pytest

from fastcontext.agent.tool.glob import GlobTool
from fastcontext.agent.tool.grep import GrepTool

ripgrep_required = pytest.mark.skipif(
    shutil.which("rg") is None and not os.getenv("FASTCONTEXT_RG"),
    reason="ripgrep (rg) is not installed",
)


@ripgrep_required
def test_grep_tool():
    grep = GrepTool()
    params = {
        "pattern": "GrepTool",
        "path": ".",
        "glob": "*.py",
        "output_mode": "content",
        "head_limit": 100,
        "-C": 3,
    }
    output = asyncio.run(grep.call(json.dumps(params)))
    assert isinstance(output, str)
    assert "GrepTool" in output


@ripgrep_required
def test_glob_tool():
    glob = GlobTool()
    params = {
        "directory": "./src",
        "pattern": "**/*.py",
    }
    output = asyncio.run(glob.call(json.dumps(params)))
    assert isinstance(output, str)
    assert ".py" in output


if __name__ == "__main__":
    test_grep_tool()
    test_glob_tool()
