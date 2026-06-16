#!/usr/bin/env bash
#
# Verify a FastContext server is up and doing native tool-calling.
#
#   ./serving/verify.sh
#   BASE_URL=http://localhost:8001/v1 MODEL=qwen3-fastcontext-4b-rl ./serving/verify.sh
#
# Two checks: (1) the model is advertised, (2) a tool call is parsed into a
# structured `tool_calls` field (NOT raw <tool_call> text in `content`, which
# means the tool-call parser is wrong and the FastContext client gets nothing).
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen3-fastcontext-4b-rl}"

echo "==> Models advertised at ${BASE_URL}:"
curl -sf "${BASE_URL}/models" | python3 -m json.tool || {
  echo "Server not reachable at ${BASE_URL}" >&2
  exit 1
}

echo
echo "==> Tool-call sanity (expect a populated 'tool_calls' field, not <tool_call> in content):"
curl -sf "${BASE_URL}/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"List files matching *.py\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"glob\",\"description\":\"find files by glob\",\"parameters\":{\"type\":\"object\",\"properties\":{\"pattern\":{\"type\":\"string\"}},\"required\":[\"pattern\"]}}}],\"tool_choice\":\"auto\"}" \
  | python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; print('tool_calls:', m.get('tool_calls')); print('content:', m.get('content'))"
