# Integrating FastContext into your coding agent

FastContext is a command-line codebase-exploration subagent. Your main coding agent calls it
to find where relevant code lives in an unfamiliar repository, and gets back compact
`file:line` citations — without spending its own context window on broad searches and reads.

Invoke it from inside the repository you are working in:

```bash
fastcontext -q "<detailed natural-language description of what to find>" --citation
```

It runs a short, read-only, multi-step exploration internally (Read/Glob/Grep only — it never
edits files) and returns a single message: a one- or two-sentence summary plus a
`<final_answer>` block of file paths with line ranges.

## When to call FastContext

Call it when you are cold-starting on a codebase and need to discover *where* something is:

- You need to find where a feature, behavior, symbol, route, or config lives, and you don't
  already know the file.
- You need a structured listing of related definitions or call sites spread across many files.
- A direct `grep`/`rg` returned nothing useful and you're unsure where to look next.

Skip it — just use your own tools — when:

- You already know the file path or symbol you need.
- A previous turn already returned a location that covers what you need.
- You only need to read one specific file you've already identified.
- You're searching within 2–3 known files for a specific definition.

## Writing a good query

The query is a prompt to an autonomous agent, so be specific and self-contained. Name the
behavior, error message, endpoint, data flow, or subsystem — not just a bare keyword. Ask one
focused question per call; make two calls for two unrelated things.

- Good: `fastcontext -q "find where incoming webhook payloads are validated and where the HMAC signature is checked" --citation`
- Weak: `fastcontext -q "webhooks" --citation`

## Consuming the result

- **Trust the listing.** After FastContext returns, go straight to reading the cited ranges
  with your own file-read tool. Do **not** re-run broad repository-wide searches (`grep -R`,
  `find . -name`) for the same information — that defeats the purpose.
- **Read narrowly.** Open the 1–2 ranges most relevant to your task first, not every line
  listed.
- **If incomplete or off-target,** your cheapest next move is to call `fastcontext` again with
  a sharper query — re-asking is faster than scanning the repo yourself. Use a narrow
  `grep -n "<symbol>" path/to/known/file` only once you already know the exact file.
- **Citations are evidence, not a verdict.** You still verify by reading, then do the actual
  engineering (edit, test, explain) yourself.

## Usage in practice

Query FastContext repeatedly, not once. One broad query rarely covers a feature that spans
layers; sharpen each follow-up using what the last one revealed:

1. **Start broad** for a beachhead — the files most obviously tied to the request.
2. **Pivot to the code's own names.** Results reveal the terms the code uses; if you asked about
   a "rate watch" but see `RateMonitor` / `/api/monitors`, re-ask with those — the explorer
   won't switch synonyms itself.
3. **Cover by layer.** Ask for the API endpoint, service logic, data model / DB table, background
   job, and delivery path explicitly; one query rarely returns all layers.
4. **Trace call chains yourself.** It won't reliably follow imports or "who calls this" — when a
   piece is missing, ask directly (e.g. "what calls `check_and_fire_monitors`, and where is it
   scheduled").

Citations are accurate even when the set is incomplete, so a partial answer is safe to build on
and "nothing found" is a real signal to re-ask.

## Drop-in system-prompt snippet

Paste the block below into your agent's system prompt.

<!-- BEGIN: fastcontext integration snippet -->
You have `fastcontext`, a fast read-only codebase-exploration subagent run as a shell command:
  fastcontext -q "<detailed description of what to find>" --citation
It returns a brief summary plus a <final_answer> block of path:line-range citations.

Use it to locate a feature/symbol/config you can't place, to map definitions or call sites
across many files, or when a direct grep found nothing. Skip it when you already know the
file(s) or a prior turn gave the location.

After it returns, read the cited ranges directly — don't repeat broad repo-wide searches for
the same thing. Citations are evidence: verify by reading, then make the change yourself.

One call is a starting point, not a full map. It anchors on your wording and won't expand
synonyms or follow imports on its own, so when a result is incomplete, sharpen the next call
instead of rephrasing: switch to the names the code uses (it may say `monitor` where you said
`watch`), ask for a specific missing layer (model, scheduler, delivery), or trace callers
directly ("what calls X, and where is it scheduled").
<!-- END: fastcontext integration snippet -->
