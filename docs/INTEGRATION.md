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

## Drop-in system-prompt snippet

Paste the block below into your agent's system prompt.

<!-- BEGIN: fastcontext integration snippet -->
You have access to `fastcontext`, a fast read-only codebase-exploration subagent available as
a shell command. Use it to discover where relevant code lives in an unfamiliar repository.

Invoke it as: fastcontext -q "<detailed description of what to find>" --citation
It returns a short summary plus a <final_answer> block of path:line-range citations.

Use fastcontext when you need to locate a feature/symbol/config you can't already place, when
you need related definitions or call sites across many files, or when a direct grep found
nothing. Skip it when you already know the file, a prior turn gave the location, or you only
need one known file or 2–3 known files.

After it returns: trust the listing and read the cited ranges directly with your own tools —
do not repeat broad repo-wide searches for the same thing. Read the most relevant 1–2 ranges
first. If incomplete, call fastcontext again with a sharper query rather than scanning the repo
yourself. Treat its citations as evidence; verify by reading, then make the change yourself.
<!-- END: fastcontext integration snippet -->
