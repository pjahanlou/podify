# Podify

`podify <url>` turns a written source (e.g. a ByteByteGo article) into a ~20-minute,
single-narrator **MP3 lecture** for walks. It doesn't read the page aloud — it *teaches* it
like a professor: motivates the topic, defines jargon the source assumes (via web research),
structures a lesson, and narrates it for the ear.

This is a **learning project**: build an agent from scratch (Phase A, no framework), then
rebuild the same logic in LangGraph (Phase B). Code is named to map onto canonical agent
concepts — see the terminology map below.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env                     # then add your OPENROUTER_API_KEY

podify https://example.com/some-article  # first run auto-downloads the Piper voice (~60MB) -> voices/
podify <url> --from-stage author         # recompute from this stage on (earlier stay cached)
podify <url> --review-script false       # skip the human-in-the-loop pause
```
MP3 is encoded in-process with lameenc — no ffmpeg needed. TTS is local via Piper, so the
audio step is free and offline; only the research agent + author calls use the network.

## Structure (Phase A)
- `src/__init__.py` — `main()` CLI, config constants, `Run` dataclass (the agent/graph
  **state**), `logging` setup, orchestration, and the unified cache: one `state.json` per run
  under `runs/<url-hash>/` (the `.mp3` sits beside it).
- `src/fetch.py` — `fetch_url()`: download + extract clean text. Also the agent's `fetch_url`
  **client-side tool**.
- `src/agent.py` — `ResearchAgent`: the hand-written **agent loop** (web_search + fetch_url).
- `src/author.py` — `write_script()`: source + notes -> the lecture script.
- `src/audio.py` — Piper synth + stitch -> MP3.

## Pipeline
`fetch -> research (the agent) -> author -> audio`. Mostly a deterministic **workflow** with
one **agentic node** (research). v0 = 2 LLM steps: the research agent, then the author call.

**Backend:** all LLM calls go through **OpenRouter** (OpenAI-compatible SDK) to Claude models
(`anthropic/claude-opus-4.7`, `anthropic/claude-sonnet-4.6`). OpenRouter doesn't expose
Anthropic server tools, so `web_search` is a **client-side** DuckDuckGo tool — meaning both of
the agent's tools run in our own process (a pure ReAct loop).

## Conventions
- **Boolean CLI flags take an explicit value:** `--review-script true|false` — never a
  `--flag/--no-flag` pair.
- **OOP:** a class only where there's real state (the agent). Stateless stages are plain
  functions. Run state is a `@dataclass`.
- **Caching:** every stage shares one `runs/<hash>/state.json` (keyed by the 12-char SHA1 of
  the URL) holding `source_text`, `notes`, `citations`, `script`; the `.mp3` is separate.
  `load_state`/`save_state` make each stage's cache check identical — a stage is skipped when
  its field is already populated; `--from-stage X` clears fields from `X` on to force recompute.
- **Logging, not print:** diagnostics go through `logging` (loggers `podify`, `podify.research`,
  `podify.audio`; configured once in `main()`). Only the interactive HITL prompt uses `print`/`input`.
- **Errors:** stages/helpers raise `PodifyError` (never `sys.exit`); `main()` catches it, logs
  the message, and exits non-zero.
- **Models (via OpenRouter):** author uses `anthropic/claude-opus-4.7` (quality); the research
  agent uses `anthropic/claude-sonnet-4.6` (speed/cost).
- Start minimal; keep enhancements (evaluator node, sectioned generation, orchestrator) as a
  backlog, not upfront work.

## Terminology map (code -> concept)
| Code | Concept |
|---|---|
| `ResearchAgent.run()` | agent loop (ReAct: reason -> act -> observe) |
| `web_search` (DuckDuckGo, we run it) | client-side tool / function calling |
| `fetch_url` (we run it) | client-side tool / function calling |
| `finish_reason` / `tool_calls` handling | the agent control loop / state machine |
| `MAX_AGENT_ITERS`, `MAX_WEB_SEARCHES` | guardrails |
| `--review-script` pause | human-in-the-loop (HITL) |
| `Run` dataclass | agent/graph state |

Full design + the "dial up autonomy later" ladder live in the approved plan at
`~/.claude/plans/i-want-us-to-toasty-stallman.md`.
