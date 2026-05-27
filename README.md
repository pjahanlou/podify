# Podify

`podify <url>` turns a written article into a ~20-minute, single-narrator **MP3 lecture** for
walks. It doesn't read the page aloud — it *teaches* the topic like a professor: it motivates
why the subject matters, defines the jargon the source assumes (via web research), structures
a lesson, and narrates it for the ear.

## Sample

**Listen:** [a sample](samples/sample.mp3) — the opening of a generated lecture on how Netflix
uses multimodal AI. GitHub plays it in its file viewer when you open the link.

## How it works

The pipeline is `fetch -> research -> author -> audio`:

- **fetch** — download the URL and extract clean article text via tiered fallback (trafilatura
  → BeautifulSoup → SPA preload JSON → Jina Reader → largest block → HITL paste).
- **research** — a **plan-and-execute** agent: one Opus call identifies 3–7 knowledge gaps
  the listener needs, then a Sonnet ReAct loop fills them using three client-side tools —
  `web_search` (DuckDuckGo), `fetch_url`, and `update_progress` (records per-gap coverage).
  Stops when all gaps are covered, the model declares done, stagnation is detected, or the
  hard iteration cap fires.
- **author** — a single LLM call turns the source plus notes into an audio-first lecture
  script (hook, roadmap, simple-to-advanced body, recaps, close).
- **audio** — narrates the script and encodes an MP3 with lameenc (no ffmpeg). Two backends:
  Piper (local, offline, free) or OpenAI TTS (`--tts openai`, routes to `api.openai.com`).

LLM calls route through [OpenRouter](https://openrouter.ai) to Claude models — authoring uses
`anthropic/claude-opus-4.7` (quality), the research agent uses `anthropic/claude-sonnet-4.6`
(speed/cost).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # add OPENROUTER_API_KEY; add OPENAI_API_KEY if using --tts openai
```

The first run auto-downloads the Piper voice (~60MB) into `voices/`.

## Usage

```bash
podify https://example.com/some-article
```

| Flag | Default | Meaning |
|---|---|---|
| `url` | — | source article URL (positional) |
| `--out` | `runs/<hash>/lecture.mp3` | output MP3 path |
| `--tts` | `piper` | TTS backend: `piper` (local, offline) or `openai` (`gpt-4o-mini-tts`, requires `OPENAI_API_KEY`) |
| `--voice` | `en_US-lessac-medium` / `onyx` | voice id — Piper voice name, or OpenAI voice (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`) |
| `--minutes` | `20` | target length in minutes |
| `--from-stage` | _(all cached)_ | force a recompute from this stage onward: `fetch`, `research`, `author`, `audio` |
| `--review-script` | `true` | pause to read/edit the script before TTS |

Examples:

```bash
podify <url> --from-stage author          # reuse cached fetch + research, re-author onward
podify <url> --review-script false        # skip the human-in-the-loop pause
podify <url> --tts openai                 # OpenAI TTS (requires OPENAI_API_KEY in .env)
podify <url> --tts openai --voice alloy   # different voice
podify <url> --voice en_US-amy-medium --minutes 15
```

With `--review-script true` (the default), podify pauses after writing the script and prints
its path; edit `runs/<hash>/script.md` if you like, then press Enter to narrate.

## Notes

- **Caching & resume:** each run caches its work in `runs/<url-hash>/state.json` (the fetched
  text, research notes, and script) alongside the `lecture.mp3`. Re-runs reuse the cache;
  `--from-stage` forces a recompute from a given stage onward (earlier stages still load from
  cache) without repaying for their LLM work.
- **Requirements:** Python 3.11+, an `OPENROUTER_API_KEY`, and the one-time Piper voice
  download. `--tts openai` additionally requires an `OPENAI_API_KEY`. No ffmpeg — the MP3 is
  encoded in-process with lameenc.
- **Code map:** `src/__init__.py` (CLI, config, run state, orchestration) plus the four
  stages — `fetch.py`, `agent.py`, `author.py`, `audio.py`.
