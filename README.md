# Podify

`podify <url>` turns a written article into a ~20-minute, single-narrator **MP3 lecture** for
walks. It doesn't read the page aloud — it *teaches* the topic like a professor: it motivates
why the subject matters, defines the jargon the source assumes (via web research), structures
a lesson, and narrates it for the ear.

## Sample

**Listen:** [a 12-second sample](samples/sample.mp3) — the opening of a generated lecture on
designing for high availability. GitHub plays it in its file viewer when you open the link.

## How it works

The pipeline is `fetch -> research -> author -> audio`:

- **fetch** — download the URL and extract clean article text.
- **research** — a small hand-written agent loop (ReAct) the model drives with two tools,
  `web_search` (DuckDuckGo) and `fetch_url`, to gather the definitions, analogies, and
  examples the source assumes; it returns cited teaching notes. Bounded by iteration and
  search guardrails.
- **author** — a single LLM call turns the source plus notes into an audio-first lecture
  script (hook, roadmap, simple-to-advanced body, recaps, close).
- **audio** — Piper narrates the script locally and lameenc encodes the MP3 (no ffmpeg).

LLM calls route through [OpenRouter](https://openrouter.ai) to Claude models — authoring uses
`anthropic/claude-opus-4.7` (quality), the research agent uses `anthropic/claude-sonnet-4.6`
(speed/cost).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env          # then add your OPENROUTER_API_KEY
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
| `--voice` | `en_US-lessac-medium` | Piper voice id |
| `--minutes` | `20` | target length in minutes |
| `--from-stage` | _(all cached)_ | force a recompute from this stage onward: `fetch`, `research`, `author`, `audio` |
| `--review-script` | `true` | pause to read/edit the script before TTS |

Examples:

```bash
podify <url> --from-stage author       # reuse cached fetch + research, re-author onward
podify <url> --review-script false      # skip the human-in-the-loop pause
podify <url> --voice en_US-amy-medium --minutes 15
```

With `--review-script true` (the default), podify pauses after writing the script and prints
its path; edit `runs/<hash>/script.md` if you like, then press Enter to narrate.

## Notes

- **Caching & resume:** each run caches its work in `runs/<url-hash>/state.json` (the fetched
  text, research notes, and script) alongside the `lecture.mp3`. Re-runs reuse the cache;
  `--from-stage` forces a recompute from a given stage onward (earlier stages still load from
  cache) without repaying for their LLM work.
- **Requirements:** Python 3.11+, an OpenRouter API key, and the one-time voice download.
  No ffmpeg — the MP3 is encoded in-process with lameenc.
- **Code map:** `src/__init__.py` (CLI, config, run state, orchestration) plus the four
  stages — `fetch.py`, `agent.py`, `author.py`, `audio.py`.
