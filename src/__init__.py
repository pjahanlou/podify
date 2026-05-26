"""podify — turn a URL into a ~20-minute single-narrator MP3 lecture.

Mostly a deterministic workflow (fetch -> research -> author -> audio) with one agentic
node: the research agent (see agent.py). `main()` is the CLI entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

# --- config -----------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_AUTHOR = "anthropic/claude-opus-4.7"      # authoring wants quality
MODEL_RESEARCH = "anthropic/claude-sonnet-4.6"  # the research agent wants speed/cost
TARGET_MINUTES = 20
WORDS_PER_MINUTE = 135                 # relaxed teacher pace
DEFAULT_VOICE = "en_US-lessac-medium"
MAX_AGENT_ITERS = 6                    # guardrail: cap the agent loop
MAX_WEB_SEARCHES = 5                   # guardrail: cap client-side searches
MIN_SOURCE_CHARS = 500                 # guardrail: refuse near-empty extractions

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

STAGES = ["fetch", "research", "author", "audio"]
# which Run fields each stage produces; cached in state.json (audio writes only the mp3)
STAGE_FIELDS = {
    "fetch": ["source_text"],
    "research": ["notes", "citations"],
    "author": ["script"],
    "audio": [],
}
CACHE_FIELDS = ["source_text", "notes", "citations", "script"]

log = logging.getLogger("podify")


class PodifyError(Exception):
    """A user-facing error; main() reports it cleanly and exits non-zero."""


# --- run state (becomes the LangGraph State in Phase B) ---------------------
@dataclass
class Run:
    url: str
    workdir: Path
    voice: str = DEFAULT_VOICE
    target_minutes: int = TARGET_MINUTES
    out_path: Path | None = None
    source_text: str = ""
    notes: str = ""
    citations: list = field(default_factory=list)
    script: str = ""
    audio_path: Path | None = None


# --- shared client (lazy, so fetch-only runs need no API key) ---------------
_client = None


def client():
    """OpenAI-compatible client pointed at OpenRouter (so we can call Claude models)."""
    global _client
    if _client is None:
        import os

        from openai import OpenAI

        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise PodifyError("Set OPENROUTER_API_KEY in .env")
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers={"X-Title": "podify"},
        )
    return _client


def workdir_for(url: str) -> Path:
    digest = hashlib.sha1(url.encode()).hexdigest()[:12]
    return RUNS_DIR / digest


# --- unified cache: one state.json per run, appended by each stage -----------
def _state_path(run: Run) -> Path:
    return run.workdir / "state.json"


def load_state(run: Run) -> None:
    """Populate cached fields onto `run` from state.json, if it exists."""
    path = _state_path(run)
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for key in CACHE_FIELDS:
        if key in data:
            setattr(run, key, data[key])
    log.info("loaded cached state -> %s", path)


def save_state(run: Run) -> None:
    data = {"url": run.url, **{key: getattr(run, key) for key in CACHE_FIELDS}}
    _state_path(run).write_text(json.dumps(data, indent=2))


def _invalidate_from(run: Run, from_stage: str) -> None:
    """Clear cached fields for `from_stage` and later so those stages recompute."""
    for stage in STAGES[STAGES.index(from_stage):]:
        for f in STAGE_FIELDS[stage]:
            setattr(run, f, [] if isinstance(getattr(run, f), list) else "")


# --- stages -----------------------------------------------------------------
def do_fetch(run: Run) -> None:
    from .fetch import fetch_url

    if run.source_text:
        log.info("fetch: cached (%d chars)", len(run.source_text))
        return
    log.info("fetch: downloading %s", run.url)
    text = fetch_url(run.url)
    if len(text) < MIN_SOURCE_CHARS:
        raise PodifyError(
            f"fetch: extracted only {len(text)} chars; the page may be paywalled or "
            "JS-rendered."
        )
    run.source_text = text
    save_state(run)
    log.info("fetch: extracted %d chars", len(text))


def do_research(run: Run) -> None:
    from .agent import ResearchAgent

    if run.notes:
        log.info("research: cached (%d chars, %d citations)", len(run.notes), len(run.citations))
        return
    log.info("research: starting agent loop...")
    result = ResearchAgent().run(run.source_text)
    run.notes, run.citations = result.notes, result.citations
    save_state(run)
    log.info("research: notes %d chars, %d citations", len(run.notes), len(run.citations))


def do_author(run: Run) -> None:
    from .author import write_script

    if run.script:
        log.info("author: cached (%d words)", len(run.script.split()))
        return
    log.info("author: writing script...")
    run.script = write_script(run.source_text, run.notes, run.target_minutes)
    save_state(run)
    log.info("author: script %d words", len(run.script.split()))


def _hitl_pause(run: Run) -> None:
    """Human-in-the-loop: write the script out so the user can read/edit it before TTS."""
    path = run.workdir / "script.md"
    path.write_text(run.script)
    print(f"\n[review] Script ready: {path}")
    print("[review] Read or edit it now. Press Enter to narrate, or Ctrl-C to abort.")
    try:
        input()
    except EOFError:
        log.info("review: non-interactive stdin; continuing.")
    edited = path.read_text()
    if edited != run.script:  # pick up and re-cache any edits the user made
        run.script = edited
        save_state(run)
        log.info("review: picked up edited script (%d words)", len(run.script.split()))


def do_audio(run: Run, review_script: bool) -> None:
    from .audio import synthesize

    out = run.out_path or (run.workdir / "lecture.mp3")
    if review_script:
        _hitl_pause(run)
    log.info("audio: synthesizing with Piper...")
    run.audio_path = synthesize(run.script, run.voice, out)
    log.info("audio: done -> %s", run.audio_path)


def run_pipeline(run: Run, from_stage: str | None, review_script: bool) -> None:
    load_state(run)
    if from_stage:
        _invalidate_from(run, from_stage)
    do_fetch(run)
    do_research(run)
    do_author(run)
    do_audio(run, review_script)


# --- CLI --------------------------------------------------------------------
def _str2bool(v: str) -> bool:
    low = v.lower()
    if low in ("true", "t", "1", "yes", "y"):
        return True
    if low in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        prog="podify", description="Turn a URL into a ~20-minute MP3 lecture."
    )
    p.add_argument("url", help="source article URL")
    p.add_argument("--out", type=Path, default=None, help="output mp3 path")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="Piper voice id")
    p.add_argument(
        "--minutes", type=int, default=TARGET_MINUTES, help="target length in minutes"
    )
    p.add_argument(
        "--from-stage",
        choices=STAGES,
        default=None,
        help="force re-running from this stage onward (default: use all cached stages)",
    )
    p.add_argument(
        "--review-script",
        type=_str2bool,
        default=True,
        metavar="true|false",
        help="pause to review/edit the script before TTS (default true)",
    )
    args = p.parse_args(argv)

    run = Run(
        url=args.url,
        workdir=workdir_for(args.url),
        voice=args.voice,
        target_minutes=args.minutes,
        out_path=args.out,
    )
    run.workdir.mkdir(parents=True, exist_ok=True)
    log.info("workdir: %s", run.workdir)
    try:
        run_pipeline(run, from_stage=args.from_stage, review_script=args.review_script)
    except PodifyError as e:
        log.error("%s", e)
        raise SystemExit(1)
