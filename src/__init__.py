"""podify — turn a URL into a ~20-minute single-narrator MP3 lecture.

Mostly a deterministic workflow (fetch -> research -> author -> audio) with one agentic
node: the research agent (see agent.py). `main()` is the CLI entry point.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
MAX_WEB_SEARCHES = 5                   # guardrail: cap server-side searches
MIN_SOURCE_CHARS = 500                 # guardrail: refuse near-empty extractions

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"

STAGES = ["fetch", "research", "author", "audio"]


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


# --- shared Anthropic client (lazy, so fetch-only runs need no API key) ------
_client = None


def client():
    """OpenAI-compatible client pointed at OpenRouter (so we can call Claude models)."""
    global _client
    if _client is None:
        import os

        from openai import OpenAI

        key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("Set OPENROUTER_API_KEY in .env")
        _client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=key,
            default_headers={"X-Title": "podify"},
        )
    return _client


def workdir_for(url: str) -> Path:
    digest = hashlib.sha1(url.encode()).hexdigest()[:12]
    return RUNS_DIR / digest


# --- stages -----------------------------------------------------------------
def do_fetch(run: Run) -> None:
    from .fetch import fetch_url

    cache = run.workdir / "extracted.txt"
    if cache.exists():
        run.source_text = cache.read_text()
        print(f"[fetch] cached -> {cache} ({len(run.source_text)} chars)")
        return
    print(f"[fetch] downloading {run.url}")
    text = fetch_url(run.url)
    if len(text) < MIN_SOURCE_CHARS:
        sys.exit(
            f"[fetch] extracted only {len(text)} chars; the page may be paywalled "
            f"or JS-rendered. Aborting."
        )
    run.source_text = text
    cache.write_text(text)
    print(f"[fetch] extracted {len(text)} chars -> {cache}")


def do_research(run: Run) -> None:
    from .agent import ResearchAgent

    cache = run.workdir / "notes.json"
    if cache.exists():
        data = json.loads(cache.read_text())
        run.notes, run.citations = data["notes"], data.get("citations", [])
        print(f"[research] cached -> {cache} ({len(run.notes)} chars)")
        return
    print("[research] starting agent loop...")
    result = ResearchAgent().run(run.source_text)
    run.notes, run.citations = result.notes, result.citations
    cache.write_text(
        json.dumps({"notes": run.notes, "citations": run.citations}, indent=2)
    )
    print(
        f"[research] notes ({len(run.notes)} chars), "
        f"{len(run.citations)} citations -> {cache}"
    )


def do_author(run: Run) -> None:
    from .author import write_script

    cache = run.workdir / "script.md"
    if cache.exists():
        run.script = cache.read_text()
        print(f"[author] cached -> {cache} ({len(run.script.split())} words)")
        return
    print("[author] writing script...")
    run.script = write_script(run.source_text, run.notes, run.target_minutes)
    cache.write_text(run.script)
    print(f"[author] script ({len(run.script.split())} words) -> {cache}")


def _hitl_pause(run: Run) -> None:
    """Human-in-the-loop: let the user read/edit the script before paying for TTS."""
    path = run.workdir / "script.md"
    print(f"\n[review] Script ready: {path}")
    print("[review] Read or edit it now. Press Enter to narrate, or Ctrl-C to abort.")
    try:
        input()
    except EOFError:
        print("[review] non-interactive stdin; continuing.")
    if path.exists():  # pick up any edits the user made
        run.script = path.read_text()


def do_audio(run: Run, review_script: bool) -> None:
    from .audio import synthesize

    out = run.out_path or (run.workdir / "lecture.mp3")
    if review_script:
        _hitl_pause(run)
    print("[audio] synthesizing with Piper...")
    run.audio_path = synthesize(run.script, run.voice, out)
    print(f"[audio] done -> {run.audio_path}")


def run_pipeline(run: Run, from_stage: str, review_script: bool) -> None:
    start = STAGES.index(from_stage)
    if start <= 0:
        do_fetch(run)
    if start <= 1:
        do_research(run)
    if start <= 2:
        do_author(run)
    if start <= 3:
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
        default="fetch",
        help="resume the pipeline from a cached stage",
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
    (run.workdir / "segments").mkdir(parents=True, exist_ok=True)
    print(f"[podify] workdir: {run.workdir}")
    run_pipeline(run, from_stage=args.from_stage, review_script=args.review_script)
