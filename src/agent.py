"""The research agent — plan-and-execute with structured progress tracking.

Architecture: three phases inside ResearchAgent.run():

  1. plan  — one Opus call identifies 3-7 knowledge gaps the listener needs filled
  2. loop  — Sonnet ReAct loop works through the gaps with web_search + fetch_url +
             update_progress (the third client-side tool that records coverage)
  3. notes — loop exits when all gaps are covered (primary), model declares done,
             stagnation is detected, or the hard iteration cap fires (guardrail)

All three tools are client-side: we execute them and feed results back.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from . import MAX_AGENT_ITERS, MAX_STAGNANT_TURNS, MAX_WEB_SEARCHES, MODEL_PLANNER, MODEL_RESEARCH, client
from .fetch import fetch_url

_MAX_FETCH_CHARS = 6000
_SEARCH_RESULTS = 5

log = logging.getLogger("podify.research")


# --- data model -------------------------------------------------------------
@dataclass
class Gap:
    id: str        # short slug, e.g. "rag"
    concept: str   # plain-English label, e.g. "Retrieval-Augmented Generation"
    why: str       # why a listener needs this
    status: str = "open"   # "open" | "covered"


# --- tool definitions -------------------------------------------------------
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web. Returns result titles, URLs, and snippets. Use to find "
            "definitions, analogies, examples, and context for a knowledge gap."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch the full readable text of a page. Use after web_search when a snippet "
            "isn't enough to fill a gap."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
}

UPDATE_PROGRESS_TOOL = {
    "type": "function",
    "function": {
        "name": "update_progress",
        "description": (
            "Record progress on a knowledge gap after web_search or fetch_url yields useful "
            "info. Mark 'covered' once you have a clear definition, an analogy, and an example. "
            "Mark 'open' if you found something useful but still need more. "
            "Call once per gap you advanced this turn; skip if the turn yielded nothing new."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gap_id": {"type": "string", "description": "The gap id slug, e.g. 'rag'"},
                "status": {"type": "string", "enum": ["open", "covered"]},
                "what_you_learned": {
                    "type": "string",
                    "description": "1-2 sentence summary of what was learned this turn.",
                },
            },
            "required": ["gap_id", "status", "what_you_learned"],
        },
    },
}


# --- system prompt ----------------------------------------------------------
def _build_system(gaps: dict[str, Gap]) -> str:
    gap_lines = "\n".join(
        f"  [{g.id}] {g.concept} — {g.why}" for g in gaps.values()
    )
    return (
        "You are a research assistant preparing material for an AUDIO lecture — a ~20-minute "
        "spoken lesson for someone listening on a walk.\n\n"
        "Your research agenda is the list of knowledge gaps below. Each gap needs a clear "
        "definition, an intuitive analogy, and at least one concrete real-world example.\n\n"
        f"KNOWLEDGE GAPS TO COVER:\n{gap_lines}\n\n"
        "HOW TO WORK:\n"
        "1. Pick the most important uncovered gap.\n"
        "2. Use web_search to find good sources, then fetch_url to read the best one.\n"
        "3. Call update_progress for each gap you advanced this turn.\n"
        "4. Repeat until all gaps are covered.\n"
        "5. When all gaps are covered, write TEACHING NOTES as your final message "
        "(no more tool calls). For each gap: plain-English explanation, analogy, example. "
        "Group related points.\n\n"
        "If you cannot find good info on a gap after one search, mark it covered with what "
        "you have and move on — don't loop on it."
    )


# --- result -----------------------------------------------------------------
@dataclass
class ResearchResult:
    notes: str
    citations: list = field(default_factory=list)
    gaps: list = field(default_factory=list)   # final Gap list with statuses


# --- agent ------------------------------------------------------------------
class ResearchAgent:
    """Plan-and-execute research agent with explicit gap tracking."""

    def __init__(
        self,
        model: str = MODEL_RESEARCH,
        max_iters: int = MAX_AGENT_ITERS,
        max_searches: int = MAX_WEB_SEARCHES,
        max_stagnant_turns: int = MAX_STAGNANT_TURNS,
    ):
        self.model = model
        self.max_iters = max_iters
        self.max_searches = max_searches
        self.max_stagnant_turns = max_stagnant_turns
        self.tools = [WEB_SEARCH_TOOL, FETCH_URL_TOOL, UPDATE_PROGRESS_TOOL]
        self.messages: list = []
        self.citations: list = []
        self.searches_used = 0
        self.gaps: dict[str, Gap] = {}
        self.stagnant_turns: int = 0
        self._seen: set = set()

    def run(self, source_text: str) -> ResearchResult:
        # Phase 1: plan — one Opus call identifies what to research
        self.gaps = self._plan_gaps(source_text)
        log.info(
            "research plan: %d gaps — %s",
            len(self.gaps),
            ", ".join(self.gaps.keys()),
        )

        # Phase 2: research loop
        self.messages = [
            {"role": "system", "content": _build_system(self.gaps)},
            {
                "role": "user",
                "content": (
                    "Source article:\n\n"
                    + source_text
                    + "\n\nResearch the knowledge gaps above. Track progress with "
                    "update_progress after each useful result, then write the teaching notes "
                    "when all gaps are covered."
                ),
            },
        ]

        notes = ""
        for i in range(1, self.max_iters + 1):
            resp = client().chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                max_tokens=4096,
            )
            msg = resp.choices[0].message
            self._log_turn(i, msg)
            self._append_assistant(msg)

            # Model declared done — accept its message as notes
            if not msg.tool_calls:
                notes = msg.content or ""
                break

            # Dispatch all tool calls; track whether anything useful happened
            research_attempted = False
            progress_updated = False
            for tc in msg.tool_calls:
                result = self._dispatch(tc)
                if tc.function.name in ("web_search", "fetch_url"):
                    research_attempted = True
                elif tc.function.name == "update_progress":
                    progress_updated = True
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            # Stagnant only if the model loops without researching or reporting
            if research_attempted or progress_updated:
                self.stagnant_turns = 0
            else:
                self.stagnant_turns += 1

            # Stop condition 1: all gaps covered → request final notes
            if all(g.status == "covered" for g in self.gaps.values()):
                log.info("all gaps covered after turn %d — requesting final notes", i)
                notes = self._request_final_notes()
                break

            # Stop condition 2: stagnation → request notes with what we have
            if self.stagnant_turns >= self.max_stagnant_turns:
                log.warning(
                    "stagnation: %d turns without progress — requesting notes", self.stagnant_turns
                )
                notes = self._request_final_notes(
                    "You've had several turns without new progress. "
                    "Write the teaching notes now with what you have."
                )
                break
        else:
            log.warning("hit max_iters=%d guardrail — requesting notes", self.max_iters)
            notes = self._request_final_notes(
                "Research time is up. Write the teaching notes with what you have so far."
            )

        covered = sum(1 for g in self.gaps.values() if g.status == "covered")
        log.info("research done: %d/%d gaps covered", covered, len(self.gaps))
        return ResearchResult(
            notes=notes.strip(),
            citations=self.citations,
            gaps=list(self.gaps.values()),
        )

    # --- planning -----------------------------------------------------------
    def _plan_gaps(self, source_text: str) -> dict[str, Gap]:
        prompt = (
            "You are preparing a research agenda for an audio lecture based on the article below.\n\n"
            "Identify 3-7 specific knowledge gaps — concepts, terms, or context that a curious "
            "listener needs to truly understand the article, but that the source uses without "
            "adequately explaining.\n\n"
            "Respond with JSON only, in this exact shape:\n"
            '{"gaps": [{"id": "slug", "concept": "plain label", "why": "one-sentence reason"}]}\n\n'
            "Rules:\n"
            "- 3-7 gaps; pick the ones that most unlock understanding\n"
            "- id: short lowercase hyphenated slug (e.g. 'rag', 'vector-db')\n"
            "- concept: plain-English name (e.g. 'Retrieval-Augmented Generation')\n"
            "- why: one sentence explaining why a listener needs this\n\n"
            f"Source article:\n\n{source_text}"
        )
        resp = client().chat.completions.create(
            model=MODEL_PLANNER,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            # Strip markdown fences if the model adds them
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("gap planner returned invalid JSON; falling back to empty plan")
            data = {}
        gaps: dict[str, Gap] = {}
        for g in data.get("gaps", []):
            gid = g.get("id", "").strip()
            if gid:
                gaps[gid] = Gap(
                    id=gid,
                    concept=g.get("concept", gid),
                    why=g.get("why", ""),
                )
        return gaps

    # --- stop helpers -------------------------------------------------------
    def _request_final_notes(self, nudge: str = "All gaps are covered.") -> str:
        self.messages.append({
            "role": "user",
            "content": nudge + " Write the teaching notes now (no more tool calls).",
        })
        resp = client().chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=4096,
        )
        return resp.choices[0].message.content or ""

    # --- tool dispatch ------------------------------------------------------
    def _dispatch(self, tc) -> str:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        if name == "web_search":
            return self._web_search(args.get("query", ""))
        if name == "fetch_url":
            return self._fetch(args.get("url", ""))
        if name == "update_progress":
            return self._update_progress(args)
        return f"Unknown tool: {name}"

    def _web_search(self, query: str) -> str:
        if self.searches_used >= self.max_searches:
            return "Search limit reached. Use what you have and write the notes now."
        self.searches_used += 1
        from ddgs import DDGS
        try:
            results = list(DDGS().text(query, max_results=_SEARCH_RESULTS))
        except Exception as e:
            log.warning("web_search failed %r: %s", query, e)
            return f"Search error: {e}"
        log.info("web_search %r -> %d results", query, len(results))
        lines = []
        for r in results:
            url, title, body = r.get("href", ""), r.get("title", ""), r.get("body", "")
            self._cite(url, title)
            lines.append(f"- {title}\n  {url}\n  {body}")
        return "\n".join(lines) if lines else "No results."

    def _fetch(self, url: str) -> str:
        try:
            text = fetch_url(url)[:_MAX_FETCH_CHARS]
            self._cite(url, url)
            log.info("fetch_url -> %s (%d chars)", url, len(text))
            return text
        except Exception as e:
            log.warning("fetch_url failed %s: %s", url, e)
            return f"Error fetching {url}: {e}"

    def _update_progress(self, args: dict) -> str:
        gap_id = args.get("gap_id", "").strip()
        status = args.get("status", "open")
        learned = args.get("what_you_learned", "")
        if gap_id not in self.gaps:
            valid = list(self.gaps.keys())
            return f"Unknown gap_id '{gap_id}'. Valid ids: {valid}"
        self.gaps[gap_id].status = status
        covered = sum(1 for g in self.gaps.values() if g.status == "covered")
        log.info("[progress] %s → %s: %s", gap_id, status, learned[:100])
        log.info("gaps: %d/%d covered", covered, len(self.gaps))
        return f"Recorded: {gap_id} = {status}. Gaps: {covered}/{len(self.gaps)} covered."

    def _cite(self, url: str, title: str) -> None:
        if url and url not in self._seen:
            self._seen.add(url)
            self.citations.append({"url": url, "title": title})

    # --- transcript + logging -----------------------------------------------
    def _append_assistant(self, msg) -> None:
        entry: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        self.messages.append(entry)

    @staticmethod
    def _log_turn(i: int, msg) -> None:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "update_progress":
                    continue  # logged inside _update_progress
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    a = {}
                arg = a.get("query") or a.get("url") or ""
                log.info("[turn %d] %s: %s", i, tc.function.name, arg)
        else:
            log.info("[turn %d] writing notes (%d chars)", i, len(msg.content or ""))
