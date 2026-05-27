"""The research agent — a hand-written agent loop (no framework).

This is the one genuinely *agentic* node in podify. Given the source article, the model
decides what to look up, calls tools, observes the results, and loops until it has enough to
write teaching notes.

We route through OpenRouter (OpenAI-compatible API), which doesn't expose Anthropic's
server-side tools — so BOTH of the agent's tools are client-side: WE execute them and feed the
results back. That makes this a textbook ReAct loop entirely in our own process:

  web_search : run a DuckDuckGo search, return result titles/URLs/snippets
  fetch_url  : download + extract the readable text of one page

The loop is a small state machine over `finish_reason` / the presence of tool_calls:
  tool_calls present -> run our tool(s), append a `tool` message per call, loop
  otherwise          -> the model produced its final notes; done
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from . import MAX_AGENT_ITERS, MAX_WEB_SEARCHES, MODEL_RESEARCH, client
from .fetch import fetch_url

_MAX_FETCH_CHARS = 6000  # cap a fetched page so one tool result can't flood the context
_SEARCH_RESULTS = 5

log = logging.getLogger("podify.research")

# --- tool definitions (OpenAI function-calling schema) ----------------------
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web. Returns a list of result titles, URLs, and snippets. Use this "
            "to find definitions, explanations, examples, and current context for concepts "
            "the source assumes the reader already knows."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}

FETCH_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": (
            "Fetch the main readable text of a web page by URL. Use after web_search to read "
            "a promising result in full when the snippet isn't enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The page URL to fetch."}},
            "required": ["url"],
        },
    },
}

SYSTEM = (
    "You are a research assistant preparing material for an AUDIO lecture — a ~20-minute "
    "spoken lesson for someone listening on a walk. You are given the full text of a source "
    "article.\n\n"
    "Your job:\n"
    "1. Read the source and identify the concepts, terms, and background a curious listener "
    "needs in order to truly understand it — especially things the source uses but never "
    "defines.\n"
    "2. Use web_search to find sources, then fetch_url to read the best ones. Gather clear "
    "definitions, intuitive analogies, concrete real-world examples, and important up-to-date "
    "context the source lacks.\n"
    "3. Then write concise TEACHING NOTES the lecture writer will use: for each gap, a "
    "plain-English explanation plus an analogy or example. Group related points.\n\n"
    "Guidelines: prioritize the few concepts that most unlock understanding — do not research "
    "everything. Prefer authoritative sources. When you have enough, write the notes as your "
    "final message (no more tool calls)."
)


@dataclass
class ResearchResult:
    notes: str
    citations: list = field(default_factory=list)


class ResearchAgent:
    """Holds the agent's state: the message transcript, its tools, model, and loop budget."""

    def __init__(
        self,
        model: str = MODEL_RESEARCH,
        max_iters: int = MAX_AGENT_ITERS,
        max_searches: int = MAX_WEB_SEARCHES,
    ):
        self.model = model
        self.max_iters = max_iters
        self.max_searches = max_searches
        self.tools = [WEB_SEARCH_TOOL, FETCH_URL_TOOL]
        self.messages: list = []
        self.citations: list = []
        self.searches_used = 0
        self._seen: set = set()

    def run(self, source_text: str) -> ResearchResult:
        self.messages = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "Source article:\n\n"
                + source_text
                + "\n\nIdentify the gaps a listener needs, research them, then write the "
                "teaching notes.",
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

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    result = self._dispatch(tc)
                    self.messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
            else:
                notes = msg.content or ""
                break
        else:
            log.warning("hit max_iters=%d guardrail", self.max_iters)

        return ResearchResult(notes=notes.strip(), citations=self.citations)

    # --- tool dispatch (the part a framework would do for us) ----------------
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
        return f"Unknown tool: {name}"

    def _web_search(self, query: str) -> str:
        if self.searches_used >= self.max_searches:  # guardrail
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
        except Exception as e:  # feed the error back so the agent can recover
            log.warning("fetch_url failed %s: %s", url, e)
            return f"Error fetching {url}: {e}"

    def _cite(self, url: str, title: str) -> None:
        if url and url not in self._seen:
            self._seen.add(url)
            self.citations.append({"url": url, "title": title})

    # --- transcript + logging helpers ---------------------------------------
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
                try:
                    a = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    a = {}
                arg = a.get("query") or a.get("url") or ""
                log.info("[turn %d] %s: %s", i, tc.function.name, arg)
        else:
            log.info("[turn %d] writing notes (%d chars)", i, len(msg.content or ""))
