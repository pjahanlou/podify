"""The author stage — turn the source + research notes into a spoken lecture script.

This is a single LLM call, not an agent: a deterministic transform from {source, notes} to a
single-narrator, audio-first script of roughly the target length.
"""
from __future__ import annotations

from . import MODEL_AUTHOR, WORDS_PER_MINUTE, client

SYSTEM = (
    "You are a master teacher who writes LECTURES MEANT TO BE HEARD, not read. Your script "
    "will be read aloud by a text-to-speech voice to someone walking outside — they cannot "
    "see anything.\n\n"
    "Write a single-narrator spoken lecture that TEACHES the source material — do not merely "
    "summarize it. Use the supplied research notes to define jargon and to add analogies and "
    "examples the source assumes.\n\n"
    "Structure it like a great lecture:\n"
    "- A hook that motivates why this matters.\n"
    "- A quick roadmap of what's coming.\n"
    "- The body: build ideas from simple to advanced, one concept at a time, each with a "
    "concrete analogy or example. Signpost transitions (\"First...\", \"Here's the key "
    "idea...\", \"Now that we've seen X, let's...\").\n"
    "- Brief recaps of the key points along the way.\n"
    "- A short closing that ties it together and leaves one memorable takeaway.\n\n"
    "Rules for audio:\n"
    "- Output ONLY the words to be spoken. No titles, no headings, no markdown, no bullet "
    "points, no stage directions, no \"[pause]\" markers, no URLs.\n"
    "- Conversational and warm. Short, clear sentences. Spell out an abbreviation the first "
    "time you use it.\n"
    "- NEVER reference anything visual (\"as you can see\", \"in the diagram\", \"the figure "
    "above\").\n"
    "- Separate major beats with a blank line (a new paragraph) so the narration can breathe.\n"
    "- Aim for about {words} words (~{minutes} minutes at a relaxed speaking pace)."
)


def write_script(source_text: str, notes: str, target_minutes: int) -> str:
    words = target_minutes * WORDS_PER_MINUTE
    user = (
        "SOURCE ARTICLE:\n\n" + source_text + "\n\n"
        "RESEARCH NOTES (use these to explain jargon and add analogies and examples):\n\n"
        + (notes or "(none)")
        + f"\n\nWrite the spoken lecture now, about {words} words. Output only the words to "
        "be spoken."
    )
    resp = client().chat.completions.create(
        model=MODEL_AUTHOR,
        max_tokens=8192,
        messages=[
            {"role": "system", "content": SYSTEM.format(words=words, minutes=target_minutes)},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
