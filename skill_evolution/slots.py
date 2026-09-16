"""Read/replace only the two evolve slots. Backbone is hashed as an invariant."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

QUERY_START = "<!-- EVOLVE:QUERY:START -->"
QUERY_END = "<!-- EVOLVE:QUERY:END -->"
RANK_START = "<!-- EVOLVE:RANK:START -->"
RANK_END = "<!-- EVOLVE:RANK:END -->"
FILTER_START = "<!-- EVOLVE:FILTER:START -->"
FILTER_END = "<!-- EVOLVE:FILTER:END -->"

_SLOT = re.compile(
    r"(<!-- EVOLVE:(QUERY|RANK|FILTER):START -->)(.*?)(<!-- EVOLVE:(QUERY|RANK|FILTER):END -->)",
    re.DOTALL,
)


@dataclass(frozen=True)
class SlotSnapshot:
    invariant_hash: str
    query_body: str
    rank_body: str
    filter_body: str
    text: str


def _replace_slot(text: str, start: str, end: str, body: str) -> str:
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"Missing slot markers {start} .. {end}")
    return pattern.sub(f"{start}\n{body.rstrip()}\n{end}", text, count=1)


def strip_slots(text: str) -> str:
    out = text
    for start, end in (
        (QUERY_START, QUERY_END),
        (RANK_START, RANK_END),
        (FILTER_START, FILTER_END),
    ):
        out = re.sub(
            re.escape(start) + r".*?" + re.escape(end),
            f"{start}\n<SLOT>\n{end}",
            out,
            count=1,
            flags=re.DOTALL,
        )
    return out


def invariant_hash(text: str) -> str:
    blob = strip_slots(text).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def parse_skill(text: str) -> SlotSnapshot:
    q = r = f = None
    for m in _SLOT.finditer(text):
        kind = m.group(2)
        body = m.group(3).strip()
        if kind == "QUERY":
            q = body
        elif kind == "RANK":
            r = body
        else:
            f = body
    if q is None or r is None or f is None:
        raise ValueError("SKILL.md must contain QUERY, RANK, and FILTER evolve slots")
    return SlotSnapshot(
        invariant_hash=invariant_hash(text),
        query_body=q,
        rank_body=r,
        filter_body=f,
        text=text,
    )


def write_slots(path: Path, query_body: str, rank_body: str, filter_body: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = _replace_slot(text, QUERY_START, QUERY_END, query_body)
    text = _replace_slot(text, RANK_START, RANK_END, rank_body)
    text = _replace_slot(text, FILTER_START, FILTER_END, filter_body)
    path.write_text(text, encoding="utf-8")
