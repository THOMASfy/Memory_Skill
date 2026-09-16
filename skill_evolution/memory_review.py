"""Attribution over retrieved slices — no gold ids, no answer recitation."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .types import Hit, SlotName

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'_-]{2,}|[\u4e00-\u9fff]{2,}")
_STOP = {
    "the", "and", "for", "are", "was", "were", "what", "when", "where", "who",
    "how", "why", "which", "that", "this", "with", "from", "have", "has", "had",
    "did", "does", "you", "your", "our", "their", "about", "into", "just",
    "current", "date", "please", "tell", "know", "name",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "什么", "多少", "哪个", "哪些", "怎么", "如何", "是否", "请问",
}
_GENERIC = {
    "long", "short", "daily", "weekly", "monthly", "yearly", "often", "much",
    "many", "well", "good", "new", "old", "first", "last", "really", "also",
    "still", "going", "want", "need", "like", "make", "take", "time", "times",
    "year", "years", "month", "months", "week", "weeks", "day", "days",
    "hour", "hours", "minute", "minutes", "home", "life", "work", "working",
    "job", "people", "thing", "things", "around", "today", "personal",
    "information", "history", "chat", "session",
    "工作", "每天", "时间", "生活", "很久", "一天", "小时", "分钟",
}

# Surface-form variants: matching any member covers the same slot.
_SYN: dict[str, tuple[str, ...]] = {
    "play": ("theater", "theatre", "show", "performance", "musical", "drama", "playhouse"),
    "theater": ("theatre", "play", "playhouse", "stage"),
    "theatre": ("theater", "play", "playhouse", "stage"),
    "commute": ("commuting", "drive", "driving", "travel", "trip", "commuter"),
    "duration": ("minutes", "hours", "long", "length"),
    "size": ("inches", "inch", "diagonal", "screen"),
    "color": ("colour", "gray", "grey", "shade"),
    "colour": ("color", "gray", "grey", "shade"),
    "degree": ("bachelor", "master", "phd", "graduated", "graduation"),
    "discount": ("coupon", "percent", "off", "sale", "promo"),
    "coupon": ("discount", "promo", "voucher", "code"),
    "yoga": ("studio", "class", "pilates"),
    "store": ("shop", "mall", "retailer", "outlet"),
    "where": ("store", "shop", "location", "place", "address"),
}

_MEASURE_HINT = {
    "size", "length", "duration", "long", "inches", "inch", "minutes", "hours",
    "percent", "discount", "commute", "how",
}

# WH-patterns: the noun phrase is the attribute, not the filler (how long / what / which).
_WH_HEAD = (
    re.compile(
        r"how (?:long|far|often|much|many) (?:is|was|are|were|do|does|did) "
        r"(?:my |the |our |your |a |an )?(.+?)(?:\?|$)",
        re.I,
    ),
    re.compile(r"what(?:'s| is| was| are) (?:the )?(?:name of (?:the |my |our )?)?(.+?)(?:\?|$)", re.I),
    re.compile(r"\bwhat ([a-z][a-z0-9'-]{2,}) (?:is|are|was|were|do|does|did)\b", re.I),
    re.compile(r"what ([a-z][a-z0-9'-]+) did i", re.I),
    re.compile(r"(?:which|whose) ([a-z][a-z0-9'-]+)", re.I),
)
_TIME_CUES = {
    "now", "today", "currently", "recent", "recently", "latest", "before",
    "after",     "ago", "year", "years", "month", "when", "since", "previous",
    "previously", "once", "updated", "update", "changed", "beforehand",
    "现在", "目前", "最近", "之前", "之后", "去年", "今年", "当时", "何时",
    "以前", "此前", "后来", "上次",
}

SNIPPET_CHARS = 500
REL_OV = 0.15
IRR_OV = 0.08

# Closed Target Value Classes: the answer is a concrete filler, not a topic word.
_MONTHS = (
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
)
_COLORS = {
    "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "pink",
    "purple", "orange", "brown", "navy", "beige", "ivory", "gold", "silver",
}
_DEGREE_VAL = (
    r"(?:ph\.?d|mba|b\.?a\.?|b\.?s\.?|m\.?s\.?|m\.?a\.?|"
    r"bachelors?|masters?|doctorate|associates?|"
    r"business administration|computer science|electrical engineering)"
)
_VALUE_LABEL = {
    "measure": "specific measured number",
    "count": "specific count number",
    "datetime": "specific date",
    "place": "specific place name",
    "entity": "specific named entity",
    "color": "specific color",
    "item": "specific purchased item",
}


@dataclass
class MemoryReview:
    covers_question: bool
    cause: str
    missing_terms: tuple[str, ...]
    relevant_ids: tuple[str, ...]
    soft_neg_ids: tuple[str, ...]
    hard_neg_ids: tuple[str, ...]
    query_focus: str
    slot: SlotName
    query_locked: bool
    time_mismatch: bool
    proxy: float
    missing_keys: tuple[str, ...]
    note: str
    snippets: list[dict] = field(default_factory=list)
    missing_llm: tuple[str, ...] = ()
    hop_phrase: str = ""


def _terms(text: str) -> set[str]:
    found = {m.group(0).lower() for m in _TOKEN.finditer(text or "")}
    return {t for t in found if t not in _STOP}


def _snippet(text: str) -> str:
    compact = " ".join((text or "").split())
    return compact[:SNIPPET_CHARS]


def _overlap(q_terms: set[str], text: str) -> float:
    ht = _terms(text)
    if not q_terms:
        return 0.0
    return len(q_terms & ht) / float(len(q_terms))


def _has_term(term: str, text: str) -> bool:
    if not term or not text:
        return False
    if term in _terms(text):
        return True
    return re.search(rf"\b{re.escape(term)}\b", text, re.I) is not None


def _stems(word: str) -> set[str]:
    w = (word or "").lower()
    out = {w}
    if len(w) > 4 and w.endswith("ies"):
        out.add(w[:-3] + "y")
    if len(w) > 4 and w.endswith("es"):
        out.add(w[:-2])
    if len(w) > 3 and w.endswith("s"):
        out.add(w[:-1])
    if len(w) > 5 and w.endswith("ing"):
        out.add(w[:-3])
        out.add(w[:-3] + "e")
    if len(w) > 4 and w.endswith("ed"):
        out.add(w[:-2])
        out.add(w[:-1])
    return {x for x in out if len(x) > 2}


def _term_or_syn(term: str, text: str) -> bool:
    """True if the same meaning is present — not only the same spelling."""
    if not term or not text:
        return False
    low = text.lower()
    variants = _stems(term) | set(_SYN.get(term.lower(), ()))
    for syn, group in _SYN.items():
        if term.lower() == syn or term.lower() in group:
            variants |= {syn, *group}
    for v in variants:
        if _has_term(v, text) or re.search(rf"\b{re.escape(v)}\b", low):
            return True
    return False


def _has_measure(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d+(\.\d+)?\s*(minutes?|hours?|days?|weeks?|months?|%|inches?|inch|cm|km|miles?)\b",
            text or "",
            re.I,
        )
    )


def _slot_evidenced(term: str, texts: list[str], question: str) -> bool:
    """Asked slot is present as content, including paraphrase / instance."""
    blob = " ".join(texts)
    if _term_or_syn(term, blob):
        return True
    qlow = (question or "").lower()
    wants_measure = term.lower() in _MEASURE_HINT or "how long" in qlow or "how far" in qlow
    if wants_measure and _has_measure(blob):
        return True
    if _has_concrete_filler(question, blob) and (
        term.lower() in _SYN or any(term.lower() in g for g in _SYN.values())
    ):
        return True
    return False


def _attribute_terms(question: str, q_terms: set[str]) -> set[str]:
    """Focus attributes from the question only. Slice tokens never become keys."""
    heads: set[str] = set()
    for pat in _WH_HEAD:
        m = pat.search(question)
        if m:
            heads |= _terms(m.group(1))
            g1 = m.group(1).strip().lower()
            if re.fullmatch(r"[a-z][a-z0-9'-]{1,}", g1):
                heads.add(g1)
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9'-]{1,})\b", question):
        w = m.group(1)
        if w.lower() in {"current", "date"}:
            continue
        if w.lower() not in _STOP and w.lower() not in _GENERIC:
            heads.add(w.lower())
    heads -= _GENERIC
    heads -= _STOP
    if heads:
        return heads
    return {t for t in q_terms if t not in _GENERIC}


def _doc_freq(hits: list[Hit], terms: set[str]) -> dict[str, int]:
    df = {t: 0 for t in terms}
    for h in hits:
        text = h.doc.text or ""
        for t in terms:
            if _has_term(t, text):
                df[t] += 1
    return df


def proxy_quality(hits: list[Hit], question: str, key_terms: set[str] | None = None) -> float:
    """Top-3 coverage of *key* question terms only. Generic overlap does not count."""
    keys = key_terms if key_terms is not None else _attribute_terms(question, _terms(question))
    if not hits or not keys:
        return 0.0
    ovs = [_overlap(keys, h.doc.text) for h in hits]
    top3 = ovs[:3]
    in_top3 = sum(1.0 for o in top3 if o >= REL_OV) / max(len(top3), 1)
    mean3 = float(sum(top3) / max(len(top3), 1))
    mean10 = float(sum(ovs) / max(len(ovs), 1))
    return 2.0 * mean3 + 1.0 * in_top3 + 0.3 * mean10


def _aliases_from_slices(missing: set[str], hits: list[Hit]) -> list[str]:
    """Only stems/variants that already occur in retrieved text — not guessed answers."""
    bag: set[str] = set()
    for h in hits:
        bag |= _terms(h.doc.text)
    out: list[str] = []
    for m in missing:
        for tok in bag:
            if tok == m:
                continue
            if tok.startswith(m) or m.startswith(tok):
                if abs(len(tok) - len(m)) <= 4:
                    out.append(tok)
        if m.endswith("e") and (m + "d") in bag:
            out.append(m + "d")
        if (m + "d") in bag:
            out.append(m + "d")
        if (m + "ing") in bag:
            out.append(m + "ing")
        if (m + "ion") in bag:
            out.append(m + "ion")
    # unique, short
    seen = set()
    uniq = []
    for x in out:
        if x not in seen and x not in missing:
            seen.add(x)
            uniq.append(x)
        if len(uniq) >= 6:
            break
    return uniq


def _time_mismatch(
    question: str,
    hits: list[Hit],
    relevant: list[Hit],
    query_time_days: float | None,
) -> bool:
    qlow = question.lower()
    has_cue = any(c in qlow for c in _TIME_CUES) or any(c in question for c in _TIME_CUES)
    # Recency fusion is off unless the question itself asks about time.
    if not has_cue:
        return False
    dated = [h for h in hits if h.doc.time_days is not None]
    if len(dated) < 3:
        return False
    times = [float(h.doc.time_days) for h in dated]
    spread = max(times) - min(times)
    t_star = float(query_time_days) if query_time_days is not None else max(times)
    old_rel = [
        h for h in relevant
        if h.doc.time_days is not None and float(h.doc.time_days) < t_star - 365
    ]
    return spread > 180 or len(old_rel) >= 1


def review_memory(
    question: str,
    hits: list[Hit],
    *,
    llm_complete: Callable[[str], str] | None = None,
    query_time_days: float | None = None,
) -> MemoryReview:
    q_body = _question_body(question)
    q_terms = _terms(q_body)
    keys = _attribute_terms(q_body, q_terms)
    tgt = parse_target_slot(q_body)
    top3 = hits[:3]
    df_top3 = _doc_freq(top3, keys)
    distinctive = keys

    snippets = []
    scored: list[tuple[float, Hit]] = []
    union: set[str] = set()
    for h in hits:
        ht = _terms(h.doc.text)
        union |= ht
        ov = _overlap(keys, h.doc.text) if keys else _overlap(q_terms, h.doc.text)
        scored.append((ov, h))
        snippets.append(
            {
                "id": h.doc.id,
                "sim": round(float(h.sim), 4),
                "overlap": round(ov, 4),
                "text": _snippet(h.doc.text),
                "text_full": (h.doc.text or "")[:2500],
            }
        )

    has_value = any(
        value_asserted_in_text(tgt, (h.doc.text or ""), q_body) for h in hits[:3]
    )
    missing_keys = () if has_value else ((hop_phrase(q_body, tgt),) if hop_phrase(q_body, tgt) else ())

    missing = tuple(sorted(q_terms - union))[:12]
    present_keys = {t for t in keys if df_top3.get(t, 0) > 0}
    relevant_hits = [
        h for ov, h in scored
        if present_keys and _overlap(present_keys, h.doc.text) >= REL_OV
    ]
    relevant_ids = tuple(h.doc.id for h in relevant_hits)
    soft, hard = [], []
    t_star = query_time_days
    rel_set = set(relevant_ids)
    for ov, h in scored:
        if h.doc.id in rel_set:
            continue
        if ov < IRR_OV:
            soft.append(h.doc.id)
            continue
        old = (
            t_star is not None
            and h.doc.time_days is not None
            and float(h.doc.time_days) < float(t_star) - 365
        )
        if old:
            hard.append(h.doc.id)
        else:
            soft.append(h.doc.id)

    top3 = hits[:3]
    top3_rel = sum(1 for h in top3 if h.doc.id in rel_set)
    noise_top3 = bool(relevant_ids) and top3_rel < min(2, len(relevant_ids))
    key_unrecalled = bool(missing_keys)
    semantic_gap = not relevant_ids
    time_mis = _time_mismatch(question, hits, relevant_hits, query_time_days)

    if key_unrecalled:
        cause = "unrecalled"
        slot: SlotName = "query"
        query_locked = False
    elif semantic_gap:
        cause = "semantic_gap"
        slot = "query"
        query_locked = False
    elif time_mis or noise_top3:
        cause = "time_mismatch" if time_mis else "noise_top3"
        slot = "rank"
        query_locked = True
    else:
        cause = "rank_refine"
        slot = "rank"
        query_locked = True

    aliases = _aliases_from_slices(set(missing_keys) or set(missing), hits)
    focus_parts = list(missing_keys[:6] or missing[:6]) + aliases[:4]
    focus = " ".join(focus_parts).strip()
    proxy = proxy_quality(hits, question, distinctive)
    note = (
        f"attr={cause} proxy={proxy:.3f} top3_rel={top3_rel} "
        f"tvc={tgt.cls}:{tgt.grain} missing_keys={list(missing_keys)[:5] or 'none'} "
        f"keys={list(keys)[:5]} rel={len(relevant_ids)} soft={len(soft)} hard={len(hard)}"
    )

    review = MemoryReview(
        covers_question=False,
        cause=cause,
        missing_terms=missing,
        relevant_ids=relevant_ids,
        soft_neg_ids=tuple(soft),
        hard_neg_ids=tuple(hard),
        query_focus=focus,
        slot=slot,
        query_locked=query_locked,
        time_mismatch=time_mis,
        proxy=proxy,
        missing_keys=missing_keys,
        note=note,
        snippets=snippets,
    )
    if llm_complete is not None and question.strip() and hits:
        review, llm_ok = _llm_refine(question, review, llm_complete)
        if not llm_ok:
            tgt = parse_target_slot(question)
            top3_texts = [s.get("text_full") or s.get("text") or "" for s in review.snippets[:3]]
            has_value = any(value_asserted_in_text(tgt, txt, question) for txt in top3_texts)
            if not has_value:
                review.covers_question = False
                review.cause = "unrecalled"
                review.slot = "query"
                review.query_locked = False
                review.missing_keys = (hop_phrase(question, tgt),) if hop_phrase(question, tgt) else ()
                review.query_focus = review.missing_keys[0] if review.missing_keys else ""
        return _apply_sufficiency(question, review)
    return _apply_sufficiency(question, review)


def _extract_json_obj(raw: str) -> dict | None:
    if not raw or "{" not in raw:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (
            "missing_keys" in obj or "slot_decision" in obj or "query_intents" in obj
        ):
            return obj
    return None


@dataclass
class TargetSlot:
    """Asked fact type. Gold strings never go here — only the class and a search phrase."""
    cls: str
    label: str
    grain: str
    bind: tuple[str, ...]


def _question_body(question: str) -> str:
    return re.sub(r"^Current date:.*\n", "", question or "", flags=re.I).strip() or (question or "")


def _bind_terms(question: str) -> tuple[str, ...]:
    q = _question_body(question)
    terms = [t for t in _terms(q) if t not in _GENERIC]
    # Keep distinctive nouns; drop leftover WH.
    drop = {"did", "does", "do", "take", "attend", "buy", "bought", "pack", "packed"}
    out = [t for t in terms if t not in drop]
    return tuple(out[:8])


_ABSTRACT_MISSING = {
    "specific count number",
    "specific duration minutes",
    "specific measured number",
    "specific named entity",
    "specific fact value",
    "specific place name",
    "specific degree name",
    "specific color",
    "specific date",
    "specific purchased item",
    "specific screen size number",
    "specific play title",
}


def hop_phrase(question: str, tgt: "TargetSlot | None" = None) -> str:
    """Bound encode text: question entities + value class. Empty if nothing to bind."""
    tgt = tgt or parse_target_slot(question)
    nouns = [t for t in tgt.bind if t not in _GENERIC and t not in _STOP][:5]
    if len(nouns) < 1:
        return ""
    tail = {
        ("measure", "size"): "size inches",
        ("measure", "duration"): "duration minutes",
        ("count", "count"): "count number",
        ("datetime", "day"): "calendar date",
        ("place", "place"): "place name",
        ("entity", "degree"): "degree name",
        ("entity", "title"): "play title",
        ("color", "color"): "color",
        ("item", "item"): "purchased item",
        ("entity", "entity"): "name",
    }.get((tgt.cls, tgt.grain), tgt.cls)
    text = " ".join(nouns + [tail])
    low = text.lower().strip()
    if low in _ABSTRACT_MISSING or all(w in _ABSTRACT_MISSING for w in [low]):
        return ""
    if low.startswith("specific ") and len(nouns) < 2:
        return ""
    return text[:80]


def is_abstract_missing(phrase: str) -> bool:
    p = " ".join((phrase or "").lower().split())
    if p in _ABSTRACT_MISSING:
        return True
    return bool(p.startswith("specific ") and len(p.split()) <= 4)


def parse_target_slot(question: str) -> TargetSlot:
    q = _question_body(question)
    qlow = q.lower()
    bind = _bind_terms(q)

    def lab(*parts: str) -> str:
        words = [p for p in parts if p]
        return " ".join(words)[:48].strip() or _VALUE_LABEL["entity"]

    if re.search(r"\bwhat size\b|\bhow (?:big|large|wide|tall)\b|\binches?\b|\bdiagonal\b", qlow):
        return TargetSlot("measure", lab("specific", "screen size", "number"), "size", bind)
    if re.search(r"\bhow long\b|\bhow far\b|\bcommute\b|\bduration\b", qlow):
        return TargetSlot("measure", lab("specific", "duration", "minutes"), "duration", bind)
    if re.search(r"\bhow many\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b", qlow):
        return TargetSlot("measure", lab("specific", "duration", "minutes"), "duration", bind)
    if re.search(r"\bhow many\b|\bhow much\b", qlow):
        return TargetSlot("count", lab("specific", "count", "number"), "count", bind)
    if re.search(r"\bwhat colou?r\b|\bwhich colou?r\b", qlow):
        return TargetSlot("color", lab("specific", "color"), "color", bind)
    if re.search(r"\bwhen\b|\bwhat date\b|\bwhich day\b|\bwhat day\b", qlow):
        return TargetSlot("datetime", lab("specific", "date"), "day", bind)
    if re.search(r"\bwhere\b", qlow):
        return TargetSlot("place", lab("specific", "place", "name"), "place", bind)
    if re.search(r"\bwhat degree\b|\bgraduate with\b|\bgraduated with\b", qlow):
        return TargetSlot("entity", lab("specific", "degree", "name"), "degree", bind)
    if re.search(r"\bwhat play\b|\bwhich play\b|\bplay did i\b", qlow):
        return TargetSlot("entity", lab("specific", "play", "title"), "title", bind)
    if re.search(r"\bwhat did i buy\b|\bwhat did i (?:get|purchase)\b|\bbirthday gift\b", qlow):
        return TargetSlot("item", lab("specific", "purchased", "item"), "item", bind)
    if re.search(r"\bwhat(?:'s| is| was)\b|\bwhich\b|\bwho\b|\bwhose\b", qlow):
        return TargetSlot("entity", lab("specific", "named", "entity"), "entity", bind)
    return TargetSlot("entity", lab("specific", "fact", "value"), "entity", bind)


def _slice_bound(text: str, bind: tuple[str, ...]) -> bool:
    """Value must sit in a slice that mentions the asked topic, not a random number/name."""
    if not bind:
        return True
    return any(_has_term(b, text) or _term_or_syn(b, text) for b in bind[:6])


def _has_size_number(text: str) -> bool:
    return bool(re.search(r"\b\d+(\.\d+)?\s*(inches?|inch|\"|cm)\b", text or "", re.I))


def _has_duration_number(text: str) -> bool:
    return bool(re.search(r"\b\d+(\.\d+)?\s*(minutes?|mins?|hours?|hrs?|days?)\b", text or "", re.I))


def _has_count_number(text: str) -> bool:
    t = text or ""
    t = re.sub(r"\b\d+\s*-?\s*days?\b", " ", t, flags=re.I)
    return bool(
        re.search(
            r"\b(?:packed|pack|shirts?|brought|took|items?|pairs?)\b.{0,48}\b([1-9]|[1-9]\d)\b|"
            r"\b([1-9]|[1-9]\d)\b.{0,24}\b(?:shirts?|items?|pairs?)\b",
            t,
            re.I,
        )
    )


def _has_day_date(text: str) -> bool:
    t = text or ""
    if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", t):
        return True
    if re.search(r"\b([1-9]|[12]\d|3[01])(st|nd|rd|th)\b", t, re.I):
        return True
    months = "|".join(_MONTHS)
    return bool(re.search(rf"\b({months})\s+\d{{1,2}}\b", t, re.I))


def _has_month(text: str) -> bool:
    months = "|".join(_MONTHS)
    return bool(re.search(rf"\b({months})\b", text or "", re.I))


def _has_color_value(text: str) -> bool:
    low = (text or "").lower()
    return any(re.search(rf"\b{re.escape(c)}\b", low) for c in _COLORS)


def _has_place_value(text: str, question: str) -> bool:
    qlow = _question_body(question).lower()
    if re.search(r"\b(at|from|in)\s+[A-Z][A-Za-z0-9'&.-]+", text or ""):
        return True
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9'&.-]+(?:\s+[A-Z][a-zA-Z0-9'&.-]+)+)\b", text or ""):
        if m.group(1).lower() not in qlow:
            return True
    return False


def _has_entity_value(text: str, question: str, grain: str) -> bool:
    if grain == "degree":
        return bool(re.search(rf"\b{_DEGREE_VAL}\b", text or "", re.I))
    if grain == "title":
        if not re.search(r"\b(theater|theatre|playhouse|musical|drama|attended|saw)\b", text or "", re.I):
            if "play" not in (text or "").lower():
                return False
    qlow = _question_body(question).lower()
    if re.search(r"[“\"']([^\"']{3,80})[”\"']", text or ""):
        return True
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9'&.-]+(?:\s+[A-Z][a-zA-Z0-9'&.-]+)+)\b", text or ""):
        if m.group(1).lower() not in qlow:
            return True
    return False


def _has_item_value(text: str, question: str) -> bool:
    if re.search(r"\b(bought|buy|purchased|got)\b.{0,80}\b(a|an|the)\s+[a-z]{3,}", text or "", re.I):
        return True
    return _has_entity_value(text, question, "item")


def value_asserted_in_text(slot: TargetSlot, text: str, question: str) -> bool:
    if not _slice_bound(text, slot.bind):
        return False
    if slot.cls == "measure":
        return _has_size_number(text) if slot.grain == "size" else _has_duration_number(text)
    if slot.cls == "count":
        return _has_count_number(text)
    if slot.cls == "datetime":
        return _has_day_date(text) if slot.grain == "day" else (_has_day_date(text) or _has_month(text))
    if slot.cls == "place":
        return _has_place_value(text, question)
    if slot.cls == "color":
        return _has_color_value(text)
    if slot.cls == "item":
        return _has_item_value(text, question)
    return _has_entity_value(text, question, slot.grain)


def target_value_in_top3(question: str, snippets: list[dict]) -> bool:
    slot = parse_target_slot(question)
    for s in snippets[:3]:
        if value_asserted_in_text(slot, s.get("text_full") or s.get("text") or "", question):
            return True
    return False


def _sanitize_phrases(items, question: str, slice_blob: str, *, require_q_overlap: bool = True) -> tuple[str, ...]:
    """1-3 short phrases. Prefer question overlap. Reject slice-copy and sentences."""
    qlow = _question_body(question).lower()
    blob = (slice_blob or "").lower()
    out: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        s = " ".join(str(item).split()).strip(" .,;:\"'`")
        if not s or "." in s or "?" in s or len(s) > 48:
            continue
        words = s.split()
        if not (1 <= len(words) <= 4):
            continue
        low = s.lower()
        if all(w in _STOP or w in _GENERIC for w in words):
            continue
        if len(low) >= 12 and low in blob:
            continue
        q_hit = low in qlow or any(w.lower() in qlow for w in words if len(w) > 2)
        if require_q_overlap and not q_hit and len(words) > 2:
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(low)
        if len(out) >= 3:
            break
    return tuple(out)


def _has_concrete_filler(question: str, text: str) -> bool:
    qlow = (question or "").lower()
    if _has_measure(text or ""):
        return True
    for m in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)\b", text or ""):
        if m.group(1).lower() not in qlow:
            return True
    return False


def _top3_enough_to_answer(question: str, snippets: list[dict]) -> bool:
    slot = parse_target_slot(question)
    for s in snippets[:3]:
        text = s.get("text_full") or s.get("text") or ""
        if value_asserted_in_text(slot, text, question):
            return True
    return False


def _apply_sufficiency(question: str, review: MemoryReview) -> MemoryReview:
    """Clear missing only if Top-3 has a concrete value assertion. Never encode abstract labels."""
    if not review.missing_llm:
        review.missing_llm = tuple(review.missing_keys)
    tgt = parse_target_slot(question)
    hop = hop_phrase(question, tgt)
    review.hop_phrase = hop
    if _top3_enough_to_answer(question, review.snippets):
        review.missing_keys = ()
        review.covers_question = True
        review.slot = "rank"
        review.query_locked = True
        if review.cause in {"unrecalled", "semantic_gap"}:
            review.cause = "rank_refine"
        review.note = (review.note + f" tvc={tgt.cls} enough=1 hop={hop or '-'}")[:240]
        return review
    llm_keep = tuple(
        k for k in (review.missing_llm or ())
        if k and not is_abstract_missing(k) and k.lower() != (tgt.label or "").lower()
    )
    if hop:
        review.missing_keys = (hop,)
    elif llm_keep:
        review.missing_keys = llm_keep[:1]
        review.hop_phrase = llm_keep[0]
    else:
        review.missing_keys = ()
    review.covers_question = False
    review.slot = "query" if (hop or llm_keep) else "rank"
    review.query_locked = not bool(hop or llm_keep)
    review.cause = "unrecalled" if (hop or llm_keep) else "rank_refine"
    review.query_focus = hop or (llm_keep[0] if llm_keep else "")
    review.note = (review.note + f" tvc={tgt.cls} hop={hop or 'none'} missing={list(review.missing_keys)[:2]}")[:240]
    return review


def _llm_refine(question: str, base: MemoryReview, llm_complete) -> tuple[MemoryReview, bool]:
    payload = [
        {"id": s["id"], "rank": i, "text": s["text"][:400]}
        for i, s in enumerate(base.snippets[:10], start=1)
    ]
    q_body = _question_body(question)
    tgt = parse_target_slot(q_body)
    hop = hop_phrase(q_body, tgt)
    prompt = (
        "You are a retrieval-sufficiency auditor, not an answering model. "
        "Do NOT answer the user question. Do NOT write a chat reply.\n"
        "Do NOT guess the hidden gold answer. Do NOT copy slice sentences into missing_keys.\n"
        f"Target value class: {tgt.cls} (grain={tgt.grain}). "
        "Coverage means a CONCRETE VALUE assertion of that class is present in Top-3 "
        "(a number with unit, a calendar date, a place name, a work title, a color, an item), "
        "in a slice that is about the asked topic. Related background without that value is NOT coverage.\n"
        f"If the value is absent, missing_keys MUST be [\"{hop or 'bound topic phrase'}\"] "
        "(entity nouns from the question plus the value type). Never use abstract labels "
        "like 'specific count number'. slot_decision QUERY.\n"
        "If the value is present, missing_keys MUST be [] and slot_decision RANK.\n"
        "Reply with JSON only, first character `{`, last character `}`. No markdown.\n"
        '{"query_intents":["..."],"covered_in_memory":["..."],'
        '"missing_keys":["..."],"slot_decision":"QUERY"}\n\n'
        f"Question: {q_body}\n\nTop slices (rank order):\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        raw = llm_complete(prompt)
    except Exception:
        return base, False
    data = _extract_json_obj(raw or "")
    if not data:
        return base, False
    slice_blob = " ".join(s.get("text") or "" for s in base.snippets[:10])
    intents = _sanitize_phrases(data.get("query_intents"), q_body, "")
    covered = _sanitize_phrases(data.get("covered_in_memory"), q_body, slice_blob)
    missing = _sanitize_phrases(data.get("missing_keys"), q_body, slice_blob, require_q_overlap=False)
    decision = str(data.get("slot_decision") or data.get("slot") or "").strip().upper()
    if missing:
        slot: SlotName = "query"
        locked = False
        cause = "unrecalled"
        covers = False
    elif decision == "QUERY":
        slot = "query"
        locked = False
        cause = "semantic_gap"
        covers = False
    else:
        slot = "rank"
        locked = True
        cause = "rank_refine"
        covers = True
    focus = " ".join(missing or intents[:3])
    note = (
        f"attr-llm:{cause} slot={slot} intents={list(intents)} "
        f"covered={list(covered)} missing={list(missing)}"
    )
    return (
        MemoryReview(
            covers_question=covers,
            cause=cause,
            missing_terms=base.missing_terms,
            relevant_ids=base.relevant_ids,
            soft_neg_ids=base.soft_neg_ids,
            hard_neg_ids=base.hard_neg_ids,
            query_focus=focus,
            slot=slot,
            query_locked=locked,
            time_mismatch=base.time_mismatch,
            proxy=base.proxy,
            missing_keys=missing,
            missing_llm=missing,
            note=note[:240],
            snippets=base.snippets,
        ),
        True,
    )
