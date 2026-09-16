"""LongMemEval harness: evolve-retrieve, OpenAI-compatible reader, official yes/no judge.

Usage (intranet LiteLLM / OpenAI-compatible gateway):
  set LLM_API_KEY=-
  set LLM_API_BASE=http://10.0.62.177:4000/v1
  set LLM_MODEL=deepseek-v4-free
  set EMBEDDING_DEVICE=cpu
  python -m skill_evolution.longmemeval_eval --split s --limit 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .loop import run_evolution
from .types import Document

# Official LongMemEval QA types (print_qa_metrics.py / paper §3).
PAPER_QA_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "longmemeval"
CACHE_DIR = ROOT / "runs" / "lme_cache"
OUT_DIR = ROOT / "runs" / "longmemeval"

SPLITS = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}
HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DATE_FORMATS = (
    "%Y/%m/%d (%a) %H:%M",
    "%Y/%m/%d (%a) %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)

READER_SYSTEM = (
    "You are the answer reader. Your only job is to answer the user from History Chats. "
    "Do not output JSON. Do not audit retrieval. Do not write planning notes "
    "(no 'We need', no 'Let's'). If the history is insufficient, say you do not know."
)
AUDITOR_SYSTEM = (
    "You are a retrieval-sufficiency auditor. Do not answer the user question. "
    "Output JSON only about whether Top-k memory covers the asked attributes."
)
READER_TEMPLATE = (
    "I will give you several history chats between you and a user. "
    "Please answer the question based on the relevant chat history. "
    "First extract the relevant facts, then give the answer. "
    "If the history does not contain enough information, say you do not know.\n\n\n"
    "History Chats:\n\n{history}\n\nCurrent Date: {date}\nQuestion: {question}\n"
    "Answer:"
)


def _load_dotenv() -> None:
    for path in (ROOT / ".env", Path.cwd() / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def parse_lme_date(raw: str | None) -> float | None:
    if not raw:
        return None
    text = str(raw).strip()
    m = re.match(
        r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?:\s+\([^)]+\))?(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?",
        text,
    )
    if m:
        hour = int(m.group(4) or 0)
        minute = int(m.group(5) or 0)
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), hour, minute)
        return float(dt.toordinal()) + hour / 24.0 + minute / (24.0 * 60.0)
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
            return float(dt.toordinal()) + dt.hour / 24.0 + dt.minute / (24.0 * 60.0)
        except ValueError:
            continue
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if not m:
        return None
    return float(datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).toordinal())


def download_split(split: str) -> Path:
    name = SPLITS[split]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / name
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    url = f"{HF_BASE}/{name}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def load_instances(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Unexpected LongMemEval file: {path}")
    return data


def session_text(session: list, date: str) -> str:
    lines = [f"[session date: {date}]"]
    for turn in session:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def instance_documents(entry: dict, embeddings: np.ndarray) -> list[Document]:
    docs: list[Document] = []
    sessions = entry["haystack_sessions"]
    ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]
    for i, (sid, date, sess) in enumerate(zip(ids, dates, sessions)):
        docs.append(
            Document(
                id=str(sid),
                text=session_text(sess, str(date)),
                embedding=embeddings[i],
                time_days=parse_lme_date(str(date)),
                session_id=str(sid),
            )
        )
    return docs


CHUNK_TURNS = 8
CHUNK_OVERLAP = 2
CHUNK_CHARS = 1600


def iter_session_chunks(session: list, date: str) -> list[str]:
    """Turn windows so a buried fact is not diluted by a 10k-char session vector."""
    full = session_text(session, date)
    turns = [t for t in (session or []) if isinstance(t, dict)]
    if len(full) <= CHUNK_CHARS or len(turns) <= CHUNK_TURNS:
        return [full]
    out: list[str] = []
    i = 0
    n = len(turns)
    step = max(1, CHUNK_TURNS - CHUNK_OVERLAP)
    while i < n:
        piece = turns[i : i + CHUNK_TURNS]
        out.append(session_text(piece, date))
        if i + CHUNK_TURNS >= n:
            break
        i += step
    return out or [full]


def build_chunk_documents(entry: dict, encoder: "BGEEncoder") -> list[Document]:
    texts: list[str] = []
    meta: list[tuple[str, str, str, str]] = []
    for sid, date, sess in zip(
        entry["haystack_session_ids"], entry["haystack_dates"], entry["haystack_sessions"]
    ):
        parts = iter_session_chunks(sess, str(date))
        sid_s = str(sid)
        for j, text in enumerate(parts):
            cid = sid_s if len(parts) == 1 else f"{sid_s}::c{j}"
            meta.append((cid, sid_s, str(date), text))
            texts.append(text)
    vecs = encoder.encode(texts, is_query=False)
    docs: list[Document] = []
    for i, (cid, sid_s, date, text) in enumerate(meta):
        docs.append(
            Document(
                id=cid,
                text=text,
                embedding=vecs[i],
                time_days=parse_lme_date(date),
                session_id=sid_s,
            )
        )
    return docs


def session_id_of(hit) -> str:
    sid = str(getattr(hit.doc, "session_id", "") or "")
    if sid:
        return sid
    did = str(hit.doc.id)
    return did.split("::c")[0]


def collapse_to_sessions(hits: list, limit: int | None = None) -> list:
    seen: set[str] = set()
    out = []
    for h in hits:
        sid = session_id_of(h)
        if sid in seen:
            continue
        seen.add(sid)
        out.append(h)
        if limit is not None and len(out) >= limit:
            break
    return out


def hit_all_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """1 iff every gold session is in the top-k session list (multi-hop / conflict)."""
    if not gold_ids:
        return float("nan")
    return float(set(gold_ids) <= set(retrieved_ids[:k]))


def select_reader_hits(hits: list, k: int, qtype: str) -> list:
    """Unique sessions first (so two hops are not one session repeated as chunks)."""
    uniq = collapse_to_sessions(hits)
    take = uniq[: max(1, k)]
    if qtype in {"temporal-reasoning", "knowledge-update", "multi-session"}:
        take = sorted(
            take,
            key=lambda h: (h.doc.time_days is None, float(h.doc.time_days or 0.0)),
        )
    return take


def evidence_supports_answer(reader_hits: list, entry: dict, question: str) -> bool:
    """True iff a packed reader session asserts the asked value class. No gold ids."""
    from .memory_review import parse_target_slot, value_asserted_in_text

    tgt = parse_target_slot(question)
    id2sess = {
        str(sid): (str(date), sess)
        for sid, date, sess in zip(
            entry["haystack_session_ids"],
            entry["haystack_dates"],
            entry["haystack_sessions"],
        )
    }
    for h in reader_hits:
        sid = session_id_of(h)
        date, sess = id2sess.get(sid, ("", []))
        text = session_text(sess, date) if sess else (h.doc.text or "")
        if value_asserted_in_text(tgt, text, question):
            return True
    return False


def reader_should_abstain(
    *,
    qtype: str,
    abstention: bool,
    reader_hits: list,
    entry: dict,
    question: str,
    evo_sufficient: bool | None,
) -> bool:
    """Legacy hook — always False.

    Dense/evo QA must share the same Reader path: always call the LLM.
    Hard value_asserted gating inflated IDK and made evo QA incomparable to
    the older dense run. Evolution sufficiency still uses value_asserted in
    memory_review; that does not block answering.
    """
    return False


ABSTAIN_REPLY = "I do not know."


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        val = os.environ.get(name)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
    return default


class BGEEncoder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        self.model_name = model_name
        env_dev = _env_first("EMBEDDING_DEVICE").lower()
        self.device = device or env_dev or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.cache_dir = CACHE_DIR / hashlib.sha1(model_name.encode()).hexdigest()[:10]
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, text: str, is_query: bool) -> Path:
        h = hashlib.sha1(f"{self.model_name}|{int(is_query)}|{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.npy"

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool, batch_size: int = 16) -> np.ndarray:
        out = [None] * len(texts)
        pending_idx: list[int] = []
        pending_txt: list[str] = []
        for i, text in enumerate(texts):
            path = self._cache_key(text, is_query)
            if path.exists():
                out[i] = np.load(path)
            else:
                pending_idx.append(i)
                pending_txt.append(text)
        for start in range(0, len(pending_txt), batch_size):
            chunk = pending_txt[start : start + batch_size]
            idxs = pending_idx[start : start + batch_size]
            payload = [QUERY_PREFIX + t if is_query else t for t in chunk]
            tok = self.tokenizer(
                payload,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            hidden = self.model(**tok).last_hidden_state[:, 0]
            vecs = torch.nn.functional.normalize(hidden.float(), dim=-1).cpu().numpy()
            for j, vec in zip(idxs, vecs):
                out[j] = vec.astype(np.float32)
                np.save(self._cache_key(texts[j], is_query), out[j])
        return np.stack(out, axis=0)


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    """IR recall: |gold ∩ top-k| / |gold|. Abstention has no gold sessions."""
    if not gold_ids:
        return float("nan")
    gold = set(gold_ids)
    return float(sum(1 for i in retrieved_ids[:k] if i in gold)) / float(len(gold))


def ndcg_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return float("nan")
    gold = set(gold_ids)
    rels = [1.0 if i in gold else 0.0 for i in retrieved_ids[:k]]
    ideal = [1.0] * min(len(gold), k)

    def _dcg(vals: list[float]) -> float:
        return float(sum(rel / np.log2(idx + 2) for idx, rel in enumerate(vals)))

    idcg = _dcg(ideal)
    if idcg == 0.0:
        return 0.0
    return _dcg(rels) / idcg


def _mean(vals: list[float]) -> float | None:
    clean = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _fmt(val: float | None, digits: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "n/a"
    return f"{val:.{digits}f}"


def format_history(hits, raw_entry: dict, *, qtype: str = "", question_date: str = "") -> str:
    """Render full sessions for the reader. Lookup by session_id (chunks map back)."""
    id2session = {
        str(sid): (date, sess)
        for sid, date, sess in zip(
            raw_entry["haystack_session_ids"],
            raw_entry["haystack_dates"],
            raw_entry["haystack_sessions"],
        )
    }
    header = ""
    if qtype in {"temporal-reasoning", "knowledge-update", "multi-session"}:
        header = (
            "Sessions below are distinct haystack sessions, ordered by session date "
            "(oldest first). Current question date: "
            f"{question_date or 'unknown'}.\n"
            "For time questions, use Session Date to compute duration or which event is later. "
            "If two sessions disagree, prefer the later one. "
            "For multi-session questions, combine facts from more than one session.\n"
        )
    parts = [header] if header else []
    for i, hit in enumerate(hits, start=1):
        sid = session_id_of(hit)
        date, sess = id2session.get(sid, id2session.get(str(hit.doc.id), ("", [])))
        turns = []
        for turn in sess:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role") or "user"
            content = (turn.get("content") or "").strip()
            if content:
                turns.append(f"{role}: {content}")
        body = "\n".join(turns)
        if len(body) > 6000:
            body = body[:6000] + "\n[truncated]"
        parts.append(
            f"\n### Session {i} (session_id={sid}, score={hit.score:.4f})\n"
            f"Session Date: {date}\nSession Content:\n{body}\n"
        )
    return "".join(parts)


_JUDGE_FORMAT = (
    "Scoring rules: compare the model response to the correct answer (or rubric). "
    "Credit a response only if it states the required fact, an equivalent phrasing, "
    "or the necessary intermediate steps. "
    "If the model says it does not know, or gives a partial / different fact, the label is No. "
    "Ignore wording in this instruction itself. "
    "Reply with a single first line that is exactly Yes or exactly No. No other words on that line."
)


def judge_prompt(task: str, question: str, answer: str, response: str, abstention: bool) -> str:
    if abstention:
        return (
            f"{_JUDGE_FORMAT}\n\n"
            "The question is unanswerable. Label Yes only if the model identifies it as unanswerable "
            "(incomplete information, or related but not the asked fact).\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\nModel Response: {response}\n"
        )
    if task == "temporal-reasoning":
        extra = (
            " Do not penalize off-by-one errors for days/weeks/months "
            "(e.g. 19 days vs 18 days is still Yes)."
        )
        return (
            f"{_JUDGE_FORMAT}{extra}\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n"
        )
    if task == "knowledge-update":
        return (
            f"{_JUDGE_FORMAT} "
            "If the response mentions outdated facts but also the required updated answer, label Yes.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n"
        )
    if task == "single-session-preference":
        return (
            f"{_JUDGE_FORMAT} "
            "The response need not cover every rubric bullet; Yes if it uses the user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\nModel Response: {response}\n"
        )
    return (
        f"{_JUDGE_FORMAT}\n\n"
        f"Question: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {response}\n"
    )


def llm_client(api_key: str, base_url: str):
    from openai import OpenAI

    timeout = float(_env_first("LLM_TIMEOUT", default="180"))
    return OpenAI(api_key=api_key or "-", base_url=base_url, timeout=timeout)


_THINK_MARK = re.compile(
    r"\n(?:We need |Let's |Let me |I need to |Need (?:answer|follow|determine|obey|comply))",
    re.I,
)


def _split_message(completion) -> tuple[str, str]:
    choice = completion.choices[0]
    msg = choice.message
    extra = getattr(msg, "model_extra", None) or {}
    extra = extra if isinstance(extra, dict) else {}

    def _get(key: str) -> str:
        val = getattr(msg, key, None)
        if val and str(val).strip():
            return str(val).strip()
        val = extra.get(key)
        if val and str(val).strip():
            return str(val).strip()
        return ""

    return _get("content"), _get("reasoning_content") or _get("reasoning")


def clip_generation(text: str) -> str:
    """Drop leaked planning after a real answer. Dense-only runs rarely have this dump."""
    text = (text or "").strip()
    if not text:
        return ""
    m = _THINK_MARK.search(text)
    if m and m.start() >= 24:
        return text[: m.start()].strip()
    return text


def _last_correct_json(text: str) -> str | None:
    hits = list(_JSON_CORRECT.finditer(text or ""))
    if not hits:
        return None
    val = hits[-1].group(1).lower() == "true"
    return '{"correct": true}' if val else '{"correct": false}'


def _message_text(completion, include_reasoning: bool = True) -> str:
    """Prefer visible content. Judge/reader must not concatenate chain-of-thought."""
    content, reasoning = _split_message(completion)
    if not include_reasoning:
        return content or reasoning
    if content and reasoning and reasoning not in content:
        return f"{content}\n{reasoning}"
    return content or reasoning


def chat(
    client,
    model: str,
    prompt: str,
    max_tokens: int,
    thinking: bool,
    official_deepseek: bool,
    include_reasoning: bool = True,
    *,
    system: str | None = None,
    mode: str = "default",
) -> str:
    extras: list[dict | None]
    if official_deepseek:
        extras = [{"thinking": {"type": "enabled" if thinking else "disabled"}}]
    else:
        extras = [
            {"thinking": {"type": "disabled"}},
            {"chat_template_kwargs": {"enable_thinking": False}},
            {"enable_thinking": False},
            None,
        ]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    last_err: Exception | None = None
    for extra in extras:
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if extra is not None:
            kwargs["extra_body"] = extra
        for attempt in range(5):
            try:
                completion = client.chat.completions.create(**kwargs)
                content, reasoning = _split_message(completion)
                if mode == "judge":
                    for blob in (content, reasoning, f"{content}\n{reasoning}"):
                        labs = exact_judge_labels(blob)
                        if labs:
                            return "Yes" if labs[-1] else "No"
                    text = (content or reasoning or "").strip()
                    if text:
                        return text
                    last_err = RuntimeError("empty LLM content")
                    continue
                if mode == "reader":
                    text = clip_generation(content or reasoning)
                    if text:
                        return text
                    last_err = RuntimeError("empty LLM content")
                    continue
                if mode == "json":
                    for blob in (content, reasoning, f"{content}\n{reasoning}"):
                        packed = _last_correct_json(blob or "")
                        if packed:
                            return packed
                    text = content or reasoning or ""
                    if text:
                        return text
                    last_err = RuntimeError("empty LLM content")
                    continue
                text = _message_text(completion, include_reasoning=include_reasoning)
                if text:
                    return text
                last_err = RuntimeError("empty LLM content")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                kwargs.pop("stop", None)
                msg = str(exc).lower()
                # Timeouts / gateway blips: back off longer before next try.
                if "timeout" in msg or "timed out" in msg or "temporar" in msg:
                    time.sleep(min(8 * (attempt + 1), 40))
                else:
                    time.sleep(min(2**attempt, 12))
    if mode == "json":
        # Auditor may fall back to heuristics; do not kill the whole eval.
        return ""
    if mode == "reader":
        return "I do not know."
    raise RuntimeError(f"LLM call failed: {last_err}")


_PROMPT_ECHO = re.compile(
    r"answer yes if|please answer yes|yes\s*/\s*no|yes or no|output format:|"
    r"scoring rules:|first line that is exactly|exactly yes or exactly no",
    re.I,
)
_EXACT_LABEL = re.compile(r"^(?:label\s*[:\-]\s*)?(yes|no)$", re.I)


def exact_judge_labels(verdict: str) -> list[bool]:
    """Only standalone Yes/No lines count. Instruction echoes are ignored."""
    out: list[bool] = []
    for raw in (verdict or "").splitlines():
        line = re.sub(r"[*`#_]", "", raw).strip().rstrip(".!").strip()
        if not line or _PROMPT_ECHO.search(line):
            continue
        m = _EXACT_LABEL.match(line)
        if m:
            out.append(m.group(1).lower() == "yes")
    return out


_JSON_CORRECT = re.compile(r'"correct"\s*:\s*(true|false)', re.I)


def parse_judge_label(verdict: str) -> bool:
    packed = _last_correct_json(verdict or "")
    if packed:
        return "true" in packed
    labels = exact_judge_labels(verdict)
    return bool(labels[-1]) if labels else False


def run_judge(client, model: str, official_deepseek: bool, prompt: str, question: str, answer: str, response: str) -> str:
    clipped = clip_generation(response)
    sys_msg = (
        'You are a binary grader. Do not write planning notes. '
        'Reply with JSON only: {"correct": true} or {"correct": false}.'
    )
    body = (
        f"Question: {question}\nCorrect Answer: {answer}\n"
        f"Model Response: {clipped[:2000]}\n"
        'Is the model response correct? JSON: {"correct": true|false}'
    )
    text = chat(
        client,
        model,
        body,
        max_tokens=256,
        thinking=False,
        official_deepseek=official_deepseek,
        include_reasoning=False,
        system=sys_msg,
        mode="json",
    )
    if _JSON_CORRECT.search(text or "") or exact_judge_labels(text):
        return text
    text2 = chat(
        client,
        model,
        body,
        max_tokens=256,
        thinking=False,
        official_deepseek=official_deepseek,
        include_reasoning=True,
        system=sys_msg,
        mode="json",
    )
    return text2 or text


def already_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["question_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def summarize(rows: list[dict]) -> dict:
    """Official LongMemEval scores: 6-type QA + abstention, Recall/NDCG@5/10."""
    type2acc: dict[str, list[int]] = {t: [] for t in PAPER_QA_TYPES}
    abstention_acc: list[int] = []
    all_acc: list[int] = []
    rec5: list[float] = []
    rec10: list[float] = []
    ndcg5: list[float] = []
    ndcg10: list[float] = []
    hit5: list[float] = []
    hit10: list[float] = []
    rec_by_type: dict[str, list[float]] = defaultdict(list)
    hit10_by_type: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        qtype = r.get("question_type")
        if r.get("qa_correct") is not None:
            lab = int(bool(r["qa_correct"]))
            all_acc.append(lab)
            if qtype in type2acc:
                type2acc[qtype].append(lab)
            if r.get("abstention") or "_abs" in str(r.get("question_id", "")):
                abstention_acc.append(lab)
        if r.get("abstention"):
            continue
        if r.get("recall@5") is not None and not np.isnan(float(r["recall@5"])):
            rec5.append(float(r["recall@5"]))
            rec_by_type[qtype].append(float(r["recall@5"]))
        if r.get("recall@10") is not None and not np.isnan(float(r["recall@10"])):
            rec10.append(float(r["recall@10"]))
        if r.get("ndcg@5") is not None and not np.isnan(float(r["ndcg@5"])):
            ndcg5.append(float(r["ndcg@5"]))
        if r.get("ndcg@10") is not None and not np.isnan(float(r["ndcg@10"])):
            ndcg10.append(float(r["ndcg@10"]))
        if r.get("hit_all@5") is not None and not np.isnan(float(r["hit_all@5"])):
            hit5.append(float(r["hit_all@5"]))
        if r.get("hit_all@10") is not None and not np.isnan(float(r["hit_all@10"])):
            hit10.append(float(r["hit_all@10"]))
            hit10_by_type[qtype].append(float(r["hit_all@10"]))

    task_means = [_mean(type2acc[t]) for t in PAPER_QA_TYPES if type2acc[t]]
    qa_by_type = {}
    for t in PAPER_QA_TYPES:
        if type2acc[t]:
            qa_by_type[t] = {"accuracy": round(float(np.mean(type2acc[t])), 4), "n": len(type2acc[t])}
    return {
        "n": len(rows),
        "overall_accuracy": None if not all_acc else round(float(np.mean(all_acc)), 4),
        "task_averaged_accuracy": None if not task_means else round(float(np.mean(task_means)), 4),
        "abstention_accuracy": None if not abstention_acc else round(float(np.mean(abstention_acc)), 4),
        "abstention_n": len(abstention_acc),
        "qa_by_type": qa_by_type,
        "retrieval": {
            "granularity": "session",
            "metrics@10": {
                "recall": None if not rec10 else round(float(np.mean(rec10)), 4),
                "ndcg": None if not ndcg10 else round(float(np.mean(ndcg10)), 4),
                "hit_all": None if not hit10 else round(float(np.mean(hit10)), 4),
                "n": len(rec10),
            },
            "metrics@5": {
                "recall": None if not rec5 else round(float(np.mean(rec5)), 4),
                "ndcg": None if not ndcg5 else round(float(np.mean(ndcg5)), 4),
                "hit_all": None if not hit5 else round(float(np.mean(hit5)), 4),
                "n": len(rec5),
            },
        },
        "recall@5_by_type": {
            k: round(float(np.mean(v)), 4) for k, v in sorted(rec_by_type.items()) if v
        },
        "hit_all@10_by_type": {
            k: round(float(np.mean(v)), 4) for k, v in sorted(hit10_by_type.items()) if v
        },
    }


def print_paper_report(metrics: dict, split: str, top_k: int) -> None:
    name = {"s": "LongMemEval-S", "m": "LongMemEval-M", "oracle": "LongMemEval-Oracle"}.get(split, split)
    ret = metrics.get("retrieval") or {}
    m5 = ret.get("metrics@5") or {}
    m10 = ret.get("metrics@10") or {}
    print()
    print("=" * 72)
    print(f"{name}  |  Value = Session  (paper Table 3 / official QA script)")
    print("=" * 72)
    print()
    print("Retrieval")
    print(f"  {'':<12}{'Recall':>10}{'NDCG':>10}{'n':>8}")
    print(f"  {'Metrics@5':<12}{_fmt(m5.get('recall'), 3):>10}{_fmt(m5.get('ndcg'), 3):>10}{str(m5.get('n', 0)):>8}")
    print(f"  {'Metrics@10':<12}{_fmt(m10.get('recall'), 3):>10}{_fmt(m10.get('ndcg'), 3):>10}{str(m10.get('n', 0)):>8}")
    print(f"  Hit-all@5 (every gold session in Top-5):  {_fmt(m5.get('hit_all'), 3)}")
    print(f"  Hit-all@10 (every gold session in Top-10): {_fmt(m10.get('hit_all'), 3)}")
    by_hit = metrics.get("hit_all@10_by_type") or {}
    if by_hit:
        print("  Hit-all@10 by type:")
        for t in PAPER_QA_TYPES:
            if t in by_hit:
                print(f"    {t}: {by_hit[t]}")
    print()
    print(f"End-to-End QA  (reader uses Top-{top_k})")
    print(f"  Overall Accuracy:        {_fmt(metrics.get('overall_accuracy'))}")
    print(f"  Task-averaged Accuracy:  {_fmt(metrics.get('task_averaged_accuracy'))}")
    print(
        f"  Abstention Accuracy:     {_fmt(metrics.get('abstention_accuracy'))} "
        f"({metrics.get('abstention_n', 0)})"
    )
    print()
    print("Evaluation results by task:")
    by_type = metrics.get("qa_by_type") or {}
    for k in PAPER_QA_TYPES:
        item = by_type.get(k)
        if not item:
            print(f"\t{k}: n/a (0)")
            continue
        print(f"\t{k}: {item['accuracy']} ({item['n']})")
    print("=" * 72)


def run(args: argparse.Namespace) -> int:
    _load_dotenv()
    api_key = args.api_key or _env_first("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", default="")
    base_url = args.base_url or _env_first("LLM_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_BASE", default="")
    model = args.model or _env_first("LLM_MODEL", default="deepseek-v4-free")
    if not base_url:
        base_url = "https://api.deepseek.com"
    official_deepseek = "api.deepseek.com" in base_url
    do_llm = not args.retrieval_only
    if do_llm and not api_key and official_deepseek:
        print("LLM_API_KEY / DEEPSEEK_API_KEY is missing. Running retrieval-only.")
        do_llm = False
    if do_llm and not api_key:
        api_key = "-"
    client = llm_client(api_key, base_url) if do_llm else None
    if do_llm:
        print(f"LLM model={model} base={base_url}")

    path = download_split(args.split)
    instances = load_instances(path)
    if args.question_types:
        keep = set(args.question_types)
        instances = [e for e in instances if e.get("question_type") in keep]
        print(f"question-types filter {sorted(keep)} -> {len(instances)} instances")
    if args.limit:
        instances = instances[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_k{args.top_k}_{'evo' if args.evolve else 'dense'}"
    extra = (getattr(args, "out_tag", None) or "").strip()
    if not extra and args.question_types:
        extra = "filt"
    if extra:
        tag = f"{tag}_{extra}"
    hyp_path = OUT_DIR / f"{tag}.jsonl"
    pred_path = OUT_DIR / f"{tag}.hypotheses.jsonl"
    metrics_path = OUT_DIR / f"{tag}.metrics.json"
    done = already_done(hyp_path) if args.resume else set()
    if not args.resume and hyp_path.exists():
        hyp_path.unlink()
        if pred_path.exists():
            pred_path.unlink()

    if args.resume:
        print("WARNING: --resume keeps old RANK-only rows. Prefer deleting the jsonl and starting clean.")
    print(
        "Eval protocol: session-level IR after collapsing chunks; "
        "Hit-all@k requires every gold session; reader packs unique sessions "
        "(time-ordered for temporal / KU / multi-session)."
    )
    encoder = BGEEncoder(args.encoder)
    print(f"Encoder {args.encoder} on {encoder.device}… first load may download weights")
    rows: list[dict] = []
    if args.resume and hyp_path.exists():
        rows = [json.loads(l) for l in hyp_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    for n, entry in enumerate(instances, start=1):
        qid = entry["question_id"]
        if qid in done:
            continue
        qtype = entry["question_type"]
        question = entry["question"]
        qdate = entry.get("question_date") or ""
        gold_sessions = [str(x) for x in entry.get("answer_session_ids") or []]
        abstention = "_abs" in qid

        docs = build_chunk_documents(entry, encoder)
        q_text = f"Current date: {qdate}\n{question}"
        q_vec = encoder.encode([q_text], is_query=True)[0]
        retrieve_k = max(int(args.top_k) * 3, int(args.top_k))

        if args.evolve:
            result = run_evolution(
                q_vec,
                docs,
                max_rounds=args.max_rounds,
                top_k=retrieve_k,
                seed=n,
                query_time_days=parse_lme_date(qdate),
                question=q_text,
                encode_query=lambda text, _enc=encoder: _enc.encode([text], is_query=True)[0],
                question_type=str(qtype or ""),
                llm_complete=(
                    (lambda prompt, _c=client, _m=model, _od=official_deepseek: chat(
                        _c,
                        _m,
                        prompt,
                        max_tokens=512,
                        thinking=False,
                        official_deepseek=_od,
                        include_reasoning=False,
                        system=AUDITOR_SYSTEM,
                        mode="json",
                    ))
                    if do_llm and client is not None
                    else None
                ),
            )
            hits = result.evidence
            evo_sufficient = bool(result.sufficient)
            evo_log = [
                {
                    "round": log.round_index,
                    "gap": log.gap.kind,
                    "slot": log.slot_updated,
                    "note": log.note,
                    "missing_keys": list(log.missing_keys),
                    "top": log.top_ids[:5],
                    "top3": log.top_ids[:3],
                }
                for log in result.rounds
            ]
            from .slots import parse_skill

            snap = parse_skill(result.working_skill_text or "") if result.working_skill_text else None
            skill_query_slot = snap.query_body if snap else None
            skill_rank_slot = snap.rank_body if snap else None
            skill_filter_slot = snap.filter_body if snap else None
            r1 = evo_log[0] if evo_log else {}
            note0 = str(r1.get("note") or "")
            old_proc = (
                evo_log
                and "missing_keys=" not in note0
                and not note0.startswith("attr-llm:")
                and "missing=" not in note0
            )
            if old_proc:
                print("ABORT: old eval process — note has no missing_keys=. Kill the old python job and rerun without --resume.")
                return 2
        else:
            r1 = {}
            from .retrieve import dense_retrieve
            from .types import EvolutionState

            hits = dense_retrieve(docs, EvolutionState(q0=q_vec, q=q_vec), top_k=retrieve_k)
            evo_log = []
            skill_query_slot = None
            skill_rank_slot = None
            skill_filter_slot = None
            evo_sufficient = None

        session_hits = collapse_to_sessions(hits, limit=int(args.top_k))
        retrieved_ids = [session_id_of(h) for h in session_hits]
        rec5 = None if abstention else recall_at_k(retrieved_ids, gold_sessions, 5)
        rec10 = None if abstention else recall_at_k(retrieved_ids, gold_sessions, 10)
        n5 = None if abstention else ndcg_at_k(retrieved_ids, gold_sessions, 5)
        n10 = None if abstention else ndcg_at_k(retrieved_ids, gold_sessions, 10)
        h5 = None if abstention else hit_all_at_k(retrieved_ids, gold_sessions, 5)
        h10 = None if abstention else hit_all_at_k(retrieved_ids, gold_sessions, 10)

        hypothesis = ""
        qa_correct = None
        judge_verdict = ""
        reader_abstained = False
        if do_llm and client is not None:
            reader_k = max(1, int(args.reader_k))
            reader_hits = select_reader_hits(hits, reader_k, str(qtype or ""))
            history = format_history(
                reader_hits, entry, qtype=str(qtype or ""), question_date=qdate
            )
            # Same Reader path for dense and evolve: no hard abstain gate.
            prompt = READER_TEMPLATE.format(history=history, date=qdate, question=question)
            hypothesis = chat(
                client,
                model,
                prompt,
                max_tokens=1200,
                thinking=False,
                official_deepseek=official_deepseek,
                include_reasoning=False,
                system=READER_SYSTEM,
                mode="reader",
            )
            gold_answer = entry.get("answer") or ""
            jp = judge_prompt(qtype, question, gold_answer, hypothesis, abstention)
            judge_verdict = run_judge(
                client, model, official_deepseek, jp, question, gold_answer, hypothesis
            )
            qa_correct = parse_judge_label(judge_verdict)

        row = {
            "question_id": qid,
            "question_type": qtype,
            "question": question,
            "hypothesis": hypothesis,
            "qa_correct": qa_correct,
            "judge_verdict": judge_verdict,
            "autoeval_label": None if qa_correct is None else {"model": model, "label": bool(qa_correct)},
            "recall@5": rec5,
            "recall@10": rec10,
            "ndcg@5": n5,
            "ndcg@10": n10,
            "hit_all@5": h5,
            "hit_all@10": h10,
            "retrieved_ids": retrieved_ids,
            "gold_session_ids": gold_sessions,
            "evolution": evo_log,
            "skill_query_slot": skill_query_slot,
            "skill_rank_slot": skill_rank_slot,
            "skill_filter_slot": skill_filter_slot,
            "abstention": abstention,
            "evo_sufficient": evo_sufficient,
            "reader_abstained": reader_abstained,
        }
        rows.append(row)
        with hyp_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if hypothesis:
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"question_id": qid, "hypothesis": hypothesis}, ensure_ascii=False) + "\n")
        rec_s = "n/a" if rec10 is None else f"{rec10:.3f}"
        abs_s = " abs" if reader_abstained else ""
        suf_s = "" if evo_sufficient is None else f" evo_suf={evo_sufficient}"
        print(
            f"[{n}/{len(instances)}] {qid} type={qtype} R@10={rec_s} qa={qa_correct}{abs_s}{suf_s} top={retrieved_ids[:3]}",
            flush=True,
        )
        if args.evolve and r1:
            print(
                f"  CHECK {qid} r1 slot={r1.get('slot')} missing_keys={r1.get('missing_keys')} note={r1.get('note')}",
                flush=True,
            )

    metrics = summarize(rows)
    metrics.update(
        {
            "split": args.split,
            "evolve": args.evolve,
            "top_k": args.top_k,
            "model": model if do_llm else None,
            "base_url": base_url if do_llm else None,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print_paper_report(metrics, args.split, args.top_k)
    print(f"Saved {hyp_path}")
    print(f"Saved {metrics_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run LongMemEval with skill-evolution dense retrieval.")
    p.add_argument("--split", choices=list(SPLITS), default="s")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--encoder", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--model", default=None, help="Default: LLM_MODEL or deepseek-v4-free")
    p.add_argument("--base-url", default=None, help="Default: LLM_API_BASE")
    p.add_argument("--api-key", default=None, help="Default: LLM_API_KEY (intranet can be -)")
    p.add_argument("--evolve", action="store_true", default=True)
    p.add_argument("--no-evolve", action="store_false", dest="evolve")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--reader-k", type=int, default=10, help="Sessions fed to the reader (use 3 to isolate Top-3).")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--question-types", nargs="*", default=None)
    p.add_argument(
        "--out-tag",
        default=None,
        help="Suffix for jsonl/metrics so a type-filtered run does not overwrite the full 500-file.",
    )
    p.add_argument(
        "--relabel",
        action="store_true",
        help="Re-parse qa_correct on an existing jsonl with the strict judge parser (no LLM).",
    )
    p.add_argument("--jsonl", default=None, help="With --relabel: path to hypotheses jsonl.")
    return p


def relabel_jsonl(path: Path) -> int:
    if not path.exists():
        print(f"Missing jsonl: {path}")
        return 1
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n_flip = 0
    for r in rows:
        old = r.get("qa_correct")
        new = parse_judge_label(r.get("judge_verdict") or "")
        r["qa_correct_old"] = old
        r["qa_correct"] = new
        auto = r.get("autoeval_label")
        if isinstance(auto, dict):
            auto["label"] = bool(new)
            auto["parser"] = "strict_yes_no"
        if bool(old) != bool(new):
            n_flip += 1
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    metrics = summarize(rows)
    metrics_path = path.with_suffix(".metrics.json")
    if metrics_path.name.endswith(".jsonl.metrics.json"):
        metrics_path = path.with_name(path.stem + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Relabeled {len(rows)} rows, flipped {n_flip} QA labels -> {path}")
    print_paper_report(metrics, str(metrics.get("split") or ""), 10)
    print(f"Saved {metrics_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.relabel:
        jsonl = Path(args.jsonl) if args.jsonl else OUT_DIR / f"{args.split}_k{args.top_k}_evo.jsonl"
        return relabel_jsonl(jsonl)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
