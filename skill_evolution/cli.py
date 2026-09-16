from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .fixtures import alias_fixture, temporal_fixture
from .loop import run_evolution
from .slots import parse_skill
from .workspace import SkillWorkspace


def _safe(s: object) -> str:
    text = str(s)
    try:
        text.encode(sys.stdout.encoding or "utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", "backslashreplace").decode("ascii")


def _print_result(title: str, query: str, result) -> None:
    print(f"\n=== {title} ===")
    print(_safe(f"query: {query}"))
    print(f"sufficient: {result.sufficient}")
    for log in result.rounds:
        print(
            _safe(
                f"  round {log.round_index}: gap={log.gap.kind} "
                f"g_time={log.gap.g_time:.3f} g_alias={log.gap.g_alias:.3f} "
                f"slot={log.slot_updated} ok={log.accepted} top={log.top_ids[:4]} ({log.note})"
            )
        )
    if result.evidence:
        print("  final ranking:")
        for h in result.evidence[:5]:
            print(f"    {h.doc.id:14s}  score={h.score:.4f}  sim={h.sim:.4f}")
    if result.working_skill_text:
        snap = parse_skill(result.working_skill_text)
        print(_safe("  QUERY slot:\n" + snap.query_body))
        print(_safe("  RANK slot:\n" + snap.rank_body))


def self_test() -> int:
    q, docs, text = temporal_fixture()
    result = run_evolution(q, docs, seed=0, top_k=8)
    _print_result("temporal", text, result)
    final_ids = [h.doc.id for h in result.evidence]
    if "policy_2024" not in final_ids[:2]:
        print("FAIL: newest policy should rise to top-2 after rank fusion", file=sys.stderr)
        return 1
    compiled = result.working_skill_text or ""
    for doc in docs:
        if doc.text[:20] in compiled:
            print("FAIL: passage leaked into skill", file=sys.stderr)
            return 1
    if "score(d) = (1−λ)" not in compiled and "score(d) = (1-λ)" not in compiled:
        # Chinese minus sign in compile_skill
        if "线性融合" not in compiled:
            print("FAIL: rank formula was not compiled into the working skill", file=sys.stderr)
            return 1

    q2, docs2, text2 = alias_fixture()
    result2 = run_evolution(q2, docs2, seed=1, top_k=8)
    _print_result("alias", text2, result2)
    if not result2.rounds or result2.rounds[0].gap.kind != "alias":
        print("FAIL: mixed-sense query should open the alias gap", file=sys.stderr)
        return 1
    top = [h.doc.id for h in result2.evidence[:3]]
    if not all(i.startswith("fruit_") for i in top):
        print("FAIL: Rocchio should rank the nearer sense above the other", file=sys.stderr)
        return 1
    if "Rocchio" not in (result2.working_skill_text or ""):
        print("FAIL: alias path should compile the query slot", file=sys.stderr)
        return 1

    # Lifecycle: session directory must not remain when persist is off.
    skill = Path(__file__).resolve().parent.parent / "dense-vector-retrieval" / "SKILL.md"
    ws = SkillWorkspace.open(skill)
    path = ws.session_dir
    ws.destroy()
    if path.exists():
        print("FAIL: workspace not destroyed", file=sys.stderr)
        return 1
    print("\nSELF-TEST OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evolve a skill copy with Rocchio + recency fusion.")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--demo", choices=["temporal", "alias"], default="temporal")
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    fixture = temporal_fixture if args.demo == "temporal" else alias_fixture
    q, docs, text = fixture()
    result = run_evolution(q, docs, seed=0 if args.demo == "temporal" else 1, top_k=8)
    _print_result(args.demo, text, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
