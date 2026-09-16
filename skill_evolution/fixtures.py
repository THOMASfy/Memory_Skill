"""Synthetic corpus: same topic vectors, time only in metadata — dense sim cannot break ties."""

from __future__ import annotations

from datetime import date

import numpy as np

from .mathutil import l2_normalize
from .types import Document


def _days(y: int, m: int, d: int) -> float:
    return float(date(y, m, d).toordinal())


def temporal_fixture(rng: np.random.Generator | None = None) -> tuple[np.ndarray, list[Document], str]:
    """Query: current travel reimbursement cap. Old docs are slightly closer in sim."""
    rng = rng or np.random.default_rng(0)
    dim = 16
    travel = l2_normalize(np.concatenate([np.ones(8), np.zeros(8)]))
    it_topic = l2_normalize(np.concatenate([np.zeros(8), np.ones(8)]))

    def travel_doc(doc_id: str, year: int, cap: int, closer: bool) -> Document:
        noise = rng.normal(0, 0.02, size=dim)
        # Stale docs get a tiny extra alignment with the query so round-1 prefers them.
        bump = 0.08 * travel if closer else 0.0
        vec = l2_normalize(travel + bump + noise)
        text = (
            f"差旅报销管理办法（{year}年修订）：员工市内交通及住宿报销上限为{cap}元/天。"
            "本条仅用于评测，禁止写入 skill。"
        )
        return Document(id=doc_id, text=text, embedding=vec.astype(np.float32), time_days=_days(year, 6, 1))

    docs = [
        travel_doc("policy_2018", 2018, 3000, closer=True),
        travel_doc("policy_2019", 2019, 3000, closer=True),
        travel_doc("policy_2020", 2020, 4000, closer=True),
        travel_doc("policy_2021", 2021, 4000, closer=False),
        travel_doc("policy_2022", 2022, 5000, closer=False),
        travel_doc("policy_2024", 2024, 8000, closer=False),
        Document(
            id="it_vpn",
            text="VPN 客户端升级说明：办公网接入须使用公司签发证书。",
            embedding=l2_normalize(it_topic + rng.normal(0, 0.02, size=dim)).astype(np.float32),
            time_days=_days(2024, 1, 1),
        ),
        Document(
            id="it_laptop",
            text="笔记本电脑领用规范：入职领取、离职归还。",
            embedding=l2_normalize(it_topic + rng.normal(0, 0.03, size=dim)).astype(np.float32),
            time_days=_days(2023, 5, 1),
        ),
    ]
    query = l2_normalize(travel + 0.01 * rng.normal(0, 1, size=dim)).astype(np.float32)
    return query, docs, "现在公司差旅报销上限是多少？"


def alias_fixture(rng: np.random.Generator | None = None) -> tuple[np.ndarray, list[Document], str]:
    rng = rng or np.random.default_rng(1)
    dim = 16
    fruit = np.zeros(dim)
    fruit[:8] = 1.0
    company = np.zeros(dim)
    company[8:] = 1.0
    fruit = l2_normalize(fruit)
    company = l2_normalize(company)
    # Query sits between senses, slightly closer to fruit.
    query = l2_normalize(0.52 * fruit + 0.48 * company).astype(np.float32)
    docs: list[Document] = []
    for i in range(4):
        docs.append(
            Document(
                id=f"fruit_{i}",
                text=f"苹果（水果）品种介绍 {i}：红富士口感甜脆。",
                embedding=l2_normalize(fruit + rng.normal(0, 0.03, size=dim)).astype(np.float32),
                time_days=None,
            )
        )
    for i in range(4):
        docs.append(
            Document(
                id=f"co_{i}",
                text=f"Apple Inc 财报摘要 {i}：iPhone 销量与服务收入。",
                embedding=l2_normalize(company + rng.normal(0, 0.03, size=dim)).astype(np.float32),
                time_days=None,
            )
        )
    return query, docs, "apple 最近怎么样"
