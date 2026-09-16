from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

GapKind = Literal["time", "alias", "unknown", "unanswerable", "sufficient"]
SlotName = Literal["query", "rank", "filter"]


@dataclass
class Document:
    id: str
    text: str
    embedding: np.ndarray
    # Structured timestamp in days (e.g. datetime.toordinal()). None if unknown.
    time_days: float | None = None
    session_id: str = ""


@dataclass
class Hit:
    doc: Document
    sim: float
    score: float


@dataclass
class GapReport:
    kind: GapKind
    sufficient: bool
    g_time: float
    g_alias: float
    g_unanswerable: float
    n_eff_time: float
    n_eff_alias: float
    reason: str


@dataclass
class Signals:
    t_star: float | None
    sigma: float | None
    pos_ids: tuple[str, ...]
    neg_ids: tuple[str, ...]
    pos_mean: np.ndarray | None
    neg_mean: np.ndarray | None
    n_eff: float
    kappa: dict[str, float] = field(default_factory=dict)


@dataclass
class EvolutionState:
    """Numeric policy compiled into the two skill slots. No passage text."""

    q0: np.ndarray
    q: np.ndarray
    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0
    lambda_recency: float = 0.0
    eta_neg: float = 0.0
    t_star: float | None = None
    sigma: float | None = None
    pos_ids: tuple[str, ...] = ()
    neg_ids: tuple[str, ...] = ()
    query_slot_active: bool = False
    rank_slot_active: bool = False
    query_focus: str = ""
    eta_hard: float = 0.0
    hard_neg_ids: tuple[str, ...] = ()
    query_locked: bool = False
    focus_blacklist: tuple[str, ...] = ()
    v_slot: np.ndarray | None = None
    omega_hop: float = 0.0
    mu_noise: np.ndarray | None = None
    filter_slot_active: bool = False
    filter_bind: tuple[str, ...] = ()
    filter_drop_ids: tuple[str, ...] = ()
    filter_eta: float = 0.0
    filter_require_bind: bool = False
    div_ids: tuple[str, ...] = ()
    delta_div: float = 0.0
    rank_task: str = ""


@dataclass
class RoundLog:
    round_index: int
    gap: GapReport
    slot_updated: SlotName | None
    accepted: bool
    top_ids: list[str]
    note: str
    missing_keys: tuple[str, ...] = ()
