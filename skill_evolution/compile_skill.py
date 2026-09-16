"""Project EvolutionState into QUERY / RANK / FILTER slots as formulas — never passages."""

from __future__ import annotations

from .types import EvolutionState


def query_slot_body(state: EvolutionState) -> str:
    lines: list[str] = []
    if state.query_focus.strip():
        lines.append("memory-informed query focus（关键词，禁止粘贴会话正文）")
        lines.append(f"focus: {state.query_focus.strip()[:120]}")
        lines.append(
            "slate: S(d)=max(sim(q0,d), ω sim(v_slot,d)) − γ sim(μ_noise,d) "
            "(only if high noise-sim and low q0-sim); q stays q0"
        )
        lines.append(
            f"ω={getattr(state, 'omega_hop', 0):.4f} γ={state.gamma:.4f} "
            f"α={state.alpha:.4f} β_miss={state.beta:.4f}"
        )
    if state.query_slot_active and (state.beta != 0.0 or state.gamma != 0.0):
        lines.append("Rocchio（向量空间，禁止文本拼接）")
        lines.append("q' = normalize(α q + β μ+ − γ μ−)")
        lines.append(f"α={state.alpha:.4f} β={state.beta:.4f} γ={state.gamma:.4f}")
        lines.append(f"|M+|={len(state.pos_ids)} |M−|={len(state.neg_ids)}（仅计数与 id 指纹，无正文）")
    if state.focus_blacklist:
        shown = ", ".join(x[:40] for x in state.focus_blacklist if x)[:120]
        lines.append(f"dead-end focus blacklist: {shown}")
    if not lines:
        return "恒等：q' = q"
    return "\n".join(lines)


def filter_slot_body(state: EvolutionState) -> str:
    if not getattr(state, "filter_slot_active", False) or (
        not state.filter_drop_ids and not state.filter_require_bind and state.filter_eta == 0.0
    ):
        return "恒等：keep(d) = true"
    bind = ",".join(state.filter_bind[:8]) or "-"
    n_drop = len(state.filter_drop_ids)
    shown = ",".join(state.filter_drop_ids[:8])
    if len(state.filter_drop_ids) > 8:
        shown = shown + ",..."
    return (
        "约束过滤（仅 bind 词与 id 指纹，禁止会话正文）\n"
        "score(d) ← score(d) − η_f if d ∈ D_drop or (require_bind ∧ bind ∩ d = ∅)\n"
        f"require_bind={int(state.filter_require_bind)} η_f={state.filter_eta:.4f} "
        f"|D_drop|={n_drop} bind={bind}\n"
        f"D_drop ids: {shown or '-'}"
    )


def write_state_slots(path, state: EvolutionState) -> None:
    from .slots import write_slots

    write_slots(path, query_slot_body(state), rank_slot_body(state), filter_slot_body(state))


def rank_slot_body(state: EvolutionState) -> str:
    if not state.rank_slot_active or (
        state.lambda_recency == 0.0
        and state.eta_neg == 0.0
        and state.eta_hard == 0.0
        and not getattr(state, "div_ids", ())
    ):
        return "恒等：score(d) = sim(q, d)"
    t_star = "na" if state.t_star is None else f"{state.t_star:.2f}"
    sigma = "na" if state.sigma is None else f"{state.sigma:.2f}"
    extra = ""
    if getattr(state, "div_ids", ()):
        extra = (
            f"\nmulti-hop: +δ 1[d ∈ D_div] δ={state.delta_div:.4f} |D_div|={len(state.div_ids)}"
        )
    return (
        "线性融合（dense 内核不变；时序 κ / 多跳互补簇）\n"
        "score(d) = (1−λ) sim(q', d) + λ κ(t_d) − η_s 1[d ∈ M_soft−] − η_h 1[d ∈ M_hard−] "
        "+ δ 1[d ∈ D_div]\n"
        "κ(t) = exp(−max(0, t* − t) / σ)\n"
        f"λ={state.lambda_recency:.4f} η_s={state.eta_neg:.4f} η_h={state.eta_hard:.4f} "
        f"t*={t_star} σ={sigma}\n"
        f"|M_soft−|={len(state.neg_ids)} |M_hard−|={len(state.hard_neg_ids)}"
        f"{extra}"
    )
