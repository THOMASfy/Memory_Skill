"""Closed-form helpers used by diagnosis, Rocchio, and recency fusion."""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, EPS)


def orthogonal_noise(mu: np.ndarray, *basis: np.ndarray) -> np.ndarray:
    """μ_noise = μ − proj_S(μ), S = span(basis) after Gram-Schmidt.

    Subtracting this from q0 cannot decrease the q0 component: q0ᵀ μ_noise = 0
    when q0 is the first basis vector (up to numerical error).
    """
    mu_n = l2_normalize(np.asarray(mu, dtype=np.float64).reshape(-1), axis=0)
    axes: list[np.ndarray] = []
    for raw in basis:
        if raw is None:
            continue
        v = l2_normalize(np.asarray(raw, dtype=np.float64).reshape(-1), axis=0)
        for a in axes:
            v = v - float(np.dot(a, v)) * a
        nrm = float(np.linalg.norm(v))
        if nrm < EPS:
            continue
        axes.append(v / nrm)
    intent = np.zeros_like(mu_n)
    for a in axes:
        intent = intent + float(np.dot(a, mu_n)) * a
    return mu_n - intent


def softmax(x: np.ndarray, temperature: float = 0.15) -> np.ndarray:
    z = (x - np.max(x)) / max(temperature, EPS)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / np.maximum(e.sum(), EPS)


def effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size: 1 / Σ p_i²."""
    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0 or float(w.sum()) <= 0:
        return 0.0
    p = w / w.sum()
    return float(1.0 / np.sum(p * p))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        raise ValueError("weighted_median on empty array")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) / max(float(w.sum()), EPS)
    idx = int(np.searchsorted(cdf, 0.5, side="left"))
    idx = min(idx, v.size - 1)
    return float(v[idx])


def weighted_std(values: np.ndarray, weights: np.ndarray, mean: float | None = None) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0:
        return 0.0
    w = w / max(float(w.sum()), EPS)
    mu = float(np.sum(w * v) if mean is None else mean)
    var = float(np.sum(w * (v - mu) ** 2))
    return float(np.sqrt(max(var, 0.0)))


def kmeans_two(x: np.ndarray, rng: np.random.Generator, iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """2-means. Returns labels in {0,1} and centroids (2, d)."""
    n = x.shape[0]
    if n < 2:
        labels = np.zeros(n, dtype=np.int64)
        c = x.mean(axis=0, keepdims=True) if n else np.zeros((1, x.shape[1]))
        return labels, np.vstack([c, c])
    seeds = rng.choice(n, size=2, replace=False)
    centers = x[seeds].astype(np.float64).copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        d0 = np.linalg.norm(x - centers[0], axis=1)
        d1 = np.linalg.norm(x - centers[1], axis=1)
        labels = (d1 < d0).astype(np.int64)
        for k in (0, 1):
            mask = labels == k
            if mask.any():
                centers[k] = x[mask].mean(axis=0)
    return labels, centers
