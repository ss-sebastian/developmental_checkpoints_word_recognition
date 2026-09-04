from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .constants import SEED
from .resources import Frequency, Pronunciation


@dataclass(frozen=True)
class LexicalPair:
    word1: str
    word2: str
    relation: str
    source_record_id: str
    association_forward: float | None = None
    association_backward: float | None = None


def stable_key(label: str, *values: str) -> str:
    return hashlib.sha256((str(SEED) + "\0" + label + "\0" + "\0".join(values)).encode()).hexdigest()


def pair_covariates(
    pair: LexicalPair,
    frequencies: dict[str, Frequency],
    pronunciations: dict[str, Pronunciation],
) -> np.ndarray:
    a, b = pair.word1, pair.word2
    fa, fb = frequencies[a].zipf, frequencies[b].zipf
    pa, pb = pronunciations[a], pronunciations[b]
    return np.asarray([
        (fa + fb) / 2, abs(fa - fb), (len(a) + len(b)) / 2, abs(len(a) - len(b)),
        (len(pa.phones) + len(pb.phones)) / 2, abs(len(pa.phones) - len(pb.phones)),
        (pa.syllables + pb.syllables) / 2,
    ], dtype=float)


def _disjoint_reservoir(
    pool: list[LexicalPair], count: int, forbidden: set[str], label: str,
) -> list[LexicalPair]:
    selected: list[LexicalPair] = []
    local = set(forbidden)
    for pair in sorted(pool, key=lambda item: stable_key(label, item.word1, item.word2, item.source_record_id)):
        if pair.word1 in local or pair.word2 in local or pair.word1 == pair.word2:
            continue
        selected.append(pair)
        local.update((pair.word1, pair.word2))
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"{label}: only {len(selected)} word-disjoint candidates; need {count}")
    return selected


def select_matched_pair_sets(
    pool_a: list[LexicalPair],
    pool_b: list[LexicalPair],
    count: int,
    globally_used_words: set[str],
    frequencies: dict[str, Frequency],
    pronunciations: dict[str, Pronunciation],
    label: str,
    reservoir_factor_a: int = 2,
    reservoir_factor_b: int = 4,
    feature_weights: np.ndarray | None = None,
) -> tuple[list[LexicalPair], list[LexicalPair]]:
    reservoir_a = _disjoint_reservoir(
        pool_a, min(len(pool_a), max(count * reservoir_factor_a, count)),
        globally_used_words, label + ":a",
    )
    blocked_for_b = globally_used_words | {word for pair in reservoir_a for word in (pair.word1, pair.word2)}
    reservoir_b = _disjoint_reservoir(
        pool_b, min(len(pool_b), max(count * reservoir_factor_b, count)),
        blocked_for_b, label + ":b",
    )
    matrix_a = np.stack([pair_covariates(pair, frequencies, pronunciations) for pair in reservoir_a])
    matrix_b = np.stack([pair_covariates(pair, frequencies, pronunciations) for pair in reservoir_b])
    combined = np.concatenate([matrix_a, matrix_b])
    scale = combined.std(axis=0)
    scale[scale == 0] = 1
    normalized = (matrix_a[:, None, :] - matrix_b[None, :, :]) / scale
    if feature_weights is not None:
        normalized = normalized * feature_weights
    cost = np.square(normalized).sum(axis=2)
    rows, columns = linear_sum_assignment(cost)
    ranked = sorted(zip(rows, columns, strict=True), key=lambda rc: (cost[rc], stable_key(label, str(rc[0]), str(rc[1]))))
    chosen = ranked[:count]
    if len(chosen) < count:
        raise RuntimeError(f"{label}: matching returned {len(chosen)} pairs; need {count}")
    selected_a = [reservoir_a[row] for row, _ in chosen]
    selected_b = [reservoir_b[column] for _, column in chosen]
    globally_used_words.update(word for pair in selected_a + selected_b for word in (pair.word1, pair.word2))
    return selected_a, selected_b


def re_pair(
    positive_pairs: list[LexicalPair],
    valid: Callable[[str, str], bool],
    frequencies: dict[str, Frequency],
    pronunciations: dict[str, Pronunciation],
    label: str,
) -> list[LexicalPair]:
    left = [pair.word1 for pair in positive_pairs]
    right = [pair.word2 for pair in positive_pairs]
    n = len(left)
    cost = np.full((n, n), 1e9, dtype=float)
    for i, word1 in enumerate(left):
        for j, word2 in enumerate(right):
            if valid(word1, word2):
                original = LexicalPair(word1, word2, "unrelated", f"derived:{label}:{i}:{j}")
                anchor = positive_pairs[i]
                delta = pair_covariates(original, frequencies, pronunciations) - pair_covariates(anchor, frequencies, pronunciations)
                cost[i, j] = float(np.square(delta).sum())
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != n or np.any(cost[rows, columns] >= 1e8):
        feasible = int(np.sum(cost[rows, columns] < 1e8))
        raise RuntimeError(f"{label}: only {feasible}/{n} formal unrelated re-pairs found")
    return [
        LexicalPair(left[i], right[j], "unrelated", f"derived:{label}:{i}:{j}", 0.0, 0.0)
        for i, j in zip(rows, columns, strict=True)
    ]
