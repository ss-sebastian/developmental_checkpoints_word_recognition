"""Stimulus matching, feature pooling, and leakage-safe probe primitives."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from .mald import PhoneInterval

VOWELS = frozenset({"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"})
POA = {
    "labial": ("P", "B", frozenset({"K", "T", "L", "M", "N", "W"})),
    "alveolar": ("T", "D", frozenset({"K", "P", "L", "M", "N", "W"})),
    "velar": ("K", "G", frozenset({"P", "T", "L", "M", "N", "W"})),
}


@dataclass(frozen=True)
class ProbeItem:
    item_id: str
    audio_path: str
    task: str
    poa: str
    target_start_s: float
    target_end_s: float
    label: int
    target_phone: str
    session_id: str


def base_phone(label: str) -> str:
    return label.rstrip("012")


def is_vowel(label: str) -> bool:
    return base_phone(label) in VOWELS


def pooled_overlapping_frames(frames: np.ndarray, *, phone_start_s: float, phone_end_s: float, sample_rate: int = 16_000, hop_length: int = 160, win_length: int = 400) -> np.ndarray:
    """Mean frames with nonzero overlap with [phone_start, phone_end).

    Acoustic training uses left-aligned 25-ms windows every 10 ms.  A phone
    boundary itself is not enough: a frame is included iff
    ``frame_start < phone_end and frame_end > phone_start``.
    """
    if frames.ndim != 2 or phone_end_s <= phone_start_s:
        raise ValueError("Expected [frames, dimensions] and a positive phone interval")
    starts = np.arange(len(frames)) * hop_length / sample_rate
    ends = starts + win_length / sample_rate
    selected = (starts < phone_end_s) & (ends > phone_start_s)
    if not np.any(selected):
        raise ValueError("No 25-ms/10-ms model frame overlaps the annotated phone")
    return np.asarray(frames[selected].mean(axis=0), dtype=np.float32)


def _item_id(path: str, task: str, poa: str, target_index: int) -> str:
    digest = hashlib.sha1(f"{path}|{task}|{poa}|{target_index}".encode()).hexdigest()[:16]
    return f"{task}_{poa}_{digest}"


def _add(items: list[ProbeItem], path: str, task: str, poa: str, phones: list[PhoneInterval], index: int, label: int) -> None:
    target = phones[index]
    items.append(ProbeItem(_item_id(path, task, poa, index), path, task, poa, target.start_s, target.end_s, label, target.label, path))


def select_aspiration_items(audio_path: str, phones: list[PhoneInterval]) -> list[ProbeItem]:
    """Exact Table-1 aspiration patterns for one MALD utterance.

    Labels 0/1/2 are respectively the first Table-1 group, second group, and
    the required post-/s/ third-class confound.  ARPAbet P/T/K word-initial
    stops are the operational 'aspirated' class; no unsupported acoustic label
    is inferred from the TextGrid itself.
    """
    out: list[ProbeItem] = []
    for poa, (unvoiced, voiced, confounds) in POA.items():
        for index, phone in enumerate(phones):
            following_vowel = index + 1 < len(phones) and is_vowel(phones[index + 1].label)
            initial = index == 0
            # Table 1's ``# s target V`` is word-initial /s/+stop only, not a
            # generic within-word sC cluster.
            post_s = index == 1 and phones[0].label == "S"
            if not following_vowel:
                continue
            if initial and phone.label == unvoiced:
                _add(out, audio_path, "aspiration_phonemic", poa, phones, index, 0)
                _add(out, audio_path, "aspiration_phonetic", poa, phones, index, 0)
            elif post_s and phone.label == unvoiced:
                _add(out, audio_path, "aspiration_phonemic", poa, phones, index, 0)
                _add(out, audio_path, "aspiration_phonetic", poa, phones, index, 1)
            elif initial and phone.label == voiced:
                _add(out, audio_path, "aspiration_phonemic", poa, phones, index, 1)
                _add(out, audio_path, "aspiration_phonetic", poa, phones, index, 1)
            elif post_s and phone.label in confounds:
                _add(out, audio_path, "aspiration_phonemic", poa, phones, index, 2)
                _add(out, audio_path, "aspiration_phonetic", poa, phones, index, 2)
    return out


def select_control_items(audio_path: str, phones: list[PhoneInterval]) -> list[ProbeItem]:
    """Paper Table-1 C/V, V1/V0 and two four-phone distant controls."""
    out: list[ProbeItem] = []
    for index, phone in enumerate(phones):
        vowel = is_vowel(phone.label)
        _add(out, audio_path, "control_consonant_vowel", "all", phones, index, int(vowel))
        if vowel and phone.label[-1:] in {"0", "1"}:
            _add(out, audio_path, "control_stress", "all", phones, index, int(phone.label.endswith("1")))
        if index >= 4 and vowel:
            _add(out, audio_path, "control_distant_before", "all", phones, index, int(is_vowel(phones[index - 4].label)))
        if index + 4 < len(phones) and vowel:
            _add(out, audio_path, "control_distant_after", "all", phones, index, int(is_vowel(phones[index + 4].label)))
    return out


def fixed_item_folds(item_ids: Iterable[str], n_folds: int = 10, seed: int = 20261006) -> dict[str, int]:
    """Stable item-ID folds shared across every representation family."""
    ids = sorted(set(item_ids))
    if len(ids) < n_folds:
        raise ValueError(f"Need at least {n_folds} unique matched items for {n_folds}-fold CV")
    ordered = sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    return {item_id: index % n_folds for index, item_id in enumerate(ordered)}


def stratified_fixed_item_folds(item_ids: Iterable[str], labels: Iterable[int], n_folds: int = 10, seed: int = 20261006) -> dict[str, int]:
    """Fixed feature-independent item folds, stratified by task label."""
    grouped: dict[int, list[str]] = {}
    for item_id, label in zip(item_ids, labels, strict=True):
        grouped.setdefault(int(label), []).append(item_id)
    if not grouped or min(len(values) for values in grouped.values()) < n_folds:
        raise ValueError(f"Need at least {n_folds} items in every class for stratified {n_folds}-fold CV")
    folds: dict[str, int] = {}
    for label, ids in sorted(grouped.items()):
        ordered = sorted(set(ids), key=lambda value: hashlib.sha256(f"{seed}:{label}:{value}".encode()).hexdigest())
        for index, item_id in enumerate(ordered):
            folds[item_id] = index % n_folds
    return folds


def _binary_auc(binary: np.ndarray, score: np.ndarray) -> float:
    """Tie-correct Mann--Whitney ROC AUC without a sklearn dependency."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float); ranks[order] = np.arange(1, len(score) + 1)
    for value in np.unique(score):
        same = np.flatnonzero(score == value); ranks[same] = ranks[same].mean()
    n_pos = int(binary.sum()); n_neg = len(binary) - n_pos
    if not n_pos or not n_neg:
        raise ValueError("AUC needs both classes")
    return float((ranks[binary == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def prevalence_weighted_ovo_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Paper's prevalence-weighted one-vs-one multiclass AUC.

    For every class pair, compute an AUC once using each member's own score,
    average those two values, then prevalence-weight pairs.  This is the
    Hand--Till one-vs-one construction used by the paper's sklearn workflow
    (``multi_class='ovo', average='weighted'``), made explicit here so it has
    no version-dependent metric semantics.
    """
    labels = np.unique(y_true)
    if probabilities.ndim != 2 or probabilities.shape != (len(y_true), len(labels)):
        raise ValueError("probabilities must be [items, sorted unique labels]")
    total = 0.0; weight_total = 0.0
    for left, right in combinations(labels.tolist(), 2):
        mask = (y_true == left) | (y_true == right)
        left_binary = (y_true[mask] == left).astype(np.int8)
        right_binary = (y_true[mask] == right).astype(np.int8)
        left_index = int(np.where(labels == left)[0][0]); right_index = int(np.where(labels == right)[0][0])
        auc = (_binary_auc(left_binary, probabilities[mask, left_index]) + _binary_auc(right_binary, probabilities[mask, right_index])) / 2
        weight = int(mask.sum()) / len(y_true)
        total += weight * auc; weight_total += weight
    return float(total / weight_total)
