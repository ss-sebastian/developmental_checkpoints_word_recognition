from __future__ import annotations

import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Utterance:
    corpus_id: str
    session_id: str
    target_child_age_months: float
    utterance_order: int
    ipa: str
    text: str
    phonemes: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Session:
    corpus_id: str
    session_id: str
    target_child_age_months: float
    utterances: tuple[Utterance, ...]


def _records(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    elif path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open(encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle, delimiter=delimiter)
    else:
        raise ValueError("IPA-CHILDES input must be .jsonl, .csv, or .tsv")


def load_ipa_childes(path: str | Path, progress: bool = False) -> list[Session]:
    """Load an already-exported IPA-CHILDES table and keep North American English.

    Required fields: corpus_id, session_id, target_child_age_months,
    utterance_order, ipa, text. Optional language/dialect fields are validated when
    present. A pre-segmented ``phonemes`` JSON list is accepted and preferred.
    """
    path = Path(path)
    grouped: dict[tuple[str, str], list[Utterance]] = {}
    for row_index, row in enumerate(_records(path), 1):
        if progress and row_index % 100_000 == 0:
            print(f"Loaded {row_index:,} IPA-CHILDES utterance rows...", file=sys.stderr, flush=True)
        language = str(row.get("language", "English")).strip().lower()
        dialect = str(row.get("dialect", "North American English")).strip().lower()
        if language not in {"english", "eng", "en"}:
            continue
        if dialect and dialect not in {
            "north american english", "north american", "nae", "en-us", "en-ca"
        }:
            continue
        missing = [k for k in ("corpus_id", "session_id", "target_child_age_months", "utterance_order", "ipa", "text") if row.get(k) in (None, "")]
        if missing:
            raise ValueError(f"Missing required IPA-CHILDES fields: {', '.join(missing)}")
        raw_phonemes = row.get("phonemes")
        if isinstance(raw_phonemes, str) and raw_phonemes.strip():
            raw_phonemes = json.loads(raw_phonemes)
        phonemes = tuple(raw_phonemes) if raw_phonemes else None
        utt = Utterance(
            corpus_id=str(row["corpus_id"]), session_id=str(row["session_id"]),
            target_child_age_months=float(row["target_child_age_months"]),
            utterance_order=int(row["utterance_order"]), ipa=str(row["ipa"]),
            text=str(row["text"]), phonemes=phonemes,
        )
        grouped.setdefault((utt.corpus_id, utt.session_id), []).append(utt)
    sessions = []
    for (corpus_id, session_id), utterances in grouped.items():
        utterances.sort(key=lambda x: x.utterance_order)
        ages = {u.target_child_age_months for u in utterances}
        if len(ages) != 1:
            raise ValueError(f"Inconsistent target-child ages in {corpus_id}/{session_id}")
        sessions.append(Session(corpus_id, session_id, ages.pop(), tuple(utterances)))
    return sorted(sessions, key=lambda s: (s.target_child_age_months, s.corpus_id, s.session_id))


def split_sessions(sessions: list[Session], validation_fraction: float, seed: int) -> tuple[list[Session], list[Session]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(sessions) < 2:
        raise ValueError("At least two sessions are required for a session-level split")
    shuffled = list(sessions)
    random.Random(seed).shuffle(shuffled)
    n_val = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    val_keys = {(s.corpus_id, s.session_id) for s in shuffled[:n_val]}
    train = [s for s in sessions if (s.corpus_id, s.session_id) not in val_keys]
    val = [s for s in sessions if (s.corpus_id, s.session_id) in val_keys]
    return train, val
