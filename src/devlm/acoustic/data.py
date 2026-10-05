from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioItem:
    audio_path: Path
    corpus_id: str
    session_id: str
    target_child_age_months: float
    recording_order: int
    source_corpus: str
    speaker_role: str
    preassigned_split: str | None = None

    @property
    def session_key(self) -> tuple[str, str]:
        return self.corpus_id, self.session_id


def load_audio_manifest(path: str | Path) -> list[AudioItem]:
    """Read a local WAV manifest without ever reading linguistic labels.

    Required columns are ``audio_path``, ``corpus_id``, ``session_id``,
    ``target_child_age_months``, ``source_corpus``, ``speaker_role`` and
    ``directed_to_child``. Only explicitly adult/caregiver-to-child clips are
    accepted. ``recording_order`` is optional and defaults to source-row order.
    Audio paths may be relative to the manifest.
    """
    path = Path(path).resolve()
    if path.suffix.lower() not in {".tsv", ".csv"}:
        raise ValueError("audio_manifest_path must be a .tsv or .csv file")
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))
    required = {
        "audio_path", "corpus_id", "session_id", "target_child_age_months",
        "source_corpus", "speaker_role", "directed_to_child",
    }
    if not rows:
        raise ValueError("Audio manifest has no records")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Audio manifest missing required columns: {', '.join(sorted(missing))}")
    items: list[AudioItem] = []
    for row_number, row in enumerate(rows, 2):
        absent = [key for key in required if not str(row.get(key, "")).strip()]
        if absent:
            raise ValueError(f"Audio manifest row {row_number} has blank fields: {', '.join(sorted(absent))}")
        audio_path = Path(str(row["audio_path"]))
        audio_path = audio_path if audio_path.is_absolute() else (path.parent / audio_path).resolve()
        if audio_path.suffix.lower() != ".wav":
            raise ValueError(f"Only uncompressed .wav input is supported in this first acoustic run: {audio_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Manifest audio file does not exist: {audio_path}")
        speaker_role = str(row["speaker_role"]).strip().lower()
        adult_roles = {"adult", "caregiver", "mother", "father", "parent", "grandparent", "other_adult"}
        if speaker_role not in adult_roles:
            raise ValueError(
                f"Audio manifest row {row_number} is not an adult/caregiver speaker ({speaker_role!r}); "
                "target-child/CHI speech is intentionally excluded"
            )
        child_directed = str(row["directed_to_child"]).strip().lower()
        if child_directed not in {"1", "true", "yes", "y"}:
            raise ValueError(
                f"Audio manifest row {row_number} is not explicitly child-directed; "
                "set directed_to_child=true only for adult/caregiver speech addressed to a child"
            )
        preassigned_split = str(row.get("split", "")).strip().lower() or None
        if preassigned_split not in {None, "train", "validation"}:
            raise ValueError(f"Audio manifest row {row_number} has invalid split={preassigned_split!r}; use train or validation")
        items.append(AudioItem(
            audio_path=audio_path,
            corpus_id=str(row["corpus_id"]),
            session_id=str(row["session_id"]),
            target_child_age_months=float(row["target_child_age_months"]),
            recording_order=int(row.get("recording_order") or row_number),
            source_corpus=str(row["source_corpus"]),
            speaker_role=speaker_role,
            preassigned_split=preassigned_split,
        ))
    return sorted(items, key=lambda item: (item.target_child_age_months, item.corpus_id, item.session_id, item.recording_order))


def split_audio_sessions(items: list[AudioItem], validation_fraction: float, seed: int) -> tuple[list[AudioItem], list[AudioItem]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    session_keys = sorted({item.session_key for item in items})
    if len(session_keys) < 2:
        raise ValueError("At least two sessions are required for a session-level audio split")
    assigned = {item.preassigned_split for item in items}
    if assigned != {None}:
        if None in assigned or not {"train", "validation"}.issubset(assigned):
            raise ValueError("If manifest supplies split, every item must be assigned and both train/validation must be present")
        by_session: dict[tuple[str, str], set[str]] = {}
        for item in items:
            by_session.setdefault(item.session_key, set()).add(str(item.preassigned_split))
        ambiguous = [key for key, splits in by_session.items() if len(splits) != 1]
        if ambiguous:
            raise ValueError(f"Manifest assigns a session to multiple splits (first: {ambiguous[0]})")
        return ([item for item in items if item.preassigned_split == "train"], [item for item in items if item.preassigned_split == "validation"])
    shuffled = list(session_keys)
    random.Random(seed).shuffle(shuffled)
    n_validation = min(len(shuffled) - 1, max(1, round(len(shuffled) * validation_fraction)))
    validation_keys = set(shuffled[:n_validation])
    train = [item for item in items if item.session_key not in validation_keys]
    validation = [item for item in items if item.session_key in validation_keys]
    return train, validation
