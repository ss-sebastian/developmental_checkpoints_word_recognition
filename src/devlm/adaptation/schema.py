from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_NAMES = frozenset({"Sound", "Meaning", "Plausibility", "Grammaticality"})
STIMULUS_KINDS = frozenset({"word_pair", "sentence"})
REQUIRED_COLUMNS = frozenset({
    "item_id", "task_name", "stimulus_kind", "prime_phonemes",
    "target_phonemes", "sentence_phonemes", "binary_label", "split_group",
    "metadata_json",
})
FORBIDDEN_BEHAVIOR_FIELDS = frozenset({
    "age", "child_age", "participant", "participant_id", "subject_id",
    "rt", "reaction_time", "human_rt",
})


@dataclass(frozen=True)
class AdaptationItem:
    item_id: str
    task_name: str
    stimulus_kind: str
    binary_label: int
    split_group: str
    metadata: dict[str, Any]
    prime_phonemes: tuple[str, ...] = ()
    target_phonemes: tuple[str, ...] = ()
    sentence_phonemes: tuple[str, ...] = ()

    @property
    def phonemes(self) -> tuple[str, ...]:
        if self.stimulus_kind == "word_pair":
            return self.prime_phonemes + self.target_phonemes
        return self.sentence_phonemes


def _phoneme_list(raw: str, field: str, item_id: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{item_id}: {field} must be a JSON array of IPA tokens") from exc
    if not isinstance(value, list) or not value or any(not isinstance(token, str) or not token for token in value):
        raise ValueError(f"{item_id}: {field} must be a non-empty JSON string array")
    return tuple(value)


def _metadata(raw: str, item_id: str) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"{item_id}: metadata_json must be a JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{item_id}: metadata_json must be a JSON object")
    forbidden = FORBIDDEN_BEHAVIOR_FIELDS.intersection(key.lower() for key in value)
    if forbidden:
        raise ValueError(f"{item_id}: child-behavior metadata is forbidden during adaptation: {sorted(forbidden)}")
    return value


def load_manifest(path: str | Path) -> list[AdaptationItem]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Adaptation manifest is missing columns: {', '.join(sorted(missing))}")
        forbidden_columns = FORBIDDEN_BEHAVIOR_FIELDS.intersection(name.lower() for name in columns)
        if forbidden_columns:
            raise ValueError(f"Child-behavior columns are forbidden during adaptation: {sorted(forbidden_columns)}")
        rows = list(reader)

    items: list[AdaptationItem] = []
    seen: set[str] = set()
    for row in rows:
        item_id = row["item_id"].strip()
        if not item_id or item_id in seen:
            raise ValueError(f"item_id must be unique and non-empty: {item_id!r}")
        task_name = row["task_name"].strip()
        if task_name not in TASK_NAMES:
            raise ValueError(f"{item_id}: unsupported task_name {task_name!r}")
        kind = row["stimulus_kind"].strip()
        if kind not in STIMULUS_KINDS:
            raise ValueError(f"{item_id}: unsupported stimulus_kind {kind!r}")
        try:
            label = int(row["binary_label"])
        except ValueError as exc:
            raise ValueError(f"{item_id}: binary_label must be 0 or 1") from exc
        if label not in (0, 1):
            raise ValueError(f"{item_id}: binary_label must be 0 or 1")
        split_group = row["split_group"].strip()
        if not split_group:
            raise ValueError(f"{item_id}: split_group cannot be empty")
        prime = _phoneme_list(row["prime_phonemes"], "prime_phonemes", item_id)
        target = _phoneme_list(row["target_phonemes"], "target_phonemes", item_id)
        sentence = _phoneme_list(row["sentence_phonemes"], "sentence_phonemes", item_id)
        if kind == "word_pair" and (not prime or not target or sentence):
            raise ValueError(f"{item_id}: word_pair requires prime and target only")
        if kind == "sentence" and (not sentence or prime or target):
            raise ValueError(f"{item_id}: sentence requires sentence_phonemes only")
        items.append(AdaptationItem(
            item_id, task_name, kind, label, split_group, _metadata(row["metadata_json"], item_id),
            prime, target, sentence,
        ))
        seen.add(item_id)
    if not items:
        raise ValueError("Adaptation manifest is empty")
    return items


def manifest_fingerprint(items: list[AdaptationItem]) -> str:
    serializable = [
        {
            "item_id": item.item_id,
            "task_name": item.task_name,
            "stimulus_kind": item.stimulus_kind,
            "binary_label": item.binary_label,
            "split_group": item.split_group,
            "prime_phonemes": item.prime_phonemes,
            "target_phonemes": item.target_phonemes,
            "sentence_phonemes": item.sentence_phonemes,
            "metadata": item.metadata,
        }
        for item in sorted(items, key=lambda value: value.item_id)
    ]
    encoded = json.dumps(serializable, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
