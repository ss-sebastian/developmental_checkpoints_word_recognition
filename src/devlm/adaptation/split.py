from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .schema import AdaptationItem, manifest_fingerprint


PARTITIONS = ("train", "validation", "test")


@dataclass(frozen=True)
class ItemSplit:
    task_name: str
    seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    manifest_sha256: str
    assignments: dict[str, str]
    split_groups: dict[str, str]


def _partition_counts(group_count: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count < 3:
        raise ValueError("At least three split groups are required for train/validation/test")
    train = max(1, int(round(group_count * fractions[0])))
    validation = max(1, int(round(group_count * fractions[1])))
    if train + validation >= group_count:
        train = max(1, group_count - 2)
        validation = 1
    return train, validation, group_count - train - validation


def create_item_split(
    items: list[AdaptationItem],
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> ItemSplit:
    if not items:
        raise ValueError("Cannot split an empty manifest")
    task_names = {item.task_name for item in items}
    if len(task_names) != 1:
        raise ValueError("Create and save one split separately for each task")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must be positive and sum to 1")
    groups = sorted({item.split_group for item in items})
    random.Random(seed).shuffle(groups)
    n_train, n_validation, _ = _partition_counts(len(groups), fractions)
    group_partition = {
        group: ("train" if index < n_train else "validation" if index < n_train + n_validation else "test")
        for index, group in enumerate(groups)
    }
    assignments = {item.item_id: group_partition[item.split_group] for item in items}
    artifact = ItemSplit(
        next(iter(task_names)), seed, *fractions, manifest_fingerprint(items), assignments,
        {item.item_id: item.split_group for item in items},
    )
    verify_item_split(items, artifact)
    return artifact


def verify_item_split(items: list[AdaptationItem], split: ItemSplit) -> None:
    item_ids = {item.item_id for item in items}
    task_names = {item.task_name for item in items}
    if task_names != {split.task_name}:
        raise ValueError("Saved split task_name does not match the manifest")
    if set(split.assignments) != item_ids or set(split.split_groups) != item_ids:
        raise ValueError("Saved split items do not exactly match the manifest")
    if manifest_fingerprint(items) != split.manifest_sha256:
        raise ValueError("Saved split was created from a different manifest")
    if set(split.assignments.values()) - set(PARTITIONS):
        raise ValueError("Saved split contains an invalid partition")
    if set(split.assignments.values()) != set(PARTITIONS):
        raise ValueError("Every train/validation/test partition must be non-empty")
    group_partitions: dict[str, set[str]] = {}
    for item in items:
        group_partitions.setdefault(item.split_group, set()).add(split.assignments[item.item_id])
    if any(len(partitions) != 1 for partitions in group_partitions.values()):
        raise ValueError("A split_group appears in more than one partition")


def save_item_split(split: ItemSplit, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "task_name": split.task_name,
        "seed": split.seed,
        "fractions": {
            "train": split.train_fraction,
            "validation": split.validation_fraction,
            "test": split.test_fraction,
        },
        "manifest_sha256": split.manifest_sha256,
        "items": [
            {"item_id": item_id, "split_group": split.split_groups[item_id], "partition": split.assignments[item_id]}
            for item_id in sorted(split.assignments)
        ],
    }
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_item_split(path: str | Path, items: list[AdaptationItem]) -> ItemSplit:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported item-split schema version")
    rows = payload["items"]
    fractions = payload["fractions"]
    split = ItemSplit(
        str(payload["task_name"]), int(payload["seed"]), float(fractions["train"]),
        float(fractions["validation"]), float(fractions["test"]), str(payload["manifest_sha256"]),
        {str(row["item_id"]): str(row["partition"]) for row in rows},
        {str(row["item_id"]): str(row["split_group"]) for row in rows},
    )
    verify_item_split(items, split)
    return split


def items_by_partition(items: list[AdaptationItem], split: ItemSplit) -> dict[str, list[AdaptationItem]]:
    verify_item_split(items, split)
    return {
        partition: [item for item in items if split.assignments[item.item_id] == partition]
        for partition in PARTITIONS
    }
