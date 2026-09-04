from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch

from .model import CHECKPOINT_IDS
from .schema import TASK_NAMES


@dataclass(frozen=True)
class AdaptationResult:
    """Schema for a future task x checkpoint result; this module performs no I/O."""

    checkpoint_id: str
    task_name: str
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    test_accuracy: float
    validation_AUC: float
    test_AUC: float
    best_epoch: int
    number_train_items: int
    number_validation_items: int
    number_test_items: int
    readout_state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class SeedOutputLayout:
    """Deterministic future paths grouping all checkpoints from one initialization."""

    directory: Path
    run_manifest: Path
    all_checkpoint_metrics: Path
    head_state_dicts: dict[str, Path]


def seed_output_layout(
    output_root: str | Path,
    task_name: str,
    split_sha256: str,
    initialization_seed: int,
) -> SeedOutputLayout:
    """Describe output names only; intentionally creates no files or directories."""
    if task_name not in TASK_NAMES:
        raise ValueError(f"Unsupported task_name {task_name!r}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", split_sha256):
        raise ValueError("split_sha256 must be a 64-character hexadecimal digest")
    if initialization_seed < 0 or initialization_seed >= 10_000_000_000:
        raise ValueError("initialization_seed must be between 0 and 9,999,999,999")
    directory = (
        Path(output_root)
        / f"task-{task_name.lower()}"
        / f"split-{split_sha256.lower()[:12]}"
        / f"init-seed-{int(initialization_seed):010d}"
    )
    return SeedOutputLayout(
        directory=directory,
        run_manifest=directory / "run_manifest.json",
        all_checkpoint_metrics=directory / "all_checkpoint_metrics.tsv",
        head_state_dicts={
            checkpoint_id: directory / "heads" / f"{checkpoint_id}_linear_readout.pt"
            for checkpoint_id in CHECKPOINT_IDS
        },
    )


def seed_run_manifest(
    task_name: str,
    split_sha256: str,
    initialization_seed: int,
) -> dict:
    """JSON-ready unified index for one seed's complete M01-M30 result set."""
    layout = seed_output_layout(".", task_name, split_sha256, initialization_seed)
    return {
        "schema_version": 1,
        "task_name": task_name,
        "split_sha256": split_sha256.lower(),
        "initialization_seed": initialization_seed,
        "checkpoint_ids": list(CHECKPOINT_IDS),
        "all_checkpoint_metrics": layout.all_checkpoint_metrics.name,
        "head_state_dicts": {
            checkpoint_id: str(path.relative_to(layout.directory))
            for checkpoint_id, path in layout.head_state_dicts.items()
        },
    }
