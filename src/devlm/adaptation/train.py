from __future__ import annotations

import csv
import hashlib
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from devlm.features import FeatureTable
from devlm.train import resolve_device

from .input import build_adaptation_batch
from .model import CHECKPOINT_IDS, FrozenGRUEncoder, make_binary_readout, task_loss
from .schema import AdaptationItem, TASK_NAMES


TASK_ORDER = ("Sound", "Meaning", "Plausibility", "Grammaticality")
PARTITIONS = ("train", "validation", "test")
METRIC_COLUMNS = (
    "task_name", "checkpoint_id", "checkpoint_file", "checkpoint_hours",
    "initialization_seed", "input_noise_seed", "best_epoch", "epochs_run",
    "train_loss", "validation_loss", "test_loss", "train_accuracy",
    "validation_accuracy", "test_accuracy", "train_auc", "validation_auc", "test_auc",
    "number_train_items", "number_validation_items", "number_test_items",
)


@dataclass(frozen=True)
class TrainingOptions:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    encoding_batch_size: int = 128
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    noise_sigma: float = 0.05
    input_noise_seed: int = 20260904
    initialization_seeds: tuple[int, ...] = (1729, 2718, 3141)
    device: str = "auto"


def _json_phonemes(raw: str, item_id: str, field: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    value = json.loads(raw)
    if not isinstance(value, list) or not value or any(not isinstance(token, str) or not token for token in value):
        raise ValueError(f"{item_id}: {field} must be a non-empty JSON array of IPA strings")
    return tuple(value)


def load_construction_manifest(path: str | Path) -> list[AdaptationItem]:
    """Load the fixed 4-task stimulus manifest produced by stimulus construction."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("Stimulus manifest is empty")
    required = {"item_id", "task", "binary_label", "split", "word1_ipa", "word2_ipa", "sentence_ipa"}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Stimulus manifest is missing columns: {', '.join(sorted(missing))}")
    items: list[AdaptationItem] = []
    seen: set[str] = set()
    for row in rows:
        item_id = row["item_id"].strip()
        task = row["task"].strip()
        split = row["split"].strip()
        if not item_id or item_id in seen:
            raise ValueError(f"Duplicate or empty item_id: {item_id!r}")
        if task not in TASK_NAMES or split not in PARTITIONS:
            raise ValueError(f"{item_id}: invalid task/split {task!r}/{split!r}")
        label = int(row["binary_label"])
        if label not in (0, 1):
            raise ValueError(f"{item_id}: binary_label must be 0 or 1")
        if task in {"Sound", "Meaning"}:
            prime = _json_phonemes(row["word1_ipa"], item_id, "word1_ipa")
            target = _json_phonemes(row["word2_ipa"], item_id, "word2_ipa")
            sentence = ()
            kind = "word_pair"
        else:
            prime = target = ()
            sentence = _json_phonemes(row["sentence_ipa"], item_id, "sentence_ipa")
            kind = "sentence"
        metadata = {
            "partition": split,
            "condition": row.get("condition", ""),
            "source_record_id": row.get("source_record_id", ""),
        }
        items.append(AdaptationItem(
            item_id=item_id, task_name=task, stimulus_kind=kind,
            binary_label=label, split_group=item_id, metadata=metadata,
            prime_phonemes=prime, target_phonemes=target, sentence_phonemes=sentence,
        ))
        seen.add(item_id)
    counts = {(task, split): sum(item.task_name == task and item.metadata["partition"] == split for item in items)
              for task in TASK_ORDER for split in PARTITIONS}
    expected = {"train": 360, "validation": 100, "test": 160}
    failures = {key: value for key, value in counts.items() if value != expected[key[1]]}
    if failures:
        raise ValueError(f"Stimulus manifest does not contain the fixed 620-item design: {failures}")
    return items


def discover_checkpoints(root: str | Path) -> list[tuple[str, Path, float]]:
    """Find and developmentally order exactly 30 valid Phase 1 checkpoints."""
    valid: list[tuple[float, int, Path]] = []
    errors: list[str] = []
    for path in sorted(Path(root).rglob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not {"model", "config", "vocabulary", "metadata"}.issubset(payload):
                continue
            metadata = payload["metadata"]
            hours = float(metadata["equivalent_input_duration_hours"])
            step = int(metadata["optimizer_step"])
            valid.append((hours, step, path))
        except Exception as exc:  # report malformed .pt files only if discovery fails
            errors.append(f"{path}: {exc}")
    # Remove duplicate copies identified by their developmental exposure and
    # optimizer step (nested ZIP folders sometimes contain the same run twice).
    unique: dict[tuple[float, int], tuple[float, int, Path]] = {}
    for record in valid:
        unique.setdefault((record[0], record[1]), record)
    ordered = sorted(unique.values(), key=lambda value: (value[0], value[1], str(value[2])))
    if len(ordered) != 30:
        detail = f" Found {len(ordered)} valid unique checkpoints."
        if errors:
            detail += f" First unreadable file: {errors[0]}"
        raise ValueError("Checkpoint ZIP must contain exactly 30 Phase 1 checkpoints." + detail)
    return [(CHECKPOINT_IDS[index], path, hours) for index, (hours, _, path) in enumerate(ordered)]


def discover_feature_table(root: str | Path) -> Path:
    candidates = sorted(Path(root).rglob("ipa_feature_mapping.json"))
    if not candidates:
        raise ValueError("Checkpoint ZIP does not contain ipa_feature_mapping.json")
    digests = {hashlib.sha256(path.read_bytes()).hexdigest() for path in candidates}
    if len(digests) != 1:
        raise ValueError("Checkpoint ZIP contains conflicting IPA feature tables")
    return candidates[0]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _encode(
    encoder: FrozenGRUEncoder,
    frames: torch.Tensor,
    lengths: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    encoder.eval()
    chunks = []
    for start in range(0, len(frames), batch_size):
        end = min(len(frames), start + batch_size)
        chunks.append(encoder(frames[start:end].to(device), lengths[start:end].to(device)).cpu())
    return torch.cat(chunks)


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y = labels.detach().cpu().numpy().astype(np.int64)
    s = scores.detach().cpu().numpy().astype(np.float64)
    positive = y == 1
    negative = y == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def _evaluate(head: nn.Linear, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module) -> tuple[float, float, float]:
    logits = head(x).squeeze(-1)
    loss = float(loss_fn(logits, y))
    accuracy = float(((logits >= 0).long() == y.long()).float().mean())
    return loss, accuracy, _auc(y, logits)


def _train_head(
    representations: torch.Tensor,
    labels: torch.Tensor,
    indices: dict[str, torch.Tensor],
    hidden_dim: int,
    initialization_seed: int,
    options: TrainingOptions,
    device: torch.device,
) -> tuple[nn.Linear, dict[str, float | int]]:
    _set_seed(initialization_seed)
    head = make_binary_readout(hidden_dim, initialization_seed).to(device)
    x = representations.to(device)
    y = labels.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    loss_fn = task_loss()
    best_loss = float("inf")
    best_epoch = 0
    best_state = deepcopy(head.state_dict())
    stale = 0
    epochs_run = 0
    train_indices = indices["train"]
    for epoch in range(1, options.max_epochs + 1):
        epochs_run = epoch
        generator = torch.Generator(device="cpu").manual_seed(initialization_seed + epoch)
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        head.train()
        for start in range(0, len(permutation), options.batch_size):
            batch = permutation[start:start + options.batch_size].to(device)
            loss = loss_fn(head(x[batch]).squeeze(-1), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        validation_loss, _, _ = _evaluate(head, x[indices["validation"].to(device)], y[indices["validation"].to(device)], loss_fn)
        if validation_loss < best_loss - options.min_delta:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= options.patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    metrics: dict[str, float | int] = {"best_epoch": best_epoch, "epochs_run": epochs_run}
    for partition in PARTITIONS:
        subset = indices[partition].to(device)
        loss, accuracy, auc = _evaluate(head, x[subset], y[subset], loss_fn)
        metrics[f"{partition}_loss"] = loss
        metrics[f"{partition}_accuracy"] = accuracy
        metrics[f"{partition}_auc"] = auc
    return head.cpu(), metrics


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=METRIC_COLUMNS, delimiter="\t", lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train_all(
    stimulus_manifest: str | Path,
    checkpoint_root: str | Path,
    output_root: str | Path,
    feature_table_path: str | Path | None = None,
    options: TrainingOptions = TrainingOptions(),
) -> list[dict[str, object]]:
    """Train paired-seed linear readouts for four tasks and all 30 checkpoints."""
    device = resolve_device(options.device)
    print(f"Adaptation device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
    items = load_construction_manifest(stimulus_manifest)
    checkpoints = discover_checkpoints(checkpoint_root)
    feature_path = Path(feature_table_path) if feature_table_path else discover_feature_table(checkpoint_root)
    features = FeatureTable.from_json(feature_path)
    first_payload = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in first_payload["vocabulary"].items()}
    output_root = Path(output_root)
    metrics_path = output_root / "all_task_checkpoint_metrics.tsv"
    existing: list[dict[str, object]] = []
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle, delimiter="\t"))
    completed = {(row["task_name"], row["checkpoint_id"], int(row["initialization_seed"])) for row in existing}
    metrics_rows = existing
    manifest_sha256 = hashlib.sha256(Path(stimulus_manifest).read_bytes()).hexdigest()

    task_batches = {}
    for task_index, task in enumerate(TASK_ORDER):
        task_items = [item for item in items if item.task_name == task]
        batch = build_adaptation_batch(
            task_items, features, vocabulary,
            seed=options.input_noise_seed + task_index,
            noise_sigma=options.noise_sigma, device="cpu",
        )
        partitions = {
            partition: torch.tensor([
                index for index, item in enumerate(task_items) if item.metadata["partition"] == partition
            ], dtype=torch.long)
            for partition in PARTITIONS
        }
        task_batches[task] = (task_items, batch, partitions)

    run_manifest = {
        "schema_version": 1,
        "stimulus_manifest": str(Path(stimulus_manifest).resolve()),
        "stimulus_sha256": manifest_sha256,
        "feature_table": str(feature_path.resolve()),
        "feature_table_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "checkpoint_order": [
            {"checkpoint_id": checkpoint_id, "path": str(path.resolve()), "hours": hours}
            for checkpoint_id, path, hours in checkpoints
        ],
        "options": asdict(options),
        "scientific_constraints": {
            "encoder_frozen": True, "readout": "Linear(hidden_dim,1)",
            "loss": "BCEWithLogitsLoss", "early_stopping": "validation_loss",
            "child_age_used": False, "child_rt_used": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    progress = tqdm(checkpoints, desc="Checkpoints", unit="checkpoint", dynamic_ncols=True)
    for checkpoint_id, checkpoint_path, checkpoint_hours in progress:
        encoder = FrozenGRUEncoder.from_checkpoint(
            checkpoint_path, features.width, device=device,
            expected_hidden_dim=int(first_payload["config"]["hidden_size"]),
        )
        if encoder.vocabulary != vocabulary:
            raise ValueError(f"{checkpoint_id}: phoneme vocabulary differs from M01")
        snapshot = encoder.parameter_snapshot()
        for task in TASK_ORDER:
            task_items, batch, partitions = task_batches[task]
            representations = _encode(
                encoder, batch.frames, batch.lengths, options.encoding_batch_size, device,
            )
            labels = batch.labels.cpu()
            for initialization_seed in options.initialization_seeds:
                key = (task, checkpoint_id, initialization_seed)
                if key in completed:
                    continue
                head, measured = _train_head(
                    representations, labels, partitions, encoder.hidden_dim,
                    initialization_seed, options, device,
                )
                head_path = (
                    output_root / f"task-{task.lower()}"
                    / f"init-seed-{initialization_seed:010d}" / "heads"
                    / f"{checkpoint_id}_linear_readout.pt"
                )
                head_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "state_dict": head.state_dict(), "task_name": task,
                    "checkpoint_id": checkpoint_id, "checkpoint_hours": checkpoint_hours,
                    "initialization_seed": initialization_seed,
                    "stimulus_sha256": manifest_sha256, "metrics": measured,
                }, head_path)
                row = {
                    "task_name": task, "checkpoint_id": checkpoint_id,
                    "checkpoint_file": checkpoint_path.name,
                    "checkpoint_hours": checkpoint_hours,
                    "initialization_seed": initialization_seed,
                    "input_noise_seed": options.input_noise_seed,
                    **measured,
                    "number_train_items": len(partitions["train"]),
                    "number_validation_items": len(partitions["validation"]),
                    "number_test_items": len(partitions["test"]),
                }
                metrics_rows.append(row)
                _write_metrics(metrics_path, metrics_rows)
                tqdm.write(
                    f"{checkpoint_id} {task} seed={initialization_seed} "
                    f"best_epoch={measured['best_epoch']} val_acc={measured['validation_accuracy']:.3f} "
                    f"test_acc={measured['test_accuracy']:.3f}"
                )
            del representations
        encoder.assert_unchanged(snapshot)
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return metrics_rows
