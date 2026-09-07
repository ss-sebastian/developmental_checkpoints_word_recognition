from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
import traceback
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from devlm.features import FeatureTable
from devlm.train import resolve_device

from .input import build_adaptation_batch
from .model import FrozenGRUEncoder, make_binary_readout, task_loss
from .schema import AdaptationItem
from .train import PARTITIONS, discover_checkpoints, discover_feature_table, validate_partition_matching


EXPECTED_ITEM_COUNT = 1000
FORMAL_CONDITION_COUNTS = {
    "train": {"High": 145, "Low": 145, "Unrelated": 290},
    "validation": {"High": 40, "Low": 40, "Unrelated": 80},
    "test": {"High": 65, "Low": 65, "Unrelated": 130},
}
SUMMARY_COLUMNS = (
    "checkpoint_id", "train_n", "checkpoint_hours", "best_epoch", "epochs_run",
    "best_train_loss", "best_validation_loss", "test_loss",
    "train_accuracy", "validation_accuracy", "test_accuracy",
    "train_balanced_accuracy", "validation_balanced_accuracy", "test_balanced_accuracy",
    "train_auc", "validation_auc", "test_auc",
    "train_precision", "validation_precision", "test_precision",
    "train_recall", "validation_recall", "test_recall",
    "train_f1", "validation_f1", "test_f1",
    "train_TP", "train_TN", "train_FP", "train_FN",
    "validation_TP", "validation_TN", "validation_FP", "validation_FN",
    "test_TP", "test_TN", "test_FP", "test_FN",
)
HISTORY_COLUMNS = (
    "checkpoint_id", "train_n", "epoch", "train_loss", "validation_loss",
    "train_accuracy", "validation_accuracy", "train_balanced_accuracy",
    "validation_balanced_accuracy", "train_auc", "validation_auc", "learning_rate",
    "weight_norm", "bias_value", "gradient_norm", "best_validation_loss_so_far",
    "is_best_epoch", "seconds_this_epoch", "cumulative_seconds",
)
PREDICTION_COLUMNS = (
    "checkpoint_id", "train_n", "split", "item_id", "cue", "target", "condition",
    "true_binary_label", "logit", "probability_related", "predicted_binary_label",
    "correct", "FSG", "BSG",
)
CONDITION_COLUMNS = (
    "checkpoint_id", "train_n", "split", "comparison", "n_items", "accuracy",
    "auc", "mean_probability_related", "mean_logit",
)
STATUS_COLUMNS = ("checkpoint_id", "train_n", "status", "failure_reason")


@dataclass(frozen=True)
class MeaningProbeOptions:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    encoding_batch_size: int = 128
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    noise_sigma: float = 0.05
    input_noise_seed: int = 20260904
    initialization_seed: int = 1729
    device: str = "auto"
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.batch_size < 1 or self.encoding_batch_size < 1:
            raise ValueError("learning_rate, batch_size, and encoding_batch_size must be positive")
        if self.max_epochs < 1 or self.patience < 1 or self.min_delta < 0:
            raise ValueError("max_epochs/patience must be positive and min_delta must be non-negative")
        if self.noise_sigma < 0 or self.initialization_seed < 0 or self.input_noise_seed < 0:
            raise ValueError("noise_sigma and random seeds must be non-negative")


@dataclass(frozen=True)
class MeaningItem:
    adaptation_item: AdaptationItem
    split: str
    condition: str
    cue: str
    target: str
    fsg: float
    bsg: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_ipa(raw: str, item_id: str, field: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{item_id}: invalid {field} JSON: {exc}") from exc
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{item_id}: {field} must be a non-empty JSON array of IPA tokens")
    return tuple(value)


def _number(raw: str, item_id: str, field: str) -> float:
    if raw.strip() == "":
        raise ValueError(f"{item_id}: {field} is required")
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{item_id}: {field} must be numeric or blank") from exc


def load_meaning_manifest(path: str | Path, expected_count: int = EXPECTED_ITEM_COUNT) -> list[MeaningItem]:
    """Load the final Meaning-only dataset and reject anything but the audited design."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {
        "item_id", "task", "condition", "binary_label", "split",
        "word1_ipa", "word2_ipa", "FSG", "BSG",
    }
    if not rows:
        raise ValueError("Meaning stimulus manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Meaning stimulus manifest is missing columns: {', '.join(sorted(missing))}")
    if not ({"cue", "target"}.issubset(rows[0]) or {"word1", "word2"}.issubset(rows[0])):
        raise ValueError("Meaning stimulus manifest must contain cue/target or word1/word2 columns")
    if len(rows) != expected_count:
        raise ValueError(f"Final Meaning dataset must contain exactly {expected_count} items; found {len(rows)}")
    validate_partition_matching(rows)
    items: list[MeaningItem] = []
    seen: set[str] = set()
    for row in rows:
        item_id = row["item_id"].strip()
        if not item_id or item_id in seen:
            raise ValueError(f"Duplicate or empty item_id: {item_id!r}")
        if row["task"].strip().lower() != "meaning":
            raise ValueError(f"{item_id}: the formal probe accepts Meaning items only")
        split = row["split"].strip()
        if split not in PARTITIONS:
            raise ValueError(f"{item_id}: invalid split {split!r}")
        label = int(row["binary_label"])
        if label not in (0, 1):
            raise ValueError(f"{item_id}: binary_label must be 0 or 1")
        condition = row["condition"].strip()
        canonical = condition.lower().replace("-", "_").replace(" ", "_")
        if canonical in {"high", "high_association"}:
            expected_label = 1
        elif canonical in {"low", "low_association"}:
            expected_label = 1
        elif canonical == "unrelated":
            expected_label = 0
        else:
            raise ValueError(f"{item_id}: condition must be High, Low, or Unrelated; found {condition!r}")
        if label != expected_label:
            raise ValueError(f"{item_id}: condition {condition!r} conflicts with binary_label={label}")
        prime = _parse_ipa(row["word1_ipa"], item_id, "word1_ipa")
        target = _parse_ipa(row["word2_ipa"], item_id, "word2_ipa")
        adaptation = AdaptationItem(
            item_id=item_id, task_name="Meaning", stimulus_kind="word_pair",
            binary_label=label, split_group=item_id,
            metadata={"partition": split, "condition": condition},
            prime_phonemes=prime, target_phonemes=target,
        )
        cue = row.get("cue", row.get("word1", "")).strip()
        target = row.get("target", row.get("word2", "")).strip()
        if not cue or not target:
            raise ValueError(f"{item_id}: cue and target are required")
        fsg = _number(row["FSG"], item_id, "FSG")
        bsg = _number(row["BSG"], item_id, "BSG")
        if canonical in {"high", "high_association"} and not 0.40 <= fsg <= 0.85:
            raise ValueError(f"{item_id}: High FSG must be 0.40–0.85; found {fsg}")
        if canonical in {"low", "low_association"} and not 0.14 <= fsg <= 0.39:
            raise ValueError(f"{item_id}: Low FSG must be 0.14–0.39; found {fsg}")
        if canonical == "unrelated" and (fsg != 0 or bsg != 0):
            raise ValueError(f"{item_id}: Unrelated FSG and BSG must both be zero")
        items.append(MeaningItem(
            adaptation_item=adaptation, split=split, condition=condition,
            cue=cue, target=target,
            fsg=fsg, bsg=bsg,
        ))
        seen.add(item_id)
    for split in PARTITIONS:
        subset = [item for item in items if item.split == split]
        observed = {name: 0 for name in ("High", "Low", "Unrelated")}
        for item in subset:
            canonical = item.condition.lower().replace("-", "_").replace(" ", "_")
            name = {"high": "High", "high_association": "High", "low": "Low",
                    "low_association": "Low", "unrelated": "Unrelated"}[canonical]
            observed[name] += 1
        if observed != FORMAL_CONDITION_COUNTS[split]:
            raise ValueError(
                f"Meaning {split} condition counts must be {FORMAL_CONDITION_COUNTS[split]}; found {observed}"
            )
    return items


def _set_seed(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def _auc(y: torch.Tensor, scores: torch.Tensor, warn: Callable[[str], None], context: str) -> float:
    labels = y.detach().cpu().numpy().astype(np.int64)
    values = scores.detach().cpu().numpy().astype(np.float64)
    positive, negative = labels == 1, labels == 0
    if not positive.any() or not negative.any():
        warn(f"undefined AUC ({context}): only one label is present")
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _metrics(y: torch.Tensor, logits: torch.Tensor, loss_fn: nn.Module,
             warn: Callable[[str], None], context: str) -> dict[str, float | int]:
    y = y.detach().cpu().float()
    logits = logits.detach().cpu().float()
    predicted = (logits >= 0).long()
    truth = y.long()
    tp = int(((predicted == 1) & (truth == 1)).sum())
    tn = int(((predicted == 0) & (truth == 0)).sum())
    fp = int(((predicted == 1) & (truth == 0)).sum())
    fn = int(((predicted == 0) & (truth == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "loss": float(loss_fn(logits, y)),
        "accuracy": float((predicted == truth).float().mean()),
        "balanced_accuracy": float((recall + specificity) / 2),
        "auc": _auc(y, logits, warn, context),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


@torch.no_grad()
def _evaluate_once(head: nn.Linear, x: torch.Tensor, y: torch.Tensor, indices: torch.Tensor,
                   loss_fn: nn.Module, warn: Callable[[str], None], context: str) -> tuple[dict[str, float | int], torch.Tensor]:
    logits = head(x[indices]).squeeze(-1)
    return _metrics(y[indices], logits, loss_fn, warn, context), logits.detach().cpu()


def _gradient_norm(head: nn.Linear) -> float:
    total = sum(float(parameter.grad.detach().pow(2).sum()) for parameter in head.parameters() if parameter.grad is not None)
    return math.sqrt(total)


def train_meaning_head(
    representations: torch.Tensor,
    labels: torch.Tensor,
    indices: dict[str, torch.Tensor],
    checkpoint_id: str,
    options: MeaningProbeOptions,
    device: torch.device,
    warn: Callable[[str], None],
) -> tuple[nn.Linear, list[dict[str, object]], dict[str, object], dict[str, torch.Tensor]]:
    """Train without touching test data; evaluate test exactly once after restoration."""
    _set_seed(options.initialization_seed, options.deterministic_algorithms)
    train_indices = indices["train"]
    train_raw = representations[train_indices]
    center = train_raw.mean(0)
    scale = train_raw.std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    x = ((representations - center) / scale).to(device)
    y = labels.to(device)
    indices_device = {name: value.to(device) for name, value in indices.items()}
    head = make_binary_readout(representations.shape[1], options.initialization_seed).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    loss_fn = task_loss()
    best_loss = float("inf")
    best_state = deepcopy(head.state_dict())
    best_epoch = 0
    patience_reference = float("inf")
    stale = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, options.max_epochs + 1):
        epoch_started = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(options.initialization_seed + epoch)
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        head.train()
        gradient_norms: list[float] = []
        for start in range(0, len(permutation), options.batch_size):
            batch = permutation[start:start + options.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(x[batch]).squeeze(-1), y[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{checkpoint_id}: NaN/Inf training loss at epoch {epoch}")
            loss.backward()
            gradient_norms.append(_gradient_norm(head))
            optimizer.step()
        head.eval()
        with torch.no_grad():
            train_logits = head(x[indices_device["train"]]).squeeze(-1)
            validation_logits = head(x[indices_device["validation"]]).squeeze(-1)
        train_metrics = _metrics(y[indices_device["train"]], train_logits, loss_fn, warn, f"{checkpoint_id}:train:epoch{epoch}")
        validation_metrics = _metrics(y[indices_device["validation"]], validation_logits, loss_fn, warn, f"{checkpoint_id}:validation:epoch{epoch}")
        validation_loss = float(validation_metrics["loss"])
        if not math.isfinite(validation_loss):
            raise FloatingPointError(f"{checkpoint_id}: NaN/Inf validation loss at epoch {epoch}")
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, deepcopy(head.state_dict())
        if validation_loss < patience_reference - options.min_delta:
            patience_reference, stale = validation_loss, 0
        else:
            stale += 1
        elapsed = time.perf_counter() - started
        history.append({
            "checkpoint_id": checkpoint_id, "train_n": len(train_indices), "epoch": epoch,
            "train_loss": train_metrics["loss"], "validation_loss": validation_loss,
            "train_accuracy": train_metrics["accuracy"], "validation_accuracy": validation_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "train_auc": train_metrics["auc"], "validation_auc": validation_metrics["auc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "weight_norm": float(head.weight.detach().norm()), "bias_value": float(head.bias.detach().item()),
            "gradient_norm": float(np.mean(gradient_norms)),
            "best_validation_loss_so_far": best_loss, "is_best_epoch": False,
            "seconds_this_epoch": time.perf_counter() - epoch_started, "cumulative_seconds": elapsed,
        })
        if stale >= options.patience:
            break
    if stale < options.patience:
        warn(f"early stopping not triggered ({checkpoint_id}); reached max_epochs={options.max_epochs}")
    for row in history:
        row["is_best_epoch"] = int(row["epoch"] == best_epoch)
    head.load_state_dict(best_state)
    head.eval()
    restored: dict[str, object] = {"best_epoch": best_epoch, "epochs_run": len(history)}
    logits_by_split: dict[str, torch.Tensor] = {}
    # This is the only post-training evaluation block. In particular, the test
    # split is called exactly once and never appears in the epoch loop above.
    for split in PARTITIONS:
        measured, logits = _evaluate_once(
            head, x, y, indices_device[split], loss_fn, warn, f"{checkpoint_id}:{split}:restored",
        )
        logits_by_split[split] = logits
        for name, value in measured.items():
            prefix = "best_" if split in {"train", "validation"} and name == "loss" else ""
            key = f"{prefix}{split}_{name}" if prefix else f"{split}_{name}"
            restored[key] = value
    # Fold train-only standardization into the stored raw-state linear head.
    head = head.cpu()
    with torch.no_grad():
        standardized_weight = head.weight.detach().clone()
        head.weight.copy_(standardized_weight / scale.unsqueeze(0))
        head.bias.copy_(head.bias - (standardized_weight * center.unsqueeze(0) / scale.unsqueeze(0)).sum(1))
    return head, history, restored, logits_by_split


def _condition_name(raw: str) -> str:
    canonical = raw.lower().replace("-", "_").replace(" ", "_")
    return {"high_association": "High", "high": "High", "low_association": "Low", "low": "Low", "unrelated": "Unrelated"}[canonical]


def _predictions_and_conditions(
    checkpoint_id: str, items: list[MeaningItem], indices: dict[str, torch.Tensor],
    logits_by_split: dict[str, torch.Tensor], warn: Callable[[str], None],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_n = len(indices["train"])
    predictions: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    for split in PARTITIONS:
        item_indices = indices[split].tolist()
        logits = logits_by_split[split]
        probabilities = torch.sigmoid(logits)
        labels = torch.tensor([items[index].adaptation_item.binary_label for index in item_indices])
        for position, index in enumerate(item_indices):
            item = items[index]
            predicted = int(logits[position] >= 0)
            predictions.append({
                "checkpoint_id": checkpoint_id, "train_n": train_n, "split": split,
                "item_id": item.adaptation_item.item_id, "cue": item.cue, "target": item.target,
                "condition": _condition_name(item.condition),
                "true_binary_label": item.adaptation_item.binary_label,
                "logit": float(logits[position]), "probability_related": float(probabilities[position]),
                "predicted_binary_label": predicted,
                "correct": int(predicted == item.adaptation_item.binary_label),
                "FSG": item.fsg, "BSG": item.bsg,
            })
        names = [_condition_name(items[index].condition) for index in item_indices]
        for name in ("High", "Low", "Unrelated"):
            selected = torch.tensor([i for i, value in enumerate(names) if value == name], dtype=torch.long)
            if len(selected) == 0:
                warn(f"missing item ({checkpoint_id}:{split}:{name})")
                conditions.append({"checkpoint_id": checkpoint_id, "train_n": train_n, "split": split,
                                   "comparison": name, "n_items": 0, "accuracy": float("nan"),
                                   "auc": float("nan"), "mean_probability_related": float("nan"),
                                   "mean_logit": float("nan")})
                continue
            selected_logits, selected_labels = logits[selected], labels[selected]
            conditions.append({
                "checkpoint_id": checkpoint_id, "train_n": train_n, "split": split,
                "comparison": name, "n_items": len(selected),
                "accuracy": float(((selected_logits >= 0).long() == selected_labels).float().mean()),
                # A single condition has only one binary label, so ROC-AUC is
                # inapplicable rather than unexpectedly undefined.
                "auc": float("nan"),
                "mean_probability_related": float(torch.sigmoid(selected_logits).mean()),
                "mean_logit": float(selected_logits.mean()),
            })
        for related in ("High", "Low"):
            selected = torch.tensor([i for i, value in enumerate(names) if value in {related, "Unrelated"}], dtype=torch.long)
            selected_logits, selected_labels = logits[selected], labels[selected]
            conditions.append({
                "checkpoint_id": checkpoint_id, "train_n": train_n, "split": split,
                "comparison": f"{related}_vs_Unrelated", "n_items": len(selected),
                "accuracy": float(((selected_logits >= 0).long() == selected_labels).float().mean()),
                "auc": _auc(selected_labels, selected_logits, warn, f"{checkpoint_id}:{split}:{related}_vs_Unrelated"),
                "mean_probability_related": float(torch.sigmoid(selected_logits).mean()),
                "mean_logit": float(selected_logits.mean()),
            })
    return predictions, conditions


@torch.no_grad()
def _encode(encoder: FrozenGRUEncoder, frames: torch.Tensor, lengths: torch.Tensor,
            batch_size: int, device: torch.device) -> torch.Tensor:
    chunks = []
    for start in range(0, len(frames), batch_size):
        end = min(len(frames), start + batch_size)
        chunks.append(encoder(frames[start:end].to(device), lengths[start:end].to(device)).cpu())
    return torch.cat(chunks).float()


def _environment(device: torch.device) -> dict[str, object]:
    return {
        "python_version": sys.version, "platform": platform.platform(),
        "pytorch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device": str(device),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _make_zip(output_root: Path) -> Path:
    zip_path = output_root.parent / "meaning_probe_results.zip"
    temporary_base = output_root.parent / ".meaning_probe_results"
    made = Path(shutil.make_archive(str(temporary_base), "zip", root_dir=output_root))
    made.replace(zip_path)
    return zip_path


def run_meaning_probe(
    stimulus_manifest: str | Path,
    checkpoint_root: str | Path,
    output_root: str | Path,
    feature_table_path: str | Path | None = None,
    options: MeaningProbeOptions = MeaningProbeOptions(),
) -> Path:
    """Run the auditable 30-checkpoint, 1000-item formal Meaning probe."""
    started_wall = datetime.now(timezone.utc)
    started_perf = time.perf_counter()
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Formal Meaning-probe output directory must be new or empty to prevent stale artifact mixing: {output_root}"
        )
    for name in ("summary", "histories", "predictions", "heads", "cached_hidden", "configs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    log_path = output_root / "run_log.txt"
    log_path.write_text("", encoding="utf-8")
    warning_messages: list[str] = []

    def log(message: str) -> None:
        timestamped = f"{datetime.now(timezone.utc).isoformat()} {message}"
        print(timestamped, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")

    def warn(message: str) -> None:
        warning_messages.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        log("WARNING: " + message)

    device = resolve_device(options.device)
    _set_seed(options.initialization_seed, options.deterministic_algorithms)
    environment = _environment(device)
    environment["runtime_start_utc"] = started_wall.isoformat()
    environment["random_seeds"] = {
        "initialization_seed": options.initialization_seed,
        "input_noise_seed": options.input_noise_seed,
    }
    _write_json(output_root / "configs" / "environment.json", environment)
    log(f"Meaning probe device: {device}" + (f" ({environment['gpu_model']})" if environment["gpu_model"] else ""))

    items = load_meaning_manifest(stimulus_manifest)
    checkpoints = discover_checkpoints(checkpoint_root)
    feature_path = Path(feature_table_path) if feature_table_path else discover_feature_table(checkpoint_root)
    features = FeatureTable.from_json(feature_path)
    dataset_sha = _sha256(Path(stimulus_manifest))
    training_config = asdict(options)
    training_config_sha = _canonical_sha256(training_config)
    _write_json(output_root / "configs" / "training_config.json", {
        **training_config, "training_config_sha256": training_config_sha,
        "gradient_norm_definition": "mean global L2 norm across minibatches in the epoch",
        "selection_rule": "minimum validation loss only", "test_evaluations_per_probe": 1,
    })
    split_counts = {split: sum(item.split == split for item in items) for split in PARTITIONS}
    condition_counts = {name: sum(_condition_name(item.condition) == name for item in items)
                        for name in ("High", "Low", "Unrelated")}
    _write_json(output_root / "configs" / "dataset_manifest.json", {
        "path": str(Path(stimulus_manifest).resolve()), "dataset_sha256": dataset_sha,
        "n_items": len(items), "split_counts": split_counts, "condition_counts": condition_counts,
        "item_ids": [item.adaptation_item.item_id for item in items],
    })
    checkpoint_records = []
    for checkpoint_id, path, hours in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_records.append({
            "checkpoint_id": checkpoint_id, "path": str(path.resolve()), "hours": hours,
            "checkpoint_sha256": _sha256(path),
            "phase1_training_config_sha256": _canonical_sha256(payload.get("config", {})),
        })
    _write_json(output_root / "configs" / "checkpoint_manifest.json", {"checkpoints": checkpoint_records})

    first_payload = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in first_payload["vocabulary"].items()}
    batch = build_adaptation_batch(
        [item.adaptation_item for item in items], features, vocabulary,
        seed=options.input_noise_seed, noise_sigma=options.noise_sigma, device="cpu",
    )
    frame_digest = hashlib.sha256()
    frame_digest.update(np.ascontiguousarray(batch.frames.numpy()).tobytes())
    frame_digest.update(np.ascontiguousarray(batch.lengths.numpy()).tobytes())
    frame_digest.update("\n".join(batch.item_ids).encode())
    stimulus_frame_sha = frame_digest.hexdigest()
    _write_json(output_root / "configs" / "dataset_manifest.json", {
        "path": str(Path(stimulus_manifest).resolve()), "dataset_sha256": dataset_sha,
        "stimulus_frame_sha256": stimulus_frame_sha,
        "n_items": len(items), "split_counts": split_counts, "condition_counts": condition_counts,
        "item_ids": [item.adaptation_item.item_id for item in items],
    })
    indices = {
        split: torch.tensor([index for index, item in enumerate(items) if item.split == split], dtype=torch.long)
        for split in PARTITIONS
    }
    summary_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    active_checkpoint = "not_started"
    try:
        progress = tqdm(checkpoints, desc="Meaning checkpoints", unit="checkpoint", dynamic_ncols=True)
        for checkpoint_id, checkpoint_path, checkpoint_hours in progress:
            active_checkpoint = checkpoint_id
            checkpoint_sha = next(row["checkpoint_sha256"] for row in checkpoint_records if row["checkpoint_id"] == checkpoint_id)
            try:
                encoder = FrozenGRUEncoder.from_checkpoint(
                    checkpoint_path, features.width, device=device,
                    expected_hidden_dim=int(first_payload["config"]["hidden_size"]),
                )
                if encoder.vocabulary != vocabulary:
                    raise ValueError(f"{checkpoint_id}: phoneme vocabulary differs from M01")
                snapshot = encoder.parameter_snapshot()
                h_final = _encode(encoder, batch.frames, batch.lengths, options.encoding_batch_size, device)
                if h_final.shape != (EXPECTED_ITEM_COUNT, encoder.hidden_dim):
                    raise ValueError(f"{checkpoint_id}: hidden cache shape {tuple(h_final.shape)} is not "
                                     f"({EXPECTED_ITEM_COUNT}, {encoder.hidden_dim})")
                np.savez(
                    output_root / "cached_hidden" / f"{checkpoint_id}_hfinal.npz",
                    item_ids=np.asarray(batch.item_ids), h_final=h_final.numpy().astype(np.float32),
                    dataset_sha256=np.asarray(dataset_sha), checkpoint_sha256=np.asarray(checkpoint_sha),
                    stimulus_frame_sha256=np.asarray(stimulus_frame_sha),
                )
                head, history, measured, logits_by_split = train_meaning_head(
                    h_final, batch.labels.cpu(), indices, checkpoint_id, options, device, warn,
                )
                encoder.assert_unchanged(snapshot)
                predictions, per_condition = _predictions_and_conditions(
                    checkpoint_id, items, indices, logits_by_split, warn,
                )
                train_n = len(indices["train"])
                torch.save({
                    "state_dict": head.state_dict(), "checkpoint_id": checkpoint_id,
                    "checkpoint_sha256": checkpoint_sha, "dataset_sha256": dataset_sha,
                    "stimulus_frame_sha256": stimulus_frame_sha, "training_config_sha256": training_config_sha,
                    "initialization_seed": options.initialization_seed, "train_n": train_n,
                    "best_epoch": measured["best_epoch"],
                }, output_root / "heads" / f"{checkpoint_id}_linear_probe.pt")
                _write_csv(output_root / "histories" / f"{checkpoint_id}_history.csv", history, HISTORY_COLUMNS)
                _write_csv(output_root / "predictions" / f"{checkpoint_id}_predictions.tsv",
                           predictions, PREDICTION_COLUMNS, delimiter="\t")
                summary = {"checkpoint_id": checkpoint_id, "train_n": train_n,
                           "checkpoint_hours": checkpoint_hours, **measured}
                summary_rows.append(summary)
                condition_rows.extend(per_condition)
                status_rows.append({"checkpoint_id": checkpoint_id, "train_n": train_n,
                                    "status": "success", "failure_reason": ""})
                log(f"SUCCESS {checkpoint_id}: best_epoch={measured['best_epoch']} "
                    f"validation_auc={measured['validation_auc']:.4f} test_auc={measured['test_auc']:.4f}")
                del encoder, h_final
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                status_rows.append({"checkpoint_id": checkpoint_id, "train_n": len(indices["train"]),
                                    "status": "failed", "failure_reason": reason})
                log(f"FAILED {checkpoint_id}: {reason}\n{traceback.format_exc()}")
            _write_csv(output_root / "summary" / "checkpoint_probe_metrics.csv", summary_rows, SUMMARY_COLUMNS)
            _write_csv(output_root / "summary" / "condition_metrics.csv", condition_rows, CONDITION_COLUMNS)
            _write_csv(output_root / "summary" / "run_status.csv", status_rows, STATUS_COLUMNS)
    except KeyboardInterrupt:
        recorded = {row["checkpoint_id"] for row in status_rows}
        for checkpoint_id, _, _ in checkpoints:
            if checkpoint_id not in recorded:
                reason = ("interrupted run (KeyboardInterrupt)" if checkpoint_id == active_checkpoint
                          else "not attempted because run was interrupted")
                status_rows.append({"checkpoint_id": checkpoint_id, "train_n": len(indices["train"]),
                                    "status": "failed", "failure_reason": reason})
        _write_csv(output_root / "summary" / "run_status.csv", status_rows, STATUS_COLUMNS)
        log(f"INTERRUPTED during {active_checkpoint}")
        raise
    finally:
        ended = datetime.now(timezone.utc)
        environment.update({
            "runtime_end_utc": ended.isoformat(),
            "total_runtime_seconds": time.perf_counter() - started_perf,
        })
        _write_json(output_root / "configs" / "environment.json", environment)
        run_manifest = {
            "schema_version": 1, "dataset_sha256": dataset_sha,
            "stimulus_frame_sha256": stimulus_frame_sha,
            "training_config_sha256": training_config_sha,
            "checkpoint_sha256": {row["checkpoint_id"]: row["checkpoint_sha256"] for row in checkpoint_records},
            "expected_checkpoint_ids": [row[0] for row in checkpoints],
            "status_counts": {
                "success": sum(row["status"] == "success" for row in status_rows),
                "failed": sum(row["status"] == "failed" for row in status_rows),
            },
            "warnings": warning_messages,
            "test_policy": "test evaluated exactly once after validation-selected best head restoration",
        }
        _write_json(output_root / "run_manifest.json", run_manifest)
        log(f"Creating final bundle in {output_root.parent}")
        zip_path = _make_zip(output_root)
        print(f"Bundle written: {zip_path}", flush=True)
    failures = [row for row in status_rows if row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Meaning probe finished with {len(failures)} failed checkpoint(s); see {zip_path}")
    return zip_path
