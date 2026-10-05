from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from devlm.data import Session, Utterance
from devlm.features import FeatureTable
from devlm.model import CausalPhonemeGRU
from devlm.stream import build_session_stream
from devlm.train import resolve_device

from .model import FrozenGRUEncoder, make_binary_readout, task_loss
from .train import PARTITIONS, discover_checkpoints, discover_feature_table


SOUND_CONDITIONS = {"onset", "rhyme", "unrelated"}
DEFAULT_INITIALIZATION_SEEDS = (1729, 2718, 3141)


@dataclass(frozen=True)
class SoundProbeOptions:
    input_noise_seed: int = 20261004
    initialization_seeds: tuple[int, ...] = DEFAULT_INITIALIZATION_SEEDS
    noise_sigma: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    encoding_batch_size: int = 128
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    bootstrap_repetitions: int = 1000
    max_word_reuse: int = 40
    max_component_words: int = 50
    device: str = "auto"
    deterministic_algorithms: bool = False

    def __post_init__(self) -> None:
        if not self.initialization_seeds or len(set(self.initialization_seeds)) != len(self.initialization_seeds):
            raise ValueError("initialization_seeds must be a non-empty sequence of unique seeds")
        if min(self.initialization_seeds) < 0 or self.input_noise_seed < 0 or self.noise_sigma < 0:
            raise ValueError("Seeds and noise_sigma must be non-negative")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.encoding_batch_size < 1:
            raise ValueError("Learning rate and batch sizes must be positive")
        if self.max_epochs < 1 or self.patience < 1 or self.min_delta < 0:
            raise ValueError("Epoch/patience values must be positive and min_delta non-negative")
        if self.bootstrap_repetitions < 1 or self.max_word_reuse < 1 or self.max_component_words < 2:
            raise ValueError("Bootstrap and lexical limits must be positive")


@dataclass(frozen=True)
class SoundItem:
    item_id: str
    split: str
    condition: str
    label: int
    word1: str
    word2: str
    word1_ipa: tuple[str, ...]
    word2_ipa: tuple[str, ...]
    split_group: str


def _parse_ipa(raw: str, item_id: str, field: str) -> tuple[str, ...]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{item_id}: {field} must be a JSON array of IPA tokens") from exc
    if not isinstance(value, list) or not value or any(not isinstance(token, str) or not token for token in value):
        raise ValueError(f"{item_id}: {field} must be a non-empty JSON array of IPA strings")
    return tuple(value)


def _lexical_components(items: list[SoundItem]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for item in items:
        adjacency[item.word1].add(item.word2)
        adjacency[item.word2].add(item.word1)
    components: list[set[str]] = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        component, stack = set(), [root]
        while stack:
            word = stack.pop()
            if word in component:
                continue
            component.add(word)
            stack.extend(adjacency[word] - component)
        remaining -= component
        components.append(component)
    return components


def load_sound_manifest(
    path: str | Path,
    *,
    max_word_reuse: int = 40,
    max_component_words: int = 50,
) -> list[SoundItem]:
    """Strictly validate a preassigned Sound-only manifest; never select from test."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("Sound manifest is empty")
    required = {
        "item_id", "task", "condition", "binary_label", "split", "word1", "word2",
        "word1_ipa", "word2_ipa", "word1_n_syllables", "word2_n_syllables",
        "childes_count_word1", "childes_count_word2", "qc_pass", "phonological_relation_verified",
        "shared_onset", "shared_rime", "shared_phoneme_count",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Sound manifest is missing columns: {', '.join(sorted(missing))}")
    items: list[SoundItem] = []
    seen: set[str] = set()
    for row in rows:
        item_id = row["item_id"].strip()
        if not item_id or item_id in seen:
            raise ValueError(f"Duplicate or empty item_id: {item_id!r}")
        if row["task"].strip() != "Sound":
            raise ValueError(f"{item_id}: Sound-only probe rejects task={row['task']!r}")
        split, condition = row["split"].strip(), row["condition"].strip().lower()
        if split not in PARTITIONS or condition not in SOUND_CONDITIONS:
            raise ValueError(f"{item_id}: invalid split/condition {split!r}/{condition!r}")
        try:
            label = int(row["binary_label"])
            syllables1, syllables2 = int(row["word1_n_syllables"]), int(row["word2_n_syllables"])
            count1, count2 = int(float(row["childes_count_word1"])), int(float(row["childes_count_word2"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{item_id}: label, syllable counts, and CHILDES counts must be numeric") from exc
        expected_label = 0 if condition == "unrelated" else 1
        if label != expected_label:
            raise ValueError(f"{item_id}: condition {condition} requires binary_label={expected_label}")
        if row["qc_pass"].strip() != "1" or row["phonological_relation_verified"].strip() != "1":
            raise ValueError(f"{item_id}: primary Sound rows must pass stimulus QC and relation verification")
        if condition == "onset" and not row["shared_onset"].strip():
            raise ValueError(f"{item_id}: onset condition lacks verified shared onset")
        if condition == "rhyme" and not row["shared_rime"].strip():
            raise ValueError(f"{item_id}: rhyme condition lacks verified shared rime")
        if condition == "unrelated" and (row["shared_onset"].strip() or row["shared_rime"].strip()):
            raise ValueError(f"{item_id}: unrelated condition shares onset/rime")
        if condition == "unrelated" and int(row["shared_phoneme_count"]) != 0:
            raise ValueError(f"{item_id}: unrelated condition must share zero phonemes")
        if syllables1 != 1 or syllables2 != 1:
            raise ValueError(f"{item_id}: both words must be one-syllable")
        if count1 <= 0 or count2 <= 0:
            raise ValueError(f"{item_id}: primary Sound words must both be Phase-1/CHILDES attested")
        word1, word2 = row["word1"].strip().lower(), row["word2"].strip().lower()
        if not word1 or not word2 or word1 == word2:
            raise ValueError(f"{item_id}: two distinct real-word forms are required")
        items.append(SoundItem(
            item_id, split, condition, label, word1, word2,
            _parse_ipa(row["word1_ipa"], item_id, "word1_ipa"),
            _parse_ipa(row["word2_ipa"], item_id, "word2_ipa"), item_id,
        ))
        seen.add(item_id)

    words_by_split: dict[str, set[str]] = {}
    for split in PARTITIONS:
        subset = [item for item in items if item.split == split]
        if not subset or {item.label for item in subset} != {0, 1}:
            raise ValueError(f"Sound {split} partition must contain positive and negative items")
        counts = Counter(item.label for item in subset)
        if counts[0] != counts[1]:
            raise ValueError(f"Sound {split} labels must be balanced; observed {dict(counts)}")
        positive_counts = Counter(item.condition for item in subset if item.label == 1)
        if abs(positive_counts["onset"] - positive_counts["rhyme"]) > 1:
            raise ValueError(f"Sound {split} onset/rhyme positives must be balanced; observed {dict(positive_counts)}")
        word_counts = Counter(word for item in subset for word in (item.word1, item.word2))
        if max(word_counts.values()) > max_word_reuse:
            raise ValueError(f"Sound {split} word reuse exceeds {max_word_reuse}")
        components = _lexical_components(subset)
        if max(map(len, components)) > max_component_words:
            raise ValueError(f"Sound {split} lexical component exceeds {max_component_words} words")
        words_by_split[split] = set(word_counts)
    for i, split_a in enumerate(PARTITIONS):
        for split_b in PARTITIONS[i + 1:]:
            overlap = words_by_split[split_a] & words_by_split[split_b]
            if overlap:
                raise ValueError(
                    f"Sound lexical leakage: {split_a}/{split_b} share {len(overlap)} words; "
                    f"example={min(overlap)!r}"
                )
    return items


def prepare_sound_manifest(
    source_path: str | Path,
    output_path: str | Path,
    *,
    max_word_reuse: int = 40,
    max_component_words: int = 50,
) -> dict[str, object]:
    """Create a deterministic, model-blind attested subset within existing splits.

    Selection sees only candidate metadata/lexical forms. It neither loads model
    checkpoints nor computes probe performance. Existing train/validation/test
    assignments are immutable and lexical overlap between them is rejected.
    """
    with Path(source_path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"item_id", "task", "condition", "binary_label", "split", "word1", "word2",
                "word1_ipa", "word2_ipa", "word1_n_syllables", "word2_n_syllables",
                "childes_count_word1", "childes_count_word2", "qc_pass", "phonological_relation_verified",
                "shared_onset", "shared_rime", "shared_phoneme_count"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Source manifest missing required columns: {sorted(required - set(rows[0] if rows else []))}")
    source_items = []
    for row in rows:
        if row["task"].strip() != "Sound":
            continue
        if row["qc_pass"].strip() != "1" or row["phonological_relation_verified"].strip() != "1":
            continue
        try:
            attested = int(float(row["childes_count_word1"])) > 0 and int(float(row["childes_count_word2"])) > 0
            single_syllable = int(row["word1_n_syllables"]) == int(row["word2_n_syllables"]) == 1
        except (TypeError, ValueError):
            continue
        if not (attested and single_syllable):
            continue
        condition = row["condition"].strip().lower()
        if condition not in SOUND_CONDITIONS:
            continue
        if condition == "onset" and not row["shared_onset"].strip():
            continue
        if condition == "rhyme" and not row["shared_rime"].strip():
            continue
        if condition == "unrelated" and (row["shared_onset"].strip() or row["shared_rime"].strip()):
            continue
        if condition == "unrelated" and int(row["shared_phoneme_count"]) != 0:
            continue
        source_items.append(row)
    # Existing split membership is retained. Reject leakage before selecting, and
    # do not repair it by moving examples across partitions.
    source_words: dict[str, set[str]] = {}
    for split in PARTITIONS:
        source_words[split] = {row[word].strip().lower() for row in source_items if row["split"].strip() == split
                               for word in ("word1", "word2")}
    for index, split_a in enumerate(PARTITIONS):
        for split_b in PARTITIONS[index + 1:]:
            if source_words[split_a] & source_words[split_b]:
                raise ValueError(f"Source Sound manifest leaks lexical items across {split_a}/{split_b}")

    selected_rows: list[dict[str, str]] = []
    achieved: dict[str, object] = {}
    for split in PARTITIONS:
        subset = [row for row in source_items if row["split"].strip() == split]
        by_condition = {name: [row for row in subset if row["condition"].strip().lower() == name]
                        for name in SOUND_CONDITIONS}
        positive_n = min(len(by_condition["onset"]), len(by_condition["rhyme"]), len(by_condition["unrelated"]) // 2)
        if positive_n < 1:
            raise ValueError(f"Not enough attested candidates to form a balanced {split} split")
        quotas = {"onset": positive_n, "rhyme": positive_n, "unrelated": 2 * positive_n}
        nuisance_columns = {
            "mean_content_subtlex": lambda row: (float(row["word1_subtlex"]) + float(row["word2_subtlex"])) / 2,
            "mean_phoneme_count": lambda row: (float(row["word1_n_phonemes"]) + float(row["word2_n_phonemes"])) / 2,
            "mean_orthographic_length": lambda row: (len(row["word1"]) + len(row["word2"])) / 2,
        }
        if any(not row.get("word1_subtlex", "").strip() or not row.get("word2_subtlex", "").strip()
               or not row.get("word1_n_phonemes", "").strip() or not row.get("word2_n_phonemes", "").strip()
               for row in subset):
            raise ValueError(f"{split}: candidate manifest lacks lexical covariates required for model-blind matching")
        best_score = float("inf")
        picked: list[dict[str, str]] = []
        selection_restarts = 3000
        search_rng = np.random.default_rng(20261005 + PARTITIONS.index(split))
        for _ in range(selection_restarts):
            trial = [row for condition in ("onset", "rhyme", "unrelated")
                     for row in search_rng.choice(by_condition[condition], size=quotas[condition], replace=False)]
            reuse_trial = Counter(word.strip().lower() for row in trial for word in (row["word1"], row["word2"]))
            trial_items = [
                SoundItem(row["item_id"], split, row["condition"].strip().lower(), int(row["binary_label"]),
                          row["word1"].strip().lower(), row["word2"].strip().lower(), (), (), row["item_id"])
                for row in trial
            ]
            trial_components = _lexical_components(trial_items)
            max_reuse_trial = max(reuse_trial.values())
            max_component_trial = max(map(len, trial_components))
            if max_reuse_trial > max_word_reuse or max_component_trial > max_component_words:
                continue
            imbalance = 0.0
            for measure in nuisance_columns.values():
                unrelated_values = np.asarray([measure(row) for row in trial if row["condition"].strip().lower() == "unrelated"])
                for condition in ("onset", "rhyme"):
                    values = np.asarray([measure(row) for row in trial if row["condition"].strip().lower() == condition])
                    pooled = math.sqrt((float(values.var(ddof=1)) + float(unrelated_values.var(ddof=1))) / 2)
                    smd = 0.0 if pooled == 0 and np.isclose(values.mean(), unrelated_values.mean()) else (
                        float("inf") if pooled == 0 else float((values.mean() - unrelated_values.mean()) / pooled)
                    )
                    imbalance += smd * smd
            # Strongly favor covariate balance, with lexical concentration as a
            # secondary cost among candidates that satisfy hard lexical caps.
            score = imbalance + 0.0005 * sum(count * count for count in reuse_trial.values()) / len(reuse_trial)
            score += 0.0001 * max_component_trial
            if score < best_score:
                best_score, picked = score, trial
        if not picked:
            raise ValueError(
                f"Model-blind candidate selection cannot satisfy max_word_reuse={max_word_reuse} and "
                f"max_component_words={max_component_words} in {split} after {selection_restarts} deterministic trials"
            )
        reuse = Counter(word.strip().lower() for row in picked for word in (row["word1"], row["word2"]))
        components = _lexical_components([
            SoundItem(r["item_id"], split, r["condition"].strip().lower(), int(r["binary_label"]),
                      r["word1"].strip().lower(), r["word2"].strip().lower(), (), (), r["item_id"])
            for r in picked
        ])
        nuisance_smd = {}
        for name, measure in nuisance_columns.items():
            if all(name != "mean_content_subtlex" or row.get("word1_subtlex", "").strip() for row in picked):
                onset_values = np.asarray([measure(row) for row in picked if row["condition"].strip().lower() == "onset"])
                rhyme_values = np.asarray([measure(row) for row in picked if row["condition"].strip().lower() == "rhyme"])
                unrelated_values = np.asarray([measure(row) for row in picked if row["condition"].strip().lower() == "unrelated"])
                for subtype, values in (("onset", onset_values), ("rhyme", rhyme_values)):
                    pooled = math.sqrt((float(values.var(ddof=1)) + float(unrelated_values.var(ddof=1))) / 2)
                    smd = 0.0 if pooled == 0 and np.isclose(values.mean(), unrelated_values.mean()) else (
                        float("inf") if pooled == 0 else float((values.mean() - unrelated_values.mean()) / pooled)
                    )
                    nuisance_smd[f"{subtype}_vs_unrelated:{name}"] = smd
        achieved[split] = {
            "condition_counts": dict(Counter(row["condition"].strip().lower() for row in picked)),
            "items": len(picked), "unique_words": len(reuse), "max_word_reuse": max(reuse.values()),
            "component_count": len(components), "max_component_words": max(map(len, components)),
            "source_attested_candidates": {key: len(value) for key, value in by_condition.items()},
            "nuisance_standardized_mean_differences": nuisance_smd,
            "model_blind_selection_objective": best_score,
        }
        if max(reuse.values()) > max_word_reuse:
            raise ValueError(f"Greedy selection cannot satisfy max_word_reuse={max_word_reuse} for {split}; achieved {max(reuse.values())}")
        if max(map(len, components)) > max_component_words:
            raise ValueError(
                f"Greedy selection cannot satisfy max_component_words={max_component_words} for {split}; "
                f"achieved {max(map(len, components))}"
            )
        selected_rows.extend(picked)

    # Store only Sound rows in an isolated location; retain original columns and
    # source assignments for auditability. Sorting is deterministic.
    selected_rows.sort(key=lambda row: (PARTITIONS.index(row["split"].strip()), row["condition"], row["item_id"]))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected_rows)
    # The final gate validates all primary constraints independently of selector.
    validated = load_sound_manifest(path, max_word_reuse=max_word_reuse, max_component_words=max_component_words)
    report = {"source_manifest": str(Path(source_path).resolve()), "output_manifest": str(path.resolve()),
              "selection_is_model_blind": True, "total_items": len(validated), "by_split": achieved}
    report_path = path.with_name("sound_manifest_qc.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_sound_frames(
    item: SoundItem,
    features: FeatureTable,
    phoneme_to_id: dict[str, int],
    rng: np.random.Generator,
    noise_sigma: float,
    phoneme_envelope: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Reuse Phase-1 stream timing: word1, 3 clean zero frames, word2, no reset."""
    utterances = tuple(
        Utterance("sound-probe", item.item_id, 0.0, order, "", word, phonemes)
        for order, (word, phonemes) in enumerate(((item.word1, item.word1_ipa), (item.word2, item.word2_ipa)), 1)
    )
    stream = build_session_stream(
        Session("sound-probe", item.item_id, 0.0, utterances), features, phoneme_to_id,
        rng, noise_sigma=noise_sigma, phoneme_envelope=phoneme_envelope,
    )
    first_word_last = max((span for span in stream.spans if span.utterance_index == 0), key=lambda span: span.end_frame)
    second_word_first = min((span for span in stream.spans if span.utterance_index == 1), key=lambda span: span.start_frame)
    pause_start = first_word_last.end_frame
    pause_end = second_word_first.start_frame
    if pause_end - pause_start != 3:
        raise AssertionError(f"Expected exactly 3 pause frames, observed {pause_end - pause_start}")
    if stream.speech_mask[pause_start:pause_end].any() or np.any(stream.noisy_frames[pause_start:pause_end] != 0):
        raise AssertionError("Pause frames must remain unnoised zero vectors")
    return stream.noisy_frames, len(stream.noisy_frames)


def encode_sound_items(
    encoder: FrozenGRUEncoder,
    items: list[SoundItem],
    features: FeatureTable,
    noise_seed: int,
    options: SoundProbeOptions,
    device: torch.device,
    phoneme_envelope: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> torch.Tensor:
    rng = np.random.default_rng(noise_seed)
    streams: list[np.ndarray] = []
    lengths: list[int] = []
    for item in items:
        frames, length = build_sound_frames(
            item, features, encoder.vocabulary, rng, options.noise_sigma,
            phoneme_envelope=phoneme_envelope,
        )
        streams.append(frames)
        lengths.append(length)
    max_frames = max(lengths)
    padded = np.zeros((len(streams), max_frames, features.width), dtype=np.float32)
    for index, stream in enumerate(streams):
        padded[index, :len(stream)] = stream
    frames_tensor = torch.from_numpy(padded)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    reps = []
    encoder.eval()
    for start in range(0, len(streams), options.encoding_batch_size):
        end = min(len(streams), start + options.encoding_batch_size)
        with torch.no_grad():
            reps.append(encoder(frames_tensor[start:end].to(device), lengths_tensor[start:end].to(device)).cpu())
    return torch.cat(reps, dim=0)


def _auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y = labels.detach().cpu().numpy().astype(np.int64)
    s = scores.detach().cpu().numpy().astype(np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and s[order[end]] == s[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _metric_values(labels: torch.Tensor, logits: torch.Tensor) -> dict[str, float]:
    labels = labels.detach().cpu().float()
    logits = logits.detach().cpu().float()
    truth = labels.long()
    predicted = (logits >= 0).long()
    sensitivity = float(((predicted == 1) & (truth == 1)).sum()) / max(1, int((truth == 1).sum()))
    specificity = float(((predicted == 0) & (truth == 0)).sum()) / max(1, int((truth == 0).sum()))
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return {
        "auc": _auc(labels, logits),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "cross_entropy": float(loss),
    }


@torch.no_grad()
def evaluate_split(head: nn.Linear, x: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    logits = head(x).squeeze(-1)
    return _metric_values(labels, logits), logits.detach().cpu()


def _set_seed(seed: int, deterministic_algorithms: bool) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def fit_sound_head(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    *,
    initialization_seed: int,
    options: SoundProbeOptions,
    device: torch.device,
    evaluation_hook: Callable[[str, nn.Linear, torch.Tensor, torch.Tensor], tuple[dict[str, float], torch.Tensor]] =
    lambda split, head, x, y: evaluate_split(head, x, y),
) -> tuple[nn.Linear, list[dict[str, object]], dict[str, object]]:
    """Fit Linear(128,1) and select only by validation loss; test is not accepted."""
    _set_seed(initialization_seed, options.deterministic_algorithms)
    center = train_x.mean(0)
    scale = train_x.std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    train_x = ((train_x - center) / scale).to(device)
    validation_x = ((validation_x - center) / scale).to(device)
    train_y, validation_y = (x.to(device) for x in (train_y, validation_y))
    head = make_binary_readout(train_x.shape[1], initialization_seed).to(device)
    if not isinstance(head, nn.Linear) or head.in_features != 128 or head.out_features != 1:
        raise ValueError("Sound probe requires exactly Linear(128, 1)")
    optimizer = torch.optim.AdamW(head.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    best_loss, best_epoch, best_state = float("inf"), 0, deepcopy(head.state_dict())
    patience_reference, stale = float("inf"), 0
    history: list[dict[str, object]] = []
    for epoch in range(1, options.max_epochs + 1):
        permutation = torch.randperm(len(train_x), generator=torch.Generator().manual_seed(initialization_seed + epoch))
        head.train()
        for start in range(0, len(permutation), options.batch_size):
            index = permutation[start:start + options.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(head(train_x[index]).squeeze(-1), train_y[index])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        head.eval()
        # Epoch loop evaluates only train and validation. Test cannot affect selection.
        train_metrics, _ = evaluation_hook("train", head, train_x, train_y)
        validation_metrics, _ = evaluation_hook("validation", head, validation_x, validation_y)
        validation_loss = validation_metrics["cross_entropy"]
        if validation_loss < best_loss:
            best_loss, best_epoch, best_state = validation_loss, epoch, deepcopy(head.state_dict())
        if validation_loss < patience_reference - options.min_delta:
            patience_reference, stale = validation_loss, 0
        else:
            stale += 1
        history.append({
            "epoch": epoch, "train_loss": train_metrics["cross_entropy"],
            "validation_loss": validation_loss, "train_auc": train_metrics["auc"],
            "validation_auc": validation_metrics["auc"],
        })
        if stale >= options.patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    # Fold the train-only standardization into the selected linear layer so the
    # saved head consumes raw frozen hidden states without an extra scaler.
    with torch.no_grad():
        scaled_weight = head.weight.detach().cpu().clone()
        scaled_bias = head.bias.detach().cpu().clone()
        head = head.cpu()
        head.weight.copy_(scaled_weight / scale.unsqueeze(0))
        head.bias.copy_(
            scaled_bias
            - (scaled_weight * center.unsqueeze(0) / scale.unsqueeze(0)).sum(dim=1)
        )
    return head, history, {
        "best_epoch": best_epoch, "epochs_run": len(history), "best_validation_loss": best_loss,
    }


def _components_per_item(items: list[SoundItem]) -> list[list[int]]:
    by_split: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        by_split[item.split].append(index)
    groups: list[list[int]] = []
    for split, indices in by_split.items():
        components = _lexical_components([items[index] for index in indices])
        for component in components:
            groups.append([index for index in indices if items[index].word1 in component])
    return groups


def paired_bootstrap_auc_delta(
    labels: torch.Tensor,
    logits_checkpoint: torch.Tensor,
    logits_baseline: torch.Tensor,
    component_groups: list[list[int]],
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    valid_groups = [group for group in component_groups if group]
    if len(valid_groups) < 2:
        return float("nan"), float("nan"), float("nan")
    labels_np = labels.detach().cpu().numpy()
    new_np, base_np = logits_checkpoint.numpy(), logits_baseline.numpy()
    for _ in range(repetitions):
        chosen = rng.integers(0, len(valid_groups), size=len(valid_groups))
        indices = np.asarray([i for group_i in chosen for i in valid_groups[group_i]], dtype=np.int64)
        if len(np.unique(labels_np[indices])) < 2:
            continue
        deltas.append(_auc(torch.tensor(labels_np[indices]), torch.tensor(new_np[indices])) -
                      _auc(torch.tensor(labels_np[indices]), torch.tensor(base_np[indices])))
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    low, high = np.quantile(deltas, [0.025, 0.975])
    point = _auc(labels, logits_checkpoint) - _auc(labels, logits_baseline)
    return float(point), float(low), float(high)


def _write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_sound_probe(
    stimuli_path: str | Path,
    checkpoints_root: str | Path,
    output_dir: str | Path,
    *,
    feature_table_path: str | Path | None = None,
    options: SoundProbeOptions = SoundProbeOptions(),
) -> dict[str, object]:
    items = load_sound_manifest(
        stimuli_path, max_word_reuse=options.max_word_reuse,
        max_component_words=options.max_component_words,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(
            f"Sound probe output directory must be empty to prevent mixing runs: {output}"
        )
    checkpoints = discover_checkpoints(checkpoints_root)
    if feature_table_path is None:
        feature_table_path = discover_feature_table(checkpoints_root)
    features = FeatureTable.from_json(feature_table_path)
    device = resolve_device(options.device)
    print(
        f"Sound probe device: {device}"
        + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""),
        flush=True,
    )
    representations: dict[str, torch.Tensor] = {}
    checkpoint_meta: dict[str, dict[str, object]] = {}
    first_payload = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False)
    first_config, vocabulary = first_payload["config"], first_payload["vocabulary"]
    phase1_noise_sigma = float(first_config.get("noise_sigma", 0.05))
    if not math.isclose(options.noise_sigma, phase1_noise_sigma, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            f"Sound noise_sigma={options.noise_sigma} differs from Phase 1 noise_sigma={phase1_noise_sigma}; "
            "the representation probe must reuse the training input distribution"
        )
    phoneme_envelope = first_config.get("phoneme_envelope")
    # Reconstruct the original Phase-1 initialization, as in the segmentation
    # control. No trained state dictionary is loaded into M00.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(first_config["seed"]))
        baseline_lm = CausalPhonemeGRU(
            features.width, int(first_config["hidden_size"]), int(first_config["num_layers"]),
            len(vocabulary), float(first_config.get("dropout", 0.0)),
        ).to(device)
    baseline = FrozenGRUEncoder(baseline_lm, {str(k): int(v) for k, v in vocabulary.items()}, first_config)
    baseline_snapshot = baseline.parameter_snapshot()
    print("Encoding M00 (untrained Phase 1 initialization)...", flush=True)
    representations["M00"] = encode_sound_items(
        baseline, items, features, options.input_noise_seed, options, device,
        phoneme_envelope=phoneme_envelope,
    )
    checkpoint_meta["M00"] = {"checkpoint_file": "random-initialized", "checkpoint_hours": 0.0}
    baseline.assert_unchanged(baseline_snapshot)
    del baseline
    checkpoint_progress = tqdm(checkpoints, desc="Encoding Sound checkpoints", unit="checkpoint", dynamic_ncols=True)
    for checkpoint_id, checkpoint_path, hours in checkpoint_progress:
        checkpoint_progress.set_postfix(checkpoint=checkpoint_id, hours=f"{hours:.2f}")
        encoder = FrozenGRUEncoder.from_checkpoint(checkpoint_path, features.width, device, expected_hidden_dim=128)
        if encoder.vocabulary != {str(k): int(v) for k, v in vocabulary.items()}:
            raise ValueError(f"{checkpoint_id}: vocabulary differs from M00/checkpoint ordering")
        snapshot = encoder.parameter_snapshot()
        representations[checkpoint_id] = encode_sound_items(
            encoder, items, features, options.input_noise_seed, options, device,
            phoneme_envelope=phoneme_envelope,
        )
        encoder.assert_unchanged(snapshot)
        checkpoint_meta[checkpoint_id] = {"checkpoint_file": str(checkpoint_path), "checkpoint_hours": hours}
        del encoder

    labels = torch.tensor([item.label for item in items], dtype=torch.float32)
    partitions = {name: torch.tensor([i for i, item in enumerate(items) if item.split == name]) for name in PARTITIONS}
    # All split inputs use the same deterministic trial-indexed noise. Reinitialize one head per
    # checkpoint and paired seed; average its held-out logits over seeds for primary metrics.
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    checkpoint_ids = ["M00", *(identifier for identifier, _, _ in checkpoints)]
    per_seed_logits: dict[tuple[str, int], torch.Tensor] = {}
    (output / "heads").mkdir(parents=True, exist_ok=True)
    (output / "histories").mkdir(parents=True, exist_ok=True)
    head_progress = tqdm(checkpoint_ids, desc="Training Sound probes", unit="model", dynamic_ncols=True)
    for checkpoint_id in head_progress:
        head_progress.set_postfix(checkpoint=checkpoint_id)
        x = representations[checkpoint_id]
        for seed in options.initialization_seeds:
            train_ids = partitions["train"]
            validation_ids = partitions["validation"]
            test_ids = partitions["test"]
            head, history, selected = fit_sound_head(
                x[train_ids], labels[train_ids], x[validation_ids], labels[validation_ids],
                initialization_seed=seed, options=options, device=device,
            )
            # Test enters only here, after fitting and validation selection are
            # complete. This is the sole held-out forward evaluation per head.
            test_metrics, test_logits = evaluate_split(head, x[test_ids], labels[test_ids])
            per_seed_logits[(checkpoint_id, seed)] = test_logits
            torch.save({
                "state_dict": head.state_dict(), "checkpoint_id": checkpoint_id,
                "checkpoint_hours": checkpoint_meta[checkpoint_id]["checkpoint_hours"],
                "initialization_seed": seed, "hidden_size": 128,
                "selection_metric": "validation_cross_entropy",
                "best_epoch": selected["best_epoch"],
            }, output / "heads" / f"{checkpoint_id}_seed-{seed}_linear_128x1.pt")
            history_rows = [
                {"checkpoint_id": checkpoint_id, "initialization_seed": seed, **history_row}
                for history_row in history
            ]
            _write_tsv(
                output / "histories" / f"{checkpoint_id}_seed-{seed}.tsv",
                history_rows,
            )
            test_items = [items[index] for index in test_ids.tolist()]
            test_labels = labels[test_ids]
            subtype_metrics = {}
            for subtype in ("onset", "rhyme"):
                selected_rows = [i for i, item in enumerate(test_items) if item.condition in {subtype, "unrelated"}]
                idx = torch.tensor(selected_rows, dtype=torch.long)
                subtype_metrics[subtype] = _auc(test_labels[idx], test_logits[idx])
            rows.append({
                "checkpoint_id": checkpoint_id, "checkpoint_hours": checkpoint_meta[checkpoint_id]["checkpoint_hours"],
                "initialization_seed": seed, "best_epoch": selected["best_epoch"],
                "epochs_run": selected["epochs_run"], "test_auc": test_metrics["auc"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_cross_entropy": test_metrics["cross_entropy"],
                "test_onset_vs_unrelated_auc": subtype_metrics["onset"],
                "test_rhyme_vs_unrelated_auc": subtype_metrics["rhyme"],
            })
            predictions.extend({
                "checkpoint_id": checkpoint_id, "initialization_seed": seed,
                "item_id": item.item_id, "condition": item.condition, "label": item.label,
                "logit": float(logit), "probability_positive": float(torch.sigmoid(logit)),
            } for item, logit in zip(test_items, test_logits, strict=True))

    # Paired component bootstrap on averaged, identical test items.
    development: list[dict[str, object]] = []
    test_indices = partitions["test"].tolist()
    test_items = [items[index] for index in test_indices]
    test_labels = labels[partitions["test"]]
    baseline_avg = torch.stack([per_seed_logits[("M00", seed)] for seed in options.initialization_seeds]).mean(0)
    test_components = _components_per_item(test_items)
    for checkpoint_id in checkpoint_ids[1:]:
        avg = torch.stack([per_seed_logits[(checkpoint_id, seed)] for seed in options.initialization_seeds]).mean(0)
        point, low, high = paired_bootstrap_auc_delta(
            test_labels, avg, baseline_avg, test_components,
            options.bootstrap_repetitions, options.input_noise_seed,
        )
        development.append({"checkpoint_id": checkpoint_id,
                            "checkpoint_hours": checkpoint_meta[checkpoint_id]["checkpoint_hours"],
                            "delta_auc_vs_M00": point,
                            "ci95_lower": low, "ci95_upper": high})
    onset = None
    for index in range(len(development) - 2):
        triple = development[index:index + 3]
        if all(float(row["ci95_lower"]) > 0 for row in triple):
            onset = triple[0]["checkpoint_id"]
            break
    _write_tsv(output / "metrics_by_seed.tsv", rows)
    _write_tsv(output / "predictions_test.tsv", predictions)
    _write_tsv(output / "developmental_delta_auc.tsv", development)
    summary: list[dict[str, object]] = []
    metric_names = (
        "test_auc", "test_balanced_accuracy", "test_cross_entropy",
        "test_onset_vs_unrelated_auc", "test_rhyme_vs_unrelated_auc",
    )
    for checkpoint_id in checkpoint_ids:
        selected_rows = [row for row in rows if row["checkpoint_id"] == checkpoint_id]
        summary_row: dict[str, object] = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_hours": checkpoint_meta[checkpoint_id]["checkpoint_hours"],
            "initialization_seed_count": len(selected_rows),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected_rows])
            summary_row[f"mean_{metric}"] = float(values.mean())
            summary_row[f"sd_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(summary_row)
    _write_tsv(output / "checkpoint_summary.tsv", summary)
    onset_hours = next(
        (row["checkpoint_hours"] for row in development if row["checkpoint_id"] == onset), None,
    )
    (output / "run_manifest.json").write_text(json.dumps({
        "task": "Sound", "manifest": str(Path(stimuli_path).resolve()),
        "test_used_for_head_fitting_or_epoch_selection": False,
        "manifest_stimulus_selection_is_model_blind": True,
        "test_evaluations_per_head": 1,
        "initialization_seeds": list(options.initialization_seeds),
        "representation_onset_checkpoint": onset,
        "representation_onset_hours": onset_hours,
        "M00_initialization_seed": int(first_config["seed"]),
        "phase1_noise_sigma": phase1_noise_sigma,
        "phase1_phoneme_envelope": phoneme_envelope,
        "options": asdict(options),
        "lexical_components_test_n": len(test_components),
        "lexical_components_test_max_items": max(map(len, test_components)),
        "checkpoint_order": checkpoint_ids,
    }, indent=2) + "\n", encoding="utf-8")
    return {"metrics": rows, "development": development, "representation_onset": onset}
