from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import shutil
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from devlm.data import Session, Utterance
from devlm.features import FeatureTable
from devlm.model import CausalPhonemeGRU
from devlm.stream import build_session_stream
from devlm.train import chunk_slices, resolve_device

from .model import make_binary_readout
from .train import PARTITIONS, discover_checkpoints, discover_feature_table


@dataclass(frozen=True)
class SegmentationProbeOptions:
    split_seed: int = 20261003
    input_noise_seed: int = 20261004
    initialization_seed: int = 1729
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    max_train_transitions: int = 120_000
    max_validation_transitions: int = 30_000
    max_test_transitions: int = 30_000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 2048
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    bootstrap_repetitions: int = 1000
    sequence_chunk_frames: int = 4096
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("train_fraction and validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be below 1")
        limits = (self.max_train_transitions, self.max_validation_transitions, self.max_test_transitions)
        if any(value < 1 for value in limits):
            raise ValueError("All transition limits must be positive")
        if self.learning_rate <= 0 or self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("Probe optimization settings must be positive")
        if self.bootstrap_repetitions < 1 or self.sequence_chunk_frames < 1:
            raise ValueError("bootstrap_repetitions and sequence_chunk_frames must be positive")


@dataclass(frozen=True)
class BoundarySession:
    session: Session
    transition_labels: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TransitionExample:
    item_id: str
    split: str
    corpus_id: str
    session_id: str
    utterance_index: int
    utterance_order: int
    next_phoneme_index: int
    next_phoneme: str
    label: int


def parse_boundary_annotated_ipa(raw: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Remove WORD_BOUNDARY and label only within-utterance phoneme transitions.

    Label j-1 describes the transition from phoneme j-1 to phoneme j. A trailing
    marker creates no example, so utterance boundaries are never labelled.
    """
    phonemes: list[str] = []
    labels: list[int] = []
    boundary_since_phone = False
    for token in raw.split():
        if token == "WORD_BOUNDARY":
            if phonemes:
                boundary_since_phone = True
            continue
        if phonemes:
            labels.append(int(boundary_since_phone))
        phonemes.append(token)
        boundary_since_phone = False
    if len(labels) != max(0, len(phonemes) - 1):
        raise AssertionError("Boundary-label construction is inconsistent")
    return tuple(phonemes), tuple(labels)


def load_phase1_validation_sessions(
    source_csv: str | Path,
    session_split_path: str | Path,
) -> list[BoundarySession]:
    split = json.loads(Path(session_split_path).read_text(encoding="utf-8"))
    validation_keys = {(str(a), str(b)) for a, b in split["validation"]}
    grouped: dict[tuple[str, str], list[tuple[Utterance, tuple[int, ...]]]] = {}
    with Path(source_csv).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"corpus_id", "transcript_id", "id", "target_child_age", "ipa_transcription"}
        missing = required - set(rows.fieldnames or ())
        if missing:
            raise ValueError(f"IPA-CHILDES source is missing columns: {', '.join(sorted(missing))}")
        for row_index, row in enumerate(rows, 1):
            key = (str(row["corpus_id"]), str(row["transcript_id"]))
            if key not in validation_keys:
                continue
            try:
                age = float(row["target_child_age"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(age):
                continue
            phonemes, labels = parse_boundary_annotated_ipa(row["ipa_transcription"])
            if not phonemes:
                continue
            utterance = Utterance(
                corpus_id=key[0], session_id=key[1], target_child_age_months=age,
                utterance_order=int(row["id"]), ipa=" ".join(phonemes),
                text=str(row.get("processed_gloss") or row.get("gloss") or ""),
                phonemes=phonemes,
            )
            grouped.setdefault(key, []).append((utterance, labels))
            if row_index % 500_000 == 0:
                print(f"Scanned {row_index:,} IPA-CHILDES rows...", flush=True)
    missing_sessions = validation_keys - set(grouped)
    if missing_sessions:
        example = sorted(missing_sessions)[0]
        raise ValueError(
            f"Raw IPA-CHILDES source is missing {len(missing_sessions)} Phase 1 validation sessions; "
            f"first missing key={example}"
        )
    result: list[BoundarySession] = []
    for key, records in grouped.items():
        records.sort(key=lambda pair: pair[0].utterance_order)
        ages = {pair[0].target_child_age_months for pair in records}
        if len(ages) != 1:
            raise ValueError(f"Inconsistent target-child ages in {key[0]}/{key[1]}")
        result.append(BoundarySession(
            Session(key[0], key[1], ages.pop(), tuple(pair[0] for pair in records)),
            tuple(pair[1] for pair in records),
        ))
    return sorted(result, key=lambda value: (
        value.session.target_child_age_months, value.session.corpus_id, value.session.session_id,
    ))


def split_probe_sessions(
    sessions: list[BoundarySession], options: SegmentationProbeOptions,
) -> dict[str, list[BoundarySession]]:
    if len(sessions) < 3:
        raise ValueError("At least three Phase 1 validation sessions are required")
    shuffled = list(sessions)
    random.Random(options.split_seed).shuffle(shuffled)
    n_train = max(1, round(len(shuffled) * options.train_fraction))
    n_validation = max(1, round(len(shuffled) * options.validation_fraction))
    if n_train + n_validation >= len(shuffled):
        n_train = len(shuffled) - 2
        n_validation = 1
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_validation],
        "test": shuffled[n_train + n_validation:],
    }


def _all_transition_examples(session: BoundarySession, split: str) -> list[TransitionExample]:
    examples: list[TransitionExample] = []
    for utterance_index, (utterance, labels) in enumerate(zip(
        session.session.utterances, session.transition_labels, strict=True,
    )):
        phonemes = utterance.phonemes or ()
        if len(labels) != len(phonemes) - 1:
            raise ValueError(f"Transition labels do not match utterance {utterance.utterance_order}")
        for next_index, label in enumerate(labels, 1):
            item_id = (
                f"{session.session.corpus_id}:{session.session.session_id}:"
                f"{utterance.utterance_order}:{next_index}"
            )
            examples.append(TransitionExample(
                item_id, split, session.session.corpus_id, session.session.session_id,
                utterance_index, utterance.utterance_order, next_index,
                phonemes[next_index], int(label),
            ))
    return examples


def sample_transitions(
    partitions: dict[str, list[BoundarySession]], options: SegmentationProbeOptions,
) -> list[TransitionExample]:
    limits = {
        "train": options.max_train_transitions,
        "validation": options.max_validation_transitions,
        "test": options.max_test_transitions,
    }
    selected: list[TransitionExample] = []
    for split_index, split in enumerate(PARTITIONS):
        # Reservoir sampling is uniform over transitions and preserves the
        # natural boundary prevalence rather than balancing the labels.
        rng = random.Random(options.split_seed + 10_000 + split_index)
        reservoir: list[TransitionExample] = []
        seen = 0
        for session in partitions[split]:
            for example in _all_transition_examples(session, split):
                seen += 1
                if len(reservoir) < limits[split]:
                    reservoir.append(example)
                else:
                    replacement = rng.randrange(seen)
                    if replacement < len(reservoir):
                        reservoir[replacement] = example
        if not reservoir or len({item.label for item in reservoir}) != 2:
            raise ValueError(f"{split} transition sample must contain both boundary labels")
        selected.extend(sorted(reservoir, key=lambda item: item.item_id))
    return selected


def _session_noise_seed(base_seed: int, corpus_id: str, session_id: str) -> int:
    value = f"{base_seed}\0{corpus_id}\0{session_id}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


@torch.no_grad()
def extract_transition_states(
    model: CausalPhonemeGRU,
    sessions: list[BoundarySession],
    examples: list[TransitionExample],
    features: FeatureTable,
    vocabulary: dict[str, int],
    noise_sigma: float,
    phoneme_envelope: list[float] | None,
    input_noise_seed: int,
    sequence_chunk_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract h at B.start_frame-1 for each selected adjacent A->B transition."""
    positions = {example.item_id: index for index, example in enumerate(examples)}
    selected_by_session: dict[tuple[str, str], list[TransitionExample]] = {}
    for example in examples:
        selected_by_session.setdefault((example.corpus_id, example.session_id), []).append(example)
    states = torch.empty((len(examples), model.gru.hidden_size), dtype=torch.float32)
    filled = torch.zeros(len(examples), dtype=torch.bool)
    model.eval()
    for boundary_session in sessions:
        session = boundary_session.session
        key = (session.corpus_id, session.session_id)
        wanted = selected_by_session.get(key)
        if not wanted:
            continue
        rng = np.random.default_rng(_session_noise_seed(input_noise_seed, *key))
        stream = build_session_stream(
            session, features, vocabulary, rng,
            noise_sigma=noise_sigma, phoneme_envelope=phoneme_envelope,
        )
        spans_by_utterance: dict[int, list] = {}
        for span in stream.spans:
            spans_by_utterance.setdefault(span.utterance_index, []).append(span)
        frame_to_output: dict[int, list[int]] = {}
        for example in wanted:
            span = spans_by_utterance[example.utterance_index][example.next_phoneme_index]
            predictor_frame = span.start_frame - 1
            if predictor_frame < 0 or predictor_frame >= span.start_frame:
                raise AssertionError("Target phoneme activation is visible to the segmentation probe")
            frame_to_output.setdefault(predictor_frame, []).append(positions[example.item_id])
        hidden = None
        for start, end in chunk_slices(len(stream.noisy_frames), sequence_chunk_frames):
            frames = torch.from_numpy(stream.noisy_frames[start:end]).unsqueeze(0).to(device).contiguous()
            chunk_states, hidden = model.gru(frames, hidden)
            hidden = hidden.detach()
            for frame in (value for value in frame_to_output if start <= value < end):
                value = chunk_states[0, frame - start].detach().cpu().float()
                for output_index in frame_to_output[frame]:
                    states[output_index] = value
                    filled[output_index] = True
    if not bool(filled.all()):
        raise AssertionError(f"Failed to extract {int((~filled).sum())} selected transition states")
    return states


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predicted = logits >= 0
    truth = labels.astype(bool)
    tp = int((predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "auc": _auc(labels, logits),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "boundary_prevalence": float(labels.mean()),
        "accuracy": float((predicted == truth).mean()),
    }


def train_segmentation_head(
    states: torch.Tensor,
    labels: torch.Tensor,
    split_indices: dict[str, torch.Tensor],
    options: SegmentationProbeOptions,
    device: torch.device,
) -> tuple[nn.Linear, dict[str, dict[str, float]], np.ndarray]:
    train_indices = split_indices["train"]
    center = states[train_indices].mean(0)
    scale = states[train_indices].std(0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    standardized = ((states - center) / scale).to(device)
    y = labels.float().to(device)
    torch.manual_seed(options.initialization_seed)
    head = make_binary_readout(states.shape[1], options.initialization_seed).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    train_y = labels[train_indices]
    positives = int(train_y.sum())
    negatives = len(train_y) - positives
    if not positives or not negatives:
        raise ValueError("Probe training split must contain both labels")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    best_loss, best_state, stale = float("inf"), deepcopy(head.state_dict()), 0
    for epoch in range(1, options.max_epochs + 1):
        generator = torch.Generator(device="cpu").manual_seed(options.initialization_seed + epoch)
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        head.train()
        for start in range(0, len(permutation), options.batch_size):
            batch = permutation[start:start + options.batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(standardized[batch]).squeeze(-1), y[batch])
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            val = split_indices["validation"].to(device)
            validation_loss = float(loss_fn(head(standardized[val]).squeeze(-1), y[val]))
        if validation_loss < best_loss - options.min_delta:
            best_loss, best_state, stale = validation_loss, deepcopy(head.state_dict()), 0
        else:
            stale += 1
            if stale >= options.patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        all_logits = head(standardized).squeeze(-1).cpu().numpy()
    measured = {
        split: _metrics(labels[index].numpy(), all_logits[index.numpy()])
        for split, index in split_indices.items()
    }
    # Store an equivalent Linear(hidden_dim,1) operating on raw states.
    head = head.cpu()
    with torch.no_grad():
        weight = head.weight.detach().clone()
        head.weight.copy_(weight / scale.unsqueeze(0))
        head.bias.copy_(head.bias - (weight * center.unsqueeze(0) / scale.unsqueeze(0)).sum(1))
    return head, measured, all_logits


def session_bootstrap_ci(
    labels: np.ndarray,
    logits: np.ndarray,
    session_ids: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    unique = np.unique(session_ids)
    if len(unique) < 2:
        raise ValueError("Session bootstrap requires at least two test sessions")
    values = {name: [] for name in ("auc", "balanced_accuracy", "boundary_prevalence", "accuracy")}
    rng = np.random.default_rng(seed)
    by_session = {session: np.flatnonzero(session_ids == session) for session in unique}
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([by_session[session] for session in sampled])
        if len(np.unique(labels[indices])) < 2:
            continue
        result = _metrics(labels[indices], logits[indices])
        for name in values:
            values[name].append(result[name])
    if not values["auc"]:
        raise ValueError("Every session bootstrap replicate had only one label")
    return {
        name: tuple(float(x) for x in np.quantile(result, [0.025, 0.975]))
        for name, result in values.items()
    }


def determine_segmentation_onset(rows: list[dict[str, object]]) -> dict[str, object]:
    m00 = next(row for row in rows if row["checkpoint_id"] == "M00")
    trained = [row for row in rows if row["checkpoint_id"] != "M00"]
    for index in range(max(0, len(trained) - 2)):
        window = trained[index:index + 3]
        if all(
            float(row["test_auc_ci_low"]) > 0.5
            and float(row["test_auc"]) > float(m00["test_auc"])
            for row in window
        ):
            return {
                "detected": True,
                "checkpoint_id": window[0]["checkpoint_id"],
                "checkpoint_hours": window[0]["checkpoint_hours"],
                "rule": "earliest of three consecutive checkpoints with test AUC CI lower > 0.5 and point AUC > M00",
            }
    return {
        "detected": False, "checkpoint_id": None, "checkpoint_hours": None,
        "rule": "earliest of three consecutive checkpoints with test AUC CI lower > 0.5 and point AUC > M00",
    }


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _make_m00(first_payload: dict, feature_width: int, vocabulary_size: int, device: torch.device) -> CausalPhonemeGRU:
    config = first_payload["config"]
    # Reconstruct the Phase 1 model's initial random weights from its original
    # seed. This is a true untrained architecture-matched baseline.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config["seed"]))
        model = CausalPhonemeGRU(
            feature_width, int(config["hidden_size"]), int(config["num_layers"]),
            vocabulary_size, float(config.get("dropout", 0.0)),
        )
    return model.to(device).requires_grad_(False).eval()


def run_segmentation_probe(
    source_csv: str | Path,
    checkpoint_root: str | Path,
    output_root: str | Path,
    session_split_path: str | Path | None = None,
    feature_table_path: str | Path | None = None,
    options: SegmentationProbeOptions = SegmentationProbeOptions(),
) -> Path:
    started = time.perf_counter()
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output_root}")
    for name in ("summary", "predictions", "heads", "configs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(checkpoint_root)
    feature_path = Path(feature_table_path) if feature_table_path else discover_feature_table(checkpoint_root)
    split_path = Path(session_split_path) if session_split_path else next(iter(sorted(Path(checkpoint_root).rglob("session_split.json"))), None)
    if split_path is None:
        raise ValueError("Checkpoint directory does not contain session_split.json")
    features = FeatureTable.from_json(feature_path)
    first_payload = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in first_payload["vocabulary"].items()}
    hidden_size = int(first_payload["config"]["hidden_size"])
    if hidden_size != 128:
        raise ValueError(f"Minimal probe requires the Phase 1 hidden size 128; found {hidden_size}")
    device = resolve_device(options.device)
    print(f"Segmentation probe device: {device}", flush=True)
    sessions = load_phase1_validation_sessions(source_csv, split_path)
    partitions = split_probe_sessions(sessions, options)
    examples = sample_transitions(partitions, options)
    split_indices = {
        split: torch.tensor([index for index, item in enumerate(examples) if item.split == split], dtype=torch.long)
        for split in PARTITIONS
    }
    labels = torch.tensor([item.label for item in examples], dtype=torch.float32)
    session_lookup = {
        split: np.asarray([f"{examples[index].corpus_id}:{examples[index].session_id}" for index in split_indices[split]])
        for split in PARTITIONS
    }
    dataset_rows = [asdict(item) for item in examples]
    _write_table(output_root / "configs" / "transition_manifest.tsv", dataset_rows)
    (output_root / "configs" / "run_config.json").write_text(json.dumps({
        **asdict(options),
        "source_csv": str(Path(source_csv).resolve()),
        "checkpoint_root": str(Path(checkpoint_root).resolve()),
        "feature_table": str(feature_path.resolve()),
        "phase1_session_split": str(split_path.resolve()),
        "session_counts": {key: len(value) for key, value in partitions.items()},
        "transition_counts": {key: len(split_indices[key]) for key in PARTITIONS},
        "label_definition": "1 iff WORD_BOUNDARY occurs between adjacent phonemes within one utterance",
        "readout_frame": "next phoneme start_frame - 1",
        "utterance_boundaries_excluded": True,
        "word_boundary_marker_input_to_model": False,
        "readout": "Linear(128,1)",
        "loss": "class-weighted BCEWithLogitsLoss using probe-train prevalence only",
    }, indent=2) + "\n", encoding="utf-8")
    model_specs: list[tuple[str, Path | None, float]] = [("M00", None, 0.0), *checkpoints]
    all_sessions = [item for split in PARTITIONS for item in partitions[split]]
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for model_index, (checkpoint_id, checkpoint_path, checkpoint_hours) in enumerate(tqdm(
        model_specs, desc="Segmentation checkpoints", unit="model", dynamic_ncols=True,
    )):
        if checkpoint_path is None:
            model = _make_m00(first_payload, features.width, len(vocabulary), device)
        else:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if {str(key): int(value) for key, value in payload["vocabulary"].items()} != vocabulary:
                raise ValueError(f"{checkpoint_id}: phoneme vocabulary differs from M01")
            config = payload["config"]
            model = CausalPhonemeGRU(
                features.width, int(config["hidden_size"]), int(config["num_layers"]),
                len(vocabulary), float(config.get("dropout", 0.0)),
            )
            model.load_state_dict(payload["model"])
            model = model.to(device).requires_grad_(False).eval()
        config = first_payload["config"]
        states = extract_transition_states(
            model, all_sessions, examples, features, vocabulary,
            float(config.get("noise_sigma", 0.05)), config.get("phoneme_envelope"),
            options.input_noise_seed, options.sequence_chunk_frames, device,
        )
        head, measured, logits = train_segmentation_head(states, labels, split_indices, options, device)
        test_index = split_indices["test"].numpy()
        intervals = session_bootstrap_ci(
            labels.numpy()[test_index], logits[test_index], session_lookup["test"],
            options.bootstrap_repetitions, options.split_seed + 50_000 + model_index,
        )
        row: dict[str, object] = {"checkpoint_id": checkpoint_id, "checkpoint_hours": checkpoint_hours}
        for split in PARTITIONS:
            for metric, value in measured[split].items():
                row[f"{split}_{metric}"] = value
        for metric, (low, high) in intervals.items():
            row[f"test_{metric}_ci_low"] = low
            row[f"test_{metric}_ci_high"] = high
        summary_rows.append(row)
        _write_table(output_root / "summary" / "checkpoint_metrics.tsv", summary_rows)
        torch.save({
            "state_dict": head.state_dict(), "checkpoint_id": checkpoint_id,
            "checkpoint_hours": checkpoint_hours, "hidden_size": hidden_size,
            "initialization_seed": options.initialization_seed,
        }, output_root / "heads" / f"{checkpoint_id}_linear_128x1.pt")
        for index in test_index:
            item = examples[index]
            prediction_rows.append({
                "checkpoint_id": checkpoint_id, "checkpoint_hours": checkpoint_hours,
                "item_id": item.item_id, "corpus_id": item.corpus_id,
                "session_id": item.session_id, "label": item.label,
                "logit": float(logits[index]), "predicted_label": int(logits[index] >= 0),
            })
        _write_table(output_root / "predictions" / "test_predictions.tsv", prediction_rows)
        tqdm.write(
            f"{checkpoint_id} hours={checkpoint_hours:.3f} "
            f"test_auc={row['test_auc']:.4f} "
            f"CI=[{row['test_auc_ci_low']:.4f}, {row['test_auc_ci_high']:.4f}] "
            f"test_bal_acc={row['test_balanced_accuracy']:.4f}"
        )
        del model, states
        if device.type == "cuda":
            torch.cuda.empty_cache()
    onset = determine_segmentation_onset(summary_rows)
    (output_root / "summary" / "segmentation_onset.json").write_text(
        json.dumps(onset, indent=2) + "\n", encoding="utf-8",
    )
    (output_root / "run_manifest.json").write_text(json.dumps({
        "status": "complete", "models": 31, "elapsed_seconds": time.perf_counter() - started,
        "scientific_interpretation": "linearly decodable within-utterance word-boundary information; not autonomous segmentation",
        "excluded": ["RT", "SOA", "target trajectory", "n-gram controls", "nonlinear probes", "semantic tasks", "lexical-disjoint splits"],
        "segmentation_onset": onset,
    }, indent=2) + "\n", encoding="utf-8")
    archive = Path(shutil.make_archive(str(output_root.parent / "segmentation_probe_results"), "zip", root_dir=output_root))
    print(f"Segmentation probe results: {archive}", flush=True)
    return archive
