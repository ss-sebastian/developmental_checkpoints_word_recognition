"""Linearly decode the currently active phoneme from frozen Phase-1 states.

This is deliberately a *sanity-check* probe.  A high score at the peak input
frame can be obtained by preserving the input feature vector, including in an
untrained recurrent net; it is not evidence for abstract phonological
categories.  The module keeps test labels out of fit/model-selection functions
and exposes them only to ``evaluate_multinomial_probe``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from devlm.features import FeatureTable
from devlm.model import CausalPhonemeGRU
from devlm.stream import build_session_stream
from devlm.train import chunk_slices, resolve_device

from .segmentation_probe import BoundarySession, _make_m00, load_phase1_validation_sessions, split_probe_sessions
from .train import PARTITIONS, discover_checkpoints, discover_feature_table


@dataclass(frozen=True)
class PhonemeDecodingProbeOptions:
    split_seed: int = 20261005
    input_noise_seed: int = 20261004
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    min_tokens_per_phoneme: int = 20
    max_tokens_per_phoneme: int = 500
    l2_values: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    lbfgs_max_iter: int = 250
    pca_points_per_phoneme: int = 150
    max_pca_classes: int = 20
    sequence_chunk_frames: int = 4096
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("train_fraction and validation_fraction must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train_fraction + validation_fraction must be below 1")
        if self.min_tokens_per_phoneme < 1 or self.max_tokens_per_phoneme < self.min_tokens_per_phoneme:
            raise ValueError("Token limits must be positive and max >= min")
        if not self.l2_values or any(value <= 0 for value in self.l2_values):
            raise ValueError("l2_values must be non-empty and strictly positive")
        if self.lbfgs_max_iter < 1 or self.pca_points_per_phoneme < 1 or self.max_pca_classes < 2 or self.sequence_chunk_frames < 1:
            raise ValueError("Iteration, PCA, and chunk settings must be positive")


@dataclass(frozen=True)
class PhonemeEvent:
    item_id: str
    split: str
    corpus_id: str
    session_id: str
    utterance_index: int
    utterance_order: int
    phoneme_index: int
    phoneme: str


@dataclass(frozen=True)
class Standardizer:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, states: np.ndarray) -> np.ndarray:
        return (states - self.center) / self.scale


@dataclass(frozen=True)
class MultinomialLinearProbe:
    classes: tuple[str, ...]
    weight: np.ndarray
    bias: np.ndarray
    standardizer: Standardizer
    l2: float

    def logits(self, states: np.ndarray) -> np.ndarray:
        return self.standardizer.transform(states) @ self.weight.T + self.bias

    def predict(self, states: np.ndarray) -> np.ndarray:
        return self.logits(states).argmax(axis=1)


@dataclass(frozen=True)
class PCAProjection:
    center: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, states: np.ndarray) -> np.ndarray:
        return (states - self.center) @ self.components.T


def _session_noise_seed(base_seed: int, corpus_id: str, session_id: str) -> int:
    raw = f"{base_seed}\0{corpus_id}\0{session_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def all_phoneme_events(partitions: dict[str, list[BoundarySession]]) -> list[PhonemeEvent]:
    """Create one event per actual phoneme, never per frame or zero pause."""
    events: list[PhonemeEvent] = []
    for split in PARTITIONS:
        for boundary_session in partitions[split]:
            session = boundary_session.session
            for utterance_index, utterance in enumerate(session.utterances):
                for phoneme_index, phoneme in enumerate(utterance.phonemes or ()):
                    events.append(PhonemeEvent(
                        f"{session.corpus_id}:{session.session_id}:{utterance.utterance_order}:{phoneme_index}",
                        split, session.corpus_id, session.session_id, utterance_index,
                        utterance.utterance_order, phoneme_index, phoneme,
                    ))
    return events


def balance_phoneme_events(events: list[PhonemeEvent], options: PhonemeDecodingProbeOptions) -> list[PhonemeEvent]:
    """Use classes attested in every split and balance each split exactly.

    The deterministic class-wise draw prevents high-frequency phones from
    defining macro metrics, while session allocation has already happened
    before any phoneme labels are considered.
    """
    grouped: dict[tuple[str, str], list[PhonemeEvent]] = {}
    for event in events:
        grouped.setdefault((event.split, event.phoneme), []).append(event)
    phonemes = sorted({event.phoneme for event in events})
    eligible = [phone for phone in phonemes if all(
        len(grouped.get((split, phone), ())) >= options.min_tokens_per_phoneme for split in PARTITIONS
    )]
    if len(eligible) < 2:
        raise ValueError("Fewer than two phonemes meet the per-split token minimum")
    selected: list[PhonemeEvent] = []
    for split_index, split in enumerate(PARTITIONS):
        n_per_class = min(options.max_tokens_per_phoneme, *(
            len(grouped[(split, phone)]) for phone in eligible
        ))
        if n_per_class < options.min_tokens_per_phoneme:
            raise AssertionError("Eligibility and balancing disagree")
        for phone_index, phone in enumerate(eligible):
            values = sorted(grouped[(split, phone)], key=lambda event: event.item_id)
            rng = random.Random(options.split_seed + 10_000 * split_index + phone_index)
            if len(values) > n_per_class:
                values = rng.sample(values, n_per_class)
            selected.extend(sorted(values, key=lambda event: event.item_id))
    return selected


@torch.no_grad()
def extract_phoneme_peak_states(
    model: CausalPhonemeGRU, sessions: list[BoundarySession], events: list[PhonemeEvent],
    features: FeatureTable, vocabulary: dict[str, int], noise_sigma: float,
    phoneme_envelope: list[float] | None, input_noise_seed: int,
    sequence_chunk_frames: int, device: torch.device,
) -> torch.Tensor:
    """Extract ``h[start_frame + 2]``: the unique 1.0-envelope peak frame."""
    output_index = {event.item_id: index for index, event in enumerate(events)}
    wanted: dict[tuple[str, str], list[PhonemeEvent]] = {}
    for event in events:
        wanted.setdefault((event.corpus_id, event.session_id), []).append(event)
    states = torch.empty((len(events), model.gru.hidden_size), dtype=torch.float32)
    filled = torch.zeros(len(events), dtype=torch.bool)
    model.eval()
    for item in tqdm(sessions, desc="Extract peak frames", leave=False, unit="session"):
        session = item.session
        session_events = wanted.get((session.corpus_id, session.session_id), ())
        if not session_events:
            continue
        stream = build_session_stream(
            session, features, vocabulary,
            np.random.default_rng(_session_noise_seed(input_noise_seed, session.corpus_id, session.session_id)),
            noise_sigma=noise_sigma, phoneme_envelope=phoneme_envelope,
        )
        by_utterance: dict[int, list] = {}
        for span in stream.spans:
            by_utterance.setdefault(span.utterance_index, []).append(span)
        by_frame: dict[int, list[int]] = {}
        for event in session_events:
            span = by_utterance[event.utterance_index][event.phoneme_index]
            peak = span.start_frame + 2
            if peak >= span.end_frame:
                raise AssertionError("Phoneme peak must be inside its five-frame span")
            by_frame.setdefault(peak, []).append(output_index[event.item_id])
        hidden = None
        for start, end in chunk_slices(len(stream.noisy_frames), sequence_chunk_frames):
            frame_tensor = torch.from_numpy(stream.noisy_frames[start:end]).unsqueeze(0).to(device).contiguous()
            values, hidden = model.gru(frame_tensor, hidden)
            hidden = hidden.detach()
            for frame in (x for x in by_frame if start <= x < end):
                state = values[0, frame - start].detach().cpu().float()
                for index in by_frame[frame]:
                    states[index] = state
                    filled[index] = True
    if not bool(filled.all()):
        raise AssertionError(f"Failed to extract {int((~filled).sum())} phoneme peak states")
    return states


def extract_phoneme_peak_inputs(
    sessions: list[BoundarySession], events: list[PhonemeEvent], features: FeatureTable,
    vocabulary: dict[str, int], noise_sigma: float, phoneme_envelope: list[float] | None,
    input_noise_seed: int,
) -> np.ndarray:
    """Raw noisy input vector at the identical peak frame (an explicit leakage control)."""
    indices = {event.item_id: i for i, event in enumerate(events)}
    wanted: dict[tuple[str, str], list[PhonemeEvent]] = {}
    for event in events: wanted.setdefault((event.corpus_id, event.session_id), []).append(event)
    values = np.empty((len(events), features.width), dtype=np.float32)
    filled = np.zeros(len(events), dtype=bool)
    for item in sessions:
        session = item.session; selected = wanted.get((session.corpus_id, session.session_id), ())
        if not selected: continue
        stream = build_session_stream(session, features, vocabulary, np.random.default_rng(_session_noise_seed(input_noise_seed, session.corpus_id, session.session_id)), noise_sigma=noise_sigma, phoneme_envelope=phoneme_envelope)
        spans: dict[int, list] = {}
        for span in stream.spans: spans.setdefault(span.utterance_index, []).append(span)
        for event in selected:
            values[indices[event.item_id]] = stream.noisy_frames[spans[event.utterance_index][event.phoneme_index].start_frame + 2]
            filled[indices[event.item_id]] = True
    if not filled.all(): raise AssertionError("Failed to extract raw peak inputs")
    return values


def fit_pca(train_states: np.ndarray, n_components: int = 2) -> PCAProjection:
    if train_states.ndim != 2 or len(train_states) < 2:
        raise ValueError("PCA needs at least two 2-D training observations")
    center = train_states.mean(0)
    _, singular, vt = np.linalg.svd(train_states - center, full_matrices=False)
    n = min(n_components, vt.shape[0])
    variance = singular ** 2
    return PCAProjection(center, vt[:n], variance[:n] / variance.sum())


def _cross_entropy(labels: np.ndarray, logits: np.ndarray) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return float((-shifted[np.arange(len(labels)), labels] + np.log(np.exp(shifted).sum(axis=1))).mean())


def fit_multinomial_probe(
    train_states: np.ndarray, train_labels: np.ndarray, validation_states: np.ndarray,
    validation_labels: np.ndarray, classes: tuple[str, ...], options: PhonemeDecodingProbeOptions,
) -> MultinomialLinearProbe:
    """Fit/select a deterministic L2 multinomial logistic regression using no test data."""
    if set(train_labels) != set(range(len(classes))):
        raise ValueError("Every class must occur in probe-train data")
    center = train_states.mean(0)
    scale = train_states.std(0)
    scale[scale < 1e-6] = 1.0
    standardizer = Standardizer(center, scale)
    x_train = torch.as_tensor(standardizer.transform(train_states), dtype=torch.float64)
    y_train = torch.as_tensor(train_labels, dtype=torch.long)
    best: MultinomialLinearProbe | None = None
    best_loss = float("inf")
    for l2 in options.l2_values:
        # Zero initialization is deterministic. Cross entropy plus L2 is convex.
        weight = torch.zeros((len(classes), x_train.shape[1]), dtype=torch.float64, requires_grad=True)
        bias = torch.zeros(len(classes), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=options.lbfgs_max_iter,
                                      line_search_fn="strong_wolfe", tolerance_grad=1e-10,
                                      tolerance_change=1e-12)
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(x_train @ weight.T + bias, y_train)
            loss = loss + float(l2) * weight.square().sum() / 2
            loss.backward()
            return loss
        optimizer.step(closure)
        candidate = MultinomialLinearProbe(classes, weight.detach().numpy(), bias.detach().numpy(), standardizer, l2)
        loss = _cross_entropy(validation_labels, candidate.logits(validation_states))
        if loss < best_loss - 1e-12:
            best, best_loss = candidate, loss
    assert best is not None
    return best


def evaluate_multinomial_probe(probe: MultinomialLinearProbe, states: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """The sole evaluation API that accepts held-out labels."""
    return evaluate_multinomial_logits(probe.logits(states), labels, len(probe.classes))


def evaluate_multinomial_logits(logits: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    """Evaluate precomputed logits, allowing one test forward pass per model."""
    prediction = logits.argmax(axis=1)
    recalls = []
    for class_index in range(n_classes):
        mask = labels == class_index
        recalls.append(float((prediction[mask] == class_index).mean()))
    accuracy = float((prediction == labels).mean())
    # In the balanced design macro recall equals balanced accuracy; macro F1 is still useful for errors.
    f1s = []
    for class_index in range(n_classes):
        tp = int(((prediction == class_index) & (labels == class_index)).sum())
        fp = int(((prediction == class_index) & (labels != class_index)).sum())
        fn = int(((prediction != class_index) & (labels == class_index)).sum())
        f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    top3 = np.argsort(logits, axis=1)[:, -min(3, logits.shape[1]):]
    return {"accuracy": accuracy, "balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1s)),
            "top3_accuracy": float((top3 == labels[:, None]).any(axis=1).mean()), "cross_entropy": _cross_entropy(labels, logits)}


def fit_pooled_pca(train_by_checkpoint: list[np.ndarray], n_components: int = 2) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Exact equal-checkpoint pooled within-checkpoint covariance PCA.

    This avoids materializing a 31x concatenated matrix and avoids treating
    arbitrary between-checkpoint GRU coordinate offsets as phoneme variance.
    """
    if not train_by_checkpoint: raise ValueError("No checkpoint train states for PCA")
    centers = [states.mean(0) for states in train_by_checkpoint]
    covariance = sum((states - center).T @ (states - center) for states, center in zip(train_by_checkpoint, centers, strict=True))
    covariance /= sum(len(states) for states in train_by_checkpoint)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][:n_components]
    selected = np.maximum(values[order], 0)
    return centers, vectors[:, order].T, selected / max(float(np.maximum(values, 0).sum()), 1e-12)


def fit_pooled_pca_from_scatters(
    centers: list[np.ndarray], scatters: list[np.ndarray], counts: list[int], n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Pooled within-checkpoint PCA from online sufficient statistics only."""
    if not centers or not (len(centers) == len(scatters) == len(counts)) or min(counts) < 2:
        raise ValueError("PCA needs matching checkpoint centers, scatters, and at least two states each")
    covariance = sum(scatters) / sum(counts)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1][:n_components]
    selected = np.maximum(values[order], 0)
    return vectors[:, order].T, selected / max(float(np.maximum(values, 0).sum()), 1e-12)


def evaluate_majority_baseline(train_labels: np.ndarray, labels: np.ndarray, n_classes: int) -> dict[str, float]:
    """Training-prevalence majority baseline; no validation/test prevalence is read."""
    majority = int(np.bincount(train_labels, minlength=n_classes).argmax())
    prediction = np.full(len(labels), majority)
    recalls = [(prediction[labels == k] == k).mean() for k in range(n_classes)]
    f1 = [0.0 if k != majority else 2 * int((labels == k).sum()) / (len(labels) + int((labels == k).sum())) for k in range(n_classes)]
    return {"accuracy": float((prediction == labels).mean()), "balanced_accuracy": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1)), "top3_accuracy": float(min(3, n_classes) / n_classes), "cross_entropy": float("nan")}


def _write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def save_pca_figure(path: Path, coordinates: np.ndarray, labels: np.ndarray, classes: tuple[str, ...], checkpoint_id: str, evr: np.ndarray | None = None) -> None:
    """Save the requested colored-dot map; points are events, diamonds are centroids."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    colors = plt.get_cmap("tab20", len(classes))
    radius = max(float(np.abs(coordinates).max(initial=1.0)), 1.0) * 1.08
    axis.add_patch(plt.Circle((0, 0), radius, fill=False, color="0.72", linewidth=1.0, zorder=0))
    for index, phone in enumerate(classes):
        points = coordinates[labels == index]
        axis.scatter(points[:, 0], points[:, 1], s=11, alpha=0.35, color=colors(index), label=phone, linewidths=0)
        centroid = points.mean(0)
        axis.scatter(*centroid, s=65, marker="D", color=colors(index), edgecolors="black", linewidths=.5, zorder=3)
        axis.annotate(phone, centroid, xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
        if len(points) >= 3:
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
            if np.all(eigenvalues >= 0):
                order = np.argsort(eigenvalues)[::-1]
                eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                from matplotlib.patches import Ellipse
                direction = eigenvectors[:, 0]
                angle = np.degrees(np.arctan2(direction[1], direction[0]))
                # sqrt(chi-square_2,.95) = 2.448; descriptive class spread only.
                axis.add_patch(Ellipse(centroid, 2 * 2.448 * np.sqrt(eigenvalues[0]), 2 * 2.448 * np.sqrt(eigenvalues[1]), angle=angle, fill=False, color=colors(index), alpha=.45, linewidth=.8))
    suffix = "" if evr is None else f" ({evr[0] * 100:.1f}%)"
    suffix2 = "" if evr is None else f" ({evr[1] * 100:.1f}%)"
    axis.set(xlabel=f"shared PC1{suffix}", ylabel=f"shared PC2{suffix2}", title=f"{checkpoint_id}: phoneme peak states")
    axis.set_aspect("equal", adjustable="box")
    for suffix, kwargs in ((".png", {"dpi": 600}), (".tiff", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        figure.savefig(path.with_suffix(suffix), **kwargs)
    plt.close(figure)


def save_trajectory_figure(path: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(7.20, 3.2), constrained_layout=True)  # 183 mm double-column width
    axis.plot(range(len(rows)), [float(row["test_balanced_accuracy"]) for row in rows], marker="o", label="GRU hidden state")
    axis.plot(range(len(rows)), [float(row["test_raw_input_balanced_accuracy"]) for row in rows], linestyle="--", label="Raw peak input")
    axis.axhline(float(rows[0]["test_majority_balanced_accuracy"]), color="0.25", linestyle=":", label="Train-majority baseline")
    axis.set(xticks=range(len(rows)), xticklabels=[str(row["checkpoint_id"]) for row in rows], ylim=(0, 1), xlabel="Frozen model", ylabel="Test balanced accuracy")
    axis.tick_params(axis="x", rotation=90); axis.legend()
    for suffix, kwargs in ((".png", {"dpi": 600}), (".tiff", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        figure.savefig(path.with_suffix(suffix), **kwargs)
    plt.close(figure)


def save_six_panel_pca(path: Path, panels: dict[str, tuple[np.ndarray, np.ndarray]], classes: tuple[str, ...], evr: np.ndarray | None = None) -> None:
    """A fixed-basis, fixed-limit PCA comparison for six prespecified checkpoints."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})
    chosen = ("M00", "M01", "M05", "M10", "M20", "M30")
    available = [(name, *panels[name]) for name in chosen if name in panels]
    if not available: return
    all_coordinates = np.concatenate([x[1] for x in available]); bound = max(float(np.abs(all_coordinates).max()), 1.) * 1.06
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey=True, constrained_layout=True)
    colors = plt.get_cmap("tab20", len(classes))
    for axis, (name, coordinates, labels) in zip(axes.flat, available, strict=False):
        axis.add_patch(plt.Circle((0, 0), bound, fill=False, color="0.78", linewidth=.8))
        for index, phone in enumerate(classes):
            points = coordinates[labels == index]; centroid = points.mean(0)
            axis.scatter(points[:, 0], points[:, 1], s=6, alpha=.32, color=colors(index), linewidths=0)
            axis.scatter(*centroid, s=34, marker="D", color=colors(index), edgecolors="black", linewidths=.35)
            axis.annotate(phone, centroid, xytext=(2, 2), textcoords="offset points", fontsize=6)
            if len(points) >= 3:
                values, vectors = np.linalg.eigh(np.cov(points.T)); order = np.argsort(values)[::-1]
                values, vectors = np.maximum(values[order], 0), vectors[:, order]
                direction = vectors[:, 0]
                angle = np.degrees(np.arctan2(direction[1], direction[0]))
                axis.add_patch(Ellipse(centroid, 2 * 2.448 * np.sqrt(values[0]), 2 * 2.448 * np.sqrt(values[1]),
                                       angle=angle, fill=False, color=colors(index), alpha=.40, linewidth=.55))
        axis.set(title=name, xlim=(-bound, bound), ylim=(-bound, bound), aspect="equal")
    for axis in axes.flat[len(available):]: axis.set_visible(False)
    if evr is not None:
        figure.supxlabel(f"shared PC1 ({evr[0] * 100:.1f}%)")
        figure.supylabel(f"shared PC2 ({evr[1] * 100:.1f}%)")
    for suffix, kwargs in ((".png", {"dpi": 600}), (".tiff", {"dpi": 600}), (".svg", {}), (".pdf", {})):
        figure.savefig(path.with_suffix(suffix), **kwargs)
    plt.close(figure)


def run_phoneme_decoding_probe(
    source_csv: str | Path, checkpoint_root: str | Path, output_root: str | Path,
    session_split_path: str | Path | None = None, feature_table_path: str | Path | None = None,
    options: PhonemeDecodingProbeOptions = PhonemeDecodingProbeOptions(),
) -> Path:
    started, output_root = time.perf_counter(), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    for name in ("summary", "configs", "figures", "predictions", "heads"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    checkpoints = discover_checkpoints(checkpoint_root)
    feature_path = Path(feature_table_path) if feature_table_path else discover_feature_table(checkpoint_root)
    split_path = Path(session_split_path) if session_split_path else next(iter(sorted(Path(checkpoint_root).rglob("session_split.json"))), None)
    if split_path is None: raise ValueError("Checkpoint directory does not contain session_split.json")
    features = FeatureTable.from_json(feature_path)
    first_payload = torch.load(checkpoints[0][1], map_location="cpu", weights_only=False)
    vocabulary = {str(key): int(value) for key, value in first_payload["vocabulary"].items()}
    device = resolve_device(options.device)
    sessions = load_phase1_validation_sessions(source_csv, split_path)
    # Same session splitter as segmentation, but options only overlap in these fields.
    from .segmentation_probe import SegmentationProbeOptions
    partitions = split_probe_sessions(sessions, SegmentationProbeOptions(split_seed=options.split_seed, train_fraction=options.train_fraction, validation_fraction=options.validation_fraction))
    session_sets = {split: {f"{item.session.corpus_id}:{item.session.session_id}" for item in partitions[split]} for split in PARTITIONS}
    if any(session_sets[a] & session_sets[b] for i, a in enumerate(PARTITIONS) for b in PARTITIONS[i + 1:]):
        raise AssertionError("Probe session partitions overlap")
    events = balance_phoneme_events(all_phoneme_events(partitions), options)
    classes = tuple(sorted({event.phoneme for event in events}))
    class_index = {phone: index for index, phone in enumerate(classes)}
    split_indices = {split: np.asarray([i for i, event in enumerate(events) if event.split == split], dtype=int) for split in PARTITIONS}
    labels = np.asarray([class_index[event.phoneme] for event in events], dtype=int)
    # Every checkpoint receives this one immutable event list; extraction indexes by item_id.
    if len({event.item_id for event in events}) != len(events): raise AssertionError("Selected events are not unique")
    class_split_counts = {split: {phone: int(((labels[split_indices[split]] == index)).sum()) for index, phone in enumerate(classes)} for split in PARTITIONS}
    _write_table(output_root / "configs" / "phoneme_event_manifest.tsv", [asdict(event) for event in events])
    (output_root / "configs" / "run_config.json").write_text(json.dumps({**asdict(options), "classes": classes,
        "source_csv": str(Path(source_csv).resolve()), "checkpoint_root": str(Path(checkpoint_root).resolve()),
        "readout_frame": "phoneme start_frame + 2 (unique envelope peak)", "split_unit": "Phase-1 validation session",
        "balancing": "equal phoneme-token count within each split", "classifier": "deterministic L2 multinomial logistic regression (LBFGS)",
        "pca": "fitted train-only; shared basis is reused for checkpoint panels", "class_split_token_counts": class_split_counts,
        "session_ids": {split: sorted(f'{x.session.corpus_id}:{x.session.session_id}' for x in partitions[split]) for split in PARTITIONS},
        "event_counts": {split: int(len(split_indices[split])) for split in PARTITIONS},
        "m00_initialization_seed": int(first_payload["config"]["seed"])}, indent=2) + "\n", encoding="utf-8")
    all_sessions = [session for split in PARTITIONS for session in partitions[split]]
    model_specs: list[tuple[str, Path | None, float]] = [("M00", None, 0.0), *checkpoints]
    def load_frozen(spec_path: Path | None) -> CausalPhonemeGRU:
        if spec_path is None:
            return _make_m00(first_payload, features.width, len(vocabulary), device)
        payload = torch.load(spec_path, map_location="cpu", weights_only=False)
        if {str(key): int(value) for key, value in payload["vocabulary"].items()} != vocabulary:
            raise ValueError("Checkpoint vocabulary differs from M01")
        loaded = CausalPhonemeGRU(features.width, int(payload["config"]["hidden_size"]), int(payload["config"]["num_layers"]), len(vocabulary), float(payload["config"].get("dropout", 0.0)))
        loaded.load_state_dict(payload["model"])
        return loaded.to(device).requires_grad_(False).eval()
    summary: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    pca_point_rows: list[dict[str, object]] = []
    pca_centroid_rows: list[dict[str, object]] = []
    config = first_payload["config"]
    raw_inputs = extract_phoneme_peak_inputs(all_sessions, events, features, vocabulary, float(config.get("noise_sigma", .05)), config.get("phoneme_envelope"), options.input_noise_seed)
    raw_probe = fit_multinomial_probe(raw_inputs[split_indices["train"]], labels[split_indices["train"]], raw_inputs[split_indices["validation"]], labels[split_indices["validation"]], classes, options)
    # First pass: only immutable balanced train events from all 31 models enter PCA.
    pca_centers: list[np.ndarray] = []
    pca_scatters: list[np.ndarray] = []
    pca_counts: list[int] = []
    train_events = [events[int(i)] for i in split_indices["train"]]
    train_sessions = partitions["train"]
    for _, checkpoint_path, _ in tqdm(model_specs, desc="Shared PCA train states", unit="model"):
        pca_model = load_frozen(checkpoint_path)
        train_state = extract_phoneme_peak_states(pca_model, train_sessions, train_events, features, vocabulary, float(config.get("noise_sigma", .05)), config.get("phoneme_envelope"), options.input_noise_seed, options.sequence_chunk_frames, device).numpy()
        center = train_state.mean(0)
        pca_centers.append(center); pca_scatters.append((train_state - center).T @ (train_state - center)); pca_counts.append(len(train_state))
        del train_state
        del pca_model
        if device.type == "cuda": torch.cuda.empty_cache()
    pca_components, pca_evr = fit_pooled_pca_from_scatters(pca_centers, pca_scatters, pca_counts)
    np.savez(output_root / "configs" / "shared_pooled_pca.npz", checkpoint_ids=np.asarray([x[0] for x in model_specs]), components=pca_components, explained_variance_ratio=pca_evr,
             **{f"train_center_{index:02d}": center for index, center in enumerate(pca_centers)})
    del pca_scatters, pca_counts
    composite_panels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_number, (checkpoint_id, checkpoint_path, hours) in enumerate(tqdm(model_specs, desc="Phoneme decoding", unit="model")):
        model = load_frozen(checkpoint_path)
        states = extract_phoneme_peak_states(model, all_sessions, events, features, vocabulary, float(config.get("noise_sigma", .05)), config.get("phoneme_envelope"), options.input_noise_seed, options.sequence_chunk_frames, device).numpy()
        probe = fit_multinomial_probe(states[split_indices["train"]], labels[split_indices["train"]], states[split_indices["validation"]], labels[split_indices["validation"]], classes, options)
        np.savez(output_root / "heads" / f"{checkpoint_id}_decoder.npz", classes=np.asarray(classes), weight=probe.weight, bias=probe.bias,
                 train_center=probe.standardizer.center, train_scale=probe.standardizer.scale, l2=probe.l2)
        row: dict[str, object] = {"checkpoint_id": checkpoint_id, "checkpoint_hours": hours, "l2": probe.l2}
        cached_test_logits = probe.logits(states[split_indices["test"]])
        for split in PARTITIONS:
            result = (evaluate_multinomial_logits(cached_test_logits, labels[split_indices[split]], len(classes)) if split == "test"
                      else evaluate_multinomial_probe(probe, states[split_indices[split]], labels[split_indices[split]]))
            row.update({f"{split}_{name}": value for name, value in result.items()})
            row.update({f"{split}_raw_input_{name}": value for name, value in evaluate_multinomial_probe(raw_probe, raw_inputs[split_indices[split]], labels[split_indices[split]]).items()})
            row.update({f"{split}_majority_{name}": value for name, value in evaluate_majority_baseline(labels[split_indices["train"]], labels[split_indices[split]], len(classes)).items()})
        summary.append(row); _write_table(output_root / "summary" / "checkpoint_metrics.tsv", summary)
        # Figure shows held-out tokens only; class-capped deterministically for legibility.
        test = split_indices["test"]
        shown = []
        shown_classes = range(min(len(classes), options.max_pca_classes))
        for number in shown_classes:
            candidates = test[labels[test] == number]
            shown.extend(candidates[:options.pca_points_per_phoneme])
        shown_array = np.asarray(shown, dtype=int)
        coordinates = (states[shown_array] - pca_centers[model_number]) @ pca_components.T
        figure_classes = classes[:options.max_pca_classes]
        save_pca_figure(output_root / "figures" / f"{checkpoint_id}_shared_pca.png", coordinates, labels[shown_array], figure_classes, checkpoint_id, pca_evr)
        if checkpoint_id in {"M00", "M01", "M05", "M10", "M20", "M30"}:
            composite_panels[checkpoint_id] = (coordinates, labels[shown_array])
        for event_index, coordinate in zip(shown_array, coordinates, strict=True):
            pca_point_rows.append({"checkpoint_id": checkpoint_id, "item_id": events[int(event_index)].item_id,
                "phoneme": events[int(event_index)].phoneme, "pc1": float(coordinate[0]), "pc2": float(coordinate[1])})
        for number, phone in enumerate(figure_classes):
            points = coordinates[labels[shown_array] == number]
            pca_centroid_rows.append({"checkpoint_id": checkpoint_id, "phoneme": phone, "pc1": float(points[:, 0].mean()), "pc2": float(points[:, 1].mean()), "n": len(points)})
        _write_table(output_root / "summary" / "pca_points.tsv", pca_point_rows)
        _write_table(output_root / "summary" / "pca_centroids.tsv", pca_centroid_rows)
        test_logits = cached_test_logits
        prediction = test_logits.argmax(axis=1)
        for index, predicted in zip(test, prediction, strict=True):
            event = events[int(index)]
            prediction_rows.append({"checkpoint_id": checkpoint_id, "checkpoint_hours": hours,
                "item_id": event.item_id, "corpus_id": event.corpus_id, "session_id": event.session_id,
                "phoneme": event.phoneme, "true_class": int(labels[index]), "predicted_class": int(predicted),
                "predicted_phoneme": classes[int(predicted)]})
        for truth in range(len(classes)):
            for predicted in range(len(classes)):
                confusion_rows.append({"checkpoint_id": checkpoint_id, "checkpoint_hours": hours,
                    "true_phoneme": classes[truth], "predicted_phoneme": classes[predicted],
                    "count": int(((labels[test] == truth) & (prediction == predicted)).sum())})
        _write_table(output_root / "predictions" / "test_predictions.tsv", prediction_rows)
        _write_table(output_root / "summary" / "test_confusion_matrices.tsv", confusion_rows)
        del model, states
        if device.type == "cuda": torch.cuda.empty_cache()
    save_six_panel_pca(output_root / "figures" / "pca_six_checkpoint_comparison", composite_panels, classes[:options.max_pca_classes], pca_evr)
    save_trajectory_figure(output_root / "figures" / "decoding_trajectory", summary)
    (output_root / "run_manifest.json").write_text(json.dumps({"status": "complete", "models": len(model_specs), "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "Current-input phoneme identity decodability; not evidence by itself for abstract phonological categories."}, indent=2) + "\n", encoding="utf-8")
    return Path(shutil.make_archive(str(output_root.parent / "phoneme_decoding_probe_results"), "zip", root_dir=output_root))
