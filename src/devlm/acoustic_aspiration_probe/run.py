"""Evaluation runner for the Martin et al. aspiration-probe replication.

No probe weight ever feeds back to a developmental checkpoint.  The expensive
audio pass is deliberately checkpoint-by-checkpoint so Colab keeps bounded RAM.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from devlm.acoustic.features import load_wav_mono, log_mel_spectrogram
from devlm.acoustic.model import CausalLogMelGRU
from devlm.acoustic.train import standardize

from .core import (
    ProbeItem, _binary_auc, pooled_overlapping_frames, prevalence_weighted_ovo_auc,
    select_aspiration_items, select_control_items, stratified_fixed_item_folds,
)
from .mald import parse_phone_textgrid, prepare_mald

CONTROL_TASKS = ("control_consonant_vowel", "control_stress", "control_distant_before", "control_distant_after")
ASPIRATION_TASKS = ("aspiration_phonemic", "aspiration_phonetic")
REGULARIZATION = (0.01, 0.1, 1.0, 10.0)


def discover_acoustic_checkpoints(root: str | Path) -> list[tuple[str, Path, float]]:
    """Locate exactly thirty independently trained acoustic checkpoints."""
    found: list[tuple[float, int, Path]] = []
    for path in sorted(Path(root).rglob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not {"model", "config", "metadata"}.issubset(payload):
                continue
            config, metadata = payload["config"], payload["metadata"]
            if not {"n_mels", "hidden_size", "num_layers", "future_horizons_frames"}.issubset(config):
                continue
            if "equivalent_input_duration_hours" not in metadata:
                continue
            found.append((float(metadata["equivalent_input_duration_hours"]), int(metadata.get("optimizer_step", 0)), path))
        except Exception:
            continue
    unique = {(hours, step): (hours, step, path) for hours, step, path in found}
    ordered = sorted(unique.values())
    if len(ordered) != 30:
        raise ValueError(f"Expected exactly 30 acoustic Phase-1 checkpoints; found {len(ordered)} valid unique files")
    return [(f"M{index + 1:02d}", path, hours) for index, (hours, _step, path) in enumerate(ordered)]


def _audio_index(audio_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in audio_root.rglob("*"):
        if path.suffix.lower() != ".wav":
            continue
        key = path.stem.upper()
        # Duplicates only matter when both collections contain the same
        # filename.  MALD's paired annotations use matching roots; preserve
        # the first deterministically and audit the ambiguity in selection QC.
        index.setdefault(key, path)
    if not index:
        raise FileNotFoundError(f"No WAV files below {audio_root}")
    return index


def collect_stimuli(audio_root: str | Path, textgrid_root: str | Path, *, max_items: int | None = None) -> tuple[list[ProbeItem], dict]:
    """Build all Table-1 items from official MALD audio/annotations."""
    audio = _audio_index(Path(audio_root)); items: list[ProbeItem] = []; skipped = Counter()
    grids = sorted(Path(textgrid_root).rglob("*.TextGrid"))
    for grid in tqdm(grids, desc="Matching MALD Table-1 stimuli", unit="TextGrid", dynamic_ncols=True):
        candidate = audio.get(grid.stem.upper())
        if candidate is None:
            skipped["no_matching_wav"] += 1; continue
        try:
            phones = parse_phone_textgrid(grid)
        except ValueError:
            skipped["malformed_phone_tier"] += 1; continue
        if not phones:
            skipped["empty_phone_tier"] += 1; continue
        source = str(candidate)
        items.extend(select_aspiration_items(source, phones)); items.extend(select_control_items(source, phones))
    if max_items is not None:
        # Explicit smoke-only option, stratified by every class-bearing factor.
        # Never default: formal runs retain every Table-1 match.
        trimmed: list[ProbeItem] = []
        groups: dict[tuple[str, str, int], list[ProbeItem]] = defaultdict(list)
        for item in items: groups[(item.task, item.poa, item.label)].append(item)
        for group in sorted(groups): trimmed.extend(groups[group][:max_items])
        items = trimmed
    counts = Counter((item.task, item.poa, item.label) for item in items)
    return items, {"textgrids_seen": len(grids), "wavs_indexed": len(audio), "skipped": dict(skipped), "counts": {"|".join(map(str, key)): value for key, value in sorted(counts.items())}}


def _model_from_checkpoint(path: Path) -> tuple[CausalLogMelGRU, dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config, metadata = payload["config"], payload["metadata"]
    model = CausalLogMelGRU(int(config["n_mels"]), int(config["hidden_size"]), int(config["num_layers"]), list(config["future_horizons_frames"]), float(config.get("dropout", 0.0)))
    model.load_state_dict(payload["model"]); model.eval()
    return model, config, metadata


def _resolve_fbank_backend(*, allow_compatibility_fallback: bool):
    """Resolve once: never mix Kaldi and fallback features within one run."""
    try:
        import torchaudio
        _ = torchaudio.compliance.kaldi.fbank
        return torchaudio, "torchaudio.compliance.kaldi.fbank (80 bins, 25-ms, 10-ms, dither=0)"
    except Exception as exc:
        if not allow_compatibility_fallback:
            raise RuntimeError(
                "Formal replication requires torchaudio.compliance.kaldi.fbank for the paper's 80-D baseline. "
                "Install a torchaudio build ABI-compatible with the installed torch; do not silently substitute project log-Mel."
            ) from exc
        return None, "SMOKE-ONLY fallback: project log-Mel frontend (Kaldi Fbank unavailable)"


def _kaldi_or_compatible_fbank(waveform: torch.Tensor, rate: int, backend) -> np.ndarray:
    """One uniform Fbank backend selected before extraction begins."""
    if backend is None:
        return log_mel_spectrogram(waveform, rate).numpy()
    torchaudio = backend
    if rate != 16_000:
        waveform = torch.nn.functional.interpolate(waveform[None, None], size=round(len(waveform) * 16_000 / rate), mode="linear", align_corners=False)[0, 0]
    # Do not catch this call: a per-file Kaldi failure invalidates formal
    # comparability and must fail rather than create a mixed representation.
    value = torchaudio.compliance.kaldi.fbank(waveform[None], sample_frequency=16_000, num_mel_bins=80, frame_length=25.0, frame_shift=10.0, dither=0.0, energy_floor=0.0, use_energy=False, snip_edges=True)
    return value.numpy().astype(np.float32)


def extract_representations(items: list[ProbeItem], *, checkpoint: Path | None, allow_fbank_fallback: bool = False) -> tuple[dict[str, np.ndarray], str]:
    """Pool frozen GRU states (or the external 80-D Kaldi Fbank) per phone."""
    grouped: dict[str, list[ProbeItem]] = defaultdict(list)
    for item in items: grouped[item.audio_path].append(item)
    model = config = metadata = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = ""
    fbank_backend = None
    if checkpoint is not None:
        model, config, metadata = _model_from_checkpoint(checkpoint)
        model.to(device)
        normalization = metadata.get("log_mel_normalization")
        if not normalization:
            raise ValueError(f"{checkpoint} lacks saved train-only log-Mel normalization")
        backend = "frozen causal GRU state after each left-aligned 25-ms/10-ms model frame"
    else:
        fbank_backend, backend = _resolve_fbank_backend(allow_compatibility_fallback=allow_fbank_fallback)
    values: dict[str, np.ndarray] = {}
    for audio_path, phone_items in tqdm(grouped.items(), desc=f"Extracting {'Kaldi Fbank' if checkpoint is None else checkpoint.name}", unit="WAV", dynamic_ncols=True):
        waveform, rate = load_wav_mono(audio_path)
        if checkpoint is None:
            frames = _kaldi_or_compatible_fbank(waveform, rate, fbank_backend)
            # The requested pooling support remains 25/10, including for the
            # baseline; Kaldi's snip_edges frame origin is 0 in this setup.
        else:
            features = log_mel_spectrogram(waveform, rate, target_sample_rate=int(config["sample_rate"]), n_mels=int(config["n_mels"]), n_fft=int(config["n_fft"]), win_length=int(config["win_length"]), hop_length=int(config["hop_length"]))
            with torch.no_grad():
                states, _ = model.gru(standardize(features, normalization).unsqueeze(0).to(device).contiguous())
            frames = states[0].cpu().numpy()
        for item in phone_items:
            values[item.item_id] = pooled_overlapping_frames(frames, phone_start_s=item.target_start_s, phone_end_s=item.target_end_s)
    return values, backend


def _fit_pca(train: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    _, _, vectors = np.linalg.svd(train - mean, full_matrices=False)
    return mean, vectors[:dimensions].T


def _softmax(value: np.ndarray) -> np.ndarray:
    value = value - value.max(axis=1, keepdims=True); value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, classes: np.ndarray, regularization: float) -> np.ndarray:
    """Deterministic L2 multinomial logistic regression; C is tuned inner-CV."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise RuntimeError("This probe requires scikit-learn>=1.3; install it without replacing PyTorch.") from exc
    # Current sklearn selects multinomial automatically for multiclass lbfgs;
    # ``multi_class='auto'`` was removed in sklearn 1.8.
    classifier = LogisticRegression(C=regularization, solver="lbfgs", max_iter=500, random_state=0)
    classifier.fit(train_x, train_y)
    source = classifier.predict_proba(test_x); ordered = np.zeros((len(test_x), len(classes)), dtype=float)
    for column, label in enumerate(classifier.classes_):
        ordered[:, int(np.where(classes == label)[0][0])] = source[:, column]
    return ordered


def _score(y: np.ndarray, probability: np.ndarray) -> float:
    return _binary_auc((y == 1).astype(np.int8), probability[:, 1]) if probability.shape[1] == 2 else prevalence_weighted_ovo_auc(y, probability)


def _fold_score(x: np.ndarray, y: np.ndarray, folds: np.ndarray, dimensions: int, regularization: float) -> tuple[float, np.ndarray]:
    classes = np.unique(y); full_probability = np.zeros((len(y), len(classes)), dtype=np.float64)
    for fold in sorted(set(folds.tolist())):
        train_mask, test_mask = folds != fold, folds == fold
        width = min(dimensions, x.shape[1], train_mask.sum() - 1)
        if width < 1: raise ValueError("Insufficient training samples for PCA")
        if width < x.shape[1]:
            mean, basis = _fit_pca(x[train_mask], width); train_x = (x[train_mask] - mean) @ basis; test_x = (x[test_mask] - mean) @ basis
        else: train_x, test_x = x[train_mask], x[test_mask]
        full_probability[test_mask] = _fit_predict(train_x, y[train_mask], test_x, classes, regularization)
    return _score(y, full_probability), full_probability


def inner_ten_fold_assignment(ids: list[str], labels: np.ndarray, outer_folds: np.ndarray, outer_fold: int, *, seed: int = 20261006) -> np.ndarray:
    """A fresh feature-independent, stratified 10-fold split inside outer train."""
    train_ids = [item_id for item_id, fold in zip(ids, outer_folds, strict=True) if fold != outer_fold]
    train_labels = [int(label) for label, fold in zip(labels, outer_folds, strict=True) if fold != outer_fold]
    assigned = stratified_fixed_item_folds(train_ids, train_labels, n_folds=10, seed=seed + outer_fold)
    return np.asarray([assigned[item_id] for item_id in train_ids], dtype=int)


def nested_cv_score(x: np.ndarray, y: np.ndarray, fold_by_id: dict[str, int], ids: list[str], dimensions: int) -> float:
    """Ten outer folds + inner CV, with PCA fitted only inside each training fold."""
    folds = np.asarray([fold_by_id[item] for item in ids]); classes = np.unique(y)
    if len(classes) < 2 or min(np.bincount(np.searchsorted(classes, y))) < 12:
        raise ValueError("Each probe class needs at least 12 items so every outer-training split supports an inner ten-fold CV")
    predictions = np.zeros((len(y), len(classes)), dtype=float)
    for outer in range(10):
        outer_train, outer_test = folds != outer, folds == outer
        inner_folds = inner_ten_fold_assignment(ids, y, folds, outer)
        if set(inner_folds.tolist()) != set(range(10)):
            raise RuntimeError("Inner CV must contain exactly ten folds")
        candidates: list[tuple[float, float]] = []
        for regularization in REGULARIZATION:
            # These folds are inner because outer holdout is excluded.  PCA is
            # separately re-fit in _fold_score's each inner training split.
            candidates.append((_fold_score(x[outer_train], y[outer_train], inner_folds, dimensions, regularization)[0], regularization))
        chosen = max(candidates)[1]
        width = min(dimensions, x.shape[1], outer_train.sum() - 1)
        if width < x.shape[1]:
            mean, basis = _fit_pca(x[outer_train], width); train_x = (x[outer_train] - mean) @ basis; test_x = (x[outer_test] - mean) @ basis
        else: train_x, test_x = x[outer_train], x[outer_test]
        predictions[outer_test] = _fit_predict(train_x, y[outer_train], test_x, classes, chosen)
    return _score(y, predictions)


def _dimensions(width: int) -> list[int]:
    candidates = [2 ** power for power in range(1, int(math.log2(width)) + 1)]
    return sorted(set([value for value in candidates if value <= width] + [width]))


def _task_arrays(items: list[ProbeItem], representations: dict[str, np.ndarray], task: str, poa: str = "all") -> tuple[np.ndarray, np.ndarray, list[str]]:
    subset = [item for item in items if item.task == task and (poa == "all" or item.poa == poa)]
    return np.stack([representations[item.item_id] for item in subset]), np.asarray([item.label for item in subset]), [item.item_id for item in subset]


def _write_tsv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t"); writer.writeheader(); writer.writerows(rows)


def _read_json_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def cache_fingerprint(items: list[ProbeItem], *, representation_identity: str, max_items: int | None) -> str:
    """Fingerprint all feature-independent inputs that define cached results."""
    ordered = sorted((item.item_id, item.task, item.poa, item.label, item.target_start_s, item.target_end_s, item.audio_path) for item in items)
    payload = {"schema": 2, "items": ordered, "representation_identity": representation_identity, "max_items": max_items, "outer_folds": 10, "inner_folds": 10, "fold_seed": 20261006, "regularization_C": REGULARIZATION, "dimensions": "powers_of_two_plus_terminal"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]


def _checkpoint_identity(checkpoint: Path | None) -> str:
    if checkpoint is None:
        return "kaldi_fbank_v1"
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"{checkpoint.name}:{checkpoint.stat().st_size}:{digest.hexdigest()}"


def _save_aspiration_vectors(path: Path, items: list[ProbeItem], representations: dict[str, np.ndarray]) -> None:
    targets = [item for item in items if item.task in ASPIRATION_TASKS]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, item_ids=np.asarray([item.item_id for item in targets]), vectors=np.stack([representations[item.item_id] for item in targets]))


def _load_aspiration_vectors(path: Path) -> dict[str, np.ndarray]:
    saved = np.load(path, allow_pickle=False)
    return {str(item_id): vector for item_id, vector in zip(saved["item_ids"], saved["vectors"], strict=True)}


def _plot(rows: list[dict], output: Path, *, selected: bool, controls: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    mode = "selected" if selected else "full"
    subset = [row for row in rows if row["representation"] != "kaldi_fbank" and row["dimension_mode"] == mode and (row["task"].startswith("control") == controls) and (controls or row["poa"] == "mean_three_poa")]
    fig, ax = plt.subplots(figsize=(9, 4))
    for task in sorted({row["task"] for row in subset}):
        trial = sorted([row for row in subset if row["task"] == task], key=lambda row: int(row["checkpoint_id"][1:]))
        ax.plot([row["checkpoint_hours"] for row in trial], [row["score"] for row in trial], marker="o", label=task.replace("_", " "))
        baseline = [row for row in rows if row["representation"] == "kaldi_fbank" and row["task"] == task and row["dimension_mode"] == mode and (controls or row["poa"] == "mean_three_poa")]
        if baseline:
            ax.axhline(float(baseline[-1]["score"]), color="firebrick", linestyle="--", linewidth=1.1, label=f"Kaldi Fbank: {task.replace('_', ' ')}")
    ax.axhline(.5, color="black", linestyle="--", linewidth=1); ax.set(xlabel="Acoustic training exposure (hours)", ylabel="ROC-AUC", ylim=(0, 1)); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def run_replication(checkpoint_root: str | Path, output_dir: str | Path, mald_cache: str | Path, *, max_items: int | None = None) -> Path:
    """Run full/checkpoint-averaged dimensionality replication, then report."""
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    # Validate the paper-required external baseline before downloading MALD or
    # extracting even one checkpoint.  A formal run must fail here, not after
    # many hours of GPU/CPU work.
    _unused_backend, verified_fbank_backend = _resolve_fbank_backend(allow_compatibility_fallback=max_items is not None)
    print(f"Verified baseline backend: {verified_fbank_backend}", flush=True)
    checkpoints = discover_acoustic_checkpoints(checkpoint_root)
    audio, grids = prepare_mald(mald_cache)
    items, qc = collect_stimuli(audio, grids, max_items=max_items)
    if not items: raise ValueError("MALD matching produced no Table-1 items")
    _write_tsv(output / "stimuli.tsv", [asdict(item) for item in items])
    (output / "selection_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    folds: dict[str, dict[str, int]] = {}
    for task in CONTROL_TASKS:
        relevant = [item for item in items if item.task == task]
        folds[task] = stratified_fixed_item_folds([item.item_id for item in relevant], [item.label for item in relevant])
    for task in ASPIRATION_TASKS:
        for poa in ("labial", "alveolar", "velar"):
            relevant = [item for item in items if item.task == task and item.poa == poa]
            folds[f"{task}|{poa}"] = stratified_fixed_item_folds([item.item_id for item in relevant], [item.label for item in relevant])
    (output / "fixed_item_folds.json").write_text(json.dumps(folds, indent=2, sort_keys=True) + "\n")

    all_rows: list[dict] = []; control_grid: dict[str, dict[int, dict[str, float]]] = defaultdict(lambda: defaultdict(dict)); backends: dict[str, str] = {}
    representations_to_run: list[tuple[str, Path | None, float]] = [(identifier, path, hours) for identifier, path, hours in checkpoints] + [("kaldi_fbank", None, float("nan"))]
    cache_dir = output / "representation_cache"
    vector_cache_dir = Path(mald_cache) / "aspiration_pooled_vectors"
    for identifier, checkpoint, hours in representations_to_run:
        print(f"\nProbe representation: {identifier}", flush=True)
        fingerprint = cache_fingerprint(items, representation_identity=_checkpoint_identity(checkpoint), max_items=max_items)
        cached = cache_dir / f"{identifier}.{fingerprint}.controls.json"
        if cached.exists():
            rows = _read_json_rows(cached); print(f"Reusing completed control cache: {cached.name}", flush=True)
        else:
            representations, backend = extract_representations(items, checkpoint=checkpoint, allow_fbank_fallback=max_items is not None); backends[identifier] = backend
            _save_aspiration_vectors(vector_cache_dir / f"{identifier}.{fingerprint}.npz", items, representations)
            width = next(iter(representations.values())).shape[0]; rows = []
            for task in CONTROL_TASKS:
                x, y, ids = _task_arrays(items, representations, task)
                for dimension in _dimensions(width):
                    score = nested_cv_score(x, y, folds[task], ids, dimension)
                    rows.append({"representation": identifier, "checkpoint_id": identifier if identifier.startswith("M") else "baseline", "checkpoint_hours": hours, "task": task, "poa": "all", "dimension": dimension, "dimension_mode": "grid", "score": score, "n_items": len(y), "n_classes": len(np.unique(y)), "backend": backend})
            _write_json_rows(cached, rows)
        for row in rows:
            backends.setdefault(identifier, row["backend"])
            control_grid[identifier][int(row["dimension"])][row["task"]] = float(row["score"])
        all_rows.extend(rows)

    selected_dimensions: dict[str, int] = {}
    for family, identifiers in {"acoustic_gru": [identifier for identifier, _, _ in representations_to_run if identifier.startswith("M")], "kaldi_fbank": ["kaldi_fbank"]}.items():
        possible = set.intersection(*(set(control_grid[identifier]) for identifier in identifiers))
        scores = {dimension: sum(control_grid[identifier][dimension]["control_consonant_vowel"] + control_grid[identifier][dimension]["control_stress"] - control_grid[identifier][dimension]["control_distant_before"] - control_grid[identifier][dimension]["control_distant_after"] for identifier in identifiers) for dimension in sorted(possible)}
        selected_dimensions[family] = max(scores.items(), key=lambda pair: (pair[1], pair[0]))[0]
        for dimension, score in scores.items(): all_rows.append({"representation": family, "checkpoint_id": "family", "checkpoint_hours": "", "task": "control_score", "poa": "all", "dimension": dimension, "dimension_mode": "selection", "score": score, "n_items": "", "n_classes": "", "backend": "checkpoint-summed controls"})
    (output / "selected_dimensions.json").write_text(json.dumps(selected_dimensions, indent=2) + "\n")

    # Second pass: full and selected dimensions for both aspiration tasks, and
    # selected control rows suitable for paper-analog plots.
    for identifier, checkpoint, hours in representations_to_run:
        family = "acoustic_gru" if identifier.startswith("M") else "kaldi_fbank"; selected = selected_dimensions[family]
        fingerprint = cache_fingerprint(items, representation_identity=_checkpoint_identity(checkpoint), max_items=max_items)
        cached = cache_dir / f"{identifier}.{fingerprint}.selected_{selected}.json"
        if cached.exists():
            rows = _read_json_rows(cached); print(f"Reusing completed aspiration cache: {cached.name}", flush=True)
        else:
            vectors = vector_cache_dir / f"{identifier}.{fingerprint}.npz"
            if not vectors.exists():
                # A prior interruption before control-cache completion cannot
                # provide vectors; create both caches from one recovery pass.
                representations, backend = extract_representations(items, checkpoint=checkpoint, allow_fbank_fallback=max_items is not None)
                _save_aspiration_vectors(vectors, items, representations)
            else:
                representations = _load_aspiration_vectors(vectors)
                backend = backends.get(identifier, "pooled-vector cache")
            width = next(iter(representations.values())).shape[0]; rows = []
            for task in CONTROL_TASKS:
                grid_rows = [row for row in all_rows if row["representation"] == identifier and row["task"] == task and row["dimension_mode"] == "grid" and int(row["dimension"]) == selected]
                source = grid_rows[-1]
                rows.append({**source, "dimension_mode": "selected"})
            for task in ASPIRATION_TASKS:
                poa_scores = []
                for poa in ("labial", "alveolar", "velar"):
                    x, y, ids = _task_arrays(items, representations, task, poa)
                    for mode, dimension in (("full", width), ("selected", selected)):
                        score = nested_cv_score(x, y, folds[f"{task}|{poa}"], ids, dimension); poa_scores.append((mode, dimension, score, len(y)))
                        rows.append({"representation": identifier, "checkpoint_id": identifier if identifier.startswith("M") else "baseline", "checkpoint_hours": hours, "task": task, "poa": poa, "dimension": dimension, "dimension_mode": mode, "score": score, "n_items": len(y), "n_classes": 3, "backend": backend})
                for mode, dimension in (("full", width), ("selected", selected)):
                    relevant = [score for saved_mode, saved_dim, score, _ in poa_scores if saved_mode == mode and saved_dim == dimension]
                    rows.append({"representation": identifier, "checkpoint_id": identifier if identifier.startswith("M") else "baseline", "checkpoint_hours": hours, "task": task, "poa": "mean_three_poa", "dimension": dimension, "dimension_mode": mode, "score": float(np.mean(relevant)), "n_items": "POA mean", "n_classes": 3, "backend": backend})
            _write_json_rows(cached, rows)
        all_rows.extend(rows)
    _write_tsv(output / "probe_results.tsv", all_rows)
    _plot(all_rows, output / "figure_full_aspiration.png", selected=False, controls=False)
    _plot(all_rows, output / "figure_constrained_controls.png", selected=True, controls=True)
    _plot(all_rows, output / "figure_constrained_aspiration.png", selected=True, controls=False)
    manifest = {"replication": "Martin et al. Interspeech 2023 aspiration probe", "checkpoints": [{"id": identifier, "path": str(path), "hours": hours} for identifier, path, hours in checkpoints], "mald": "official UAlberta Scholaris word+pseudoword WAVs and time-aligned TextGrids", "phone_pooling": "mean frozen state over every 25-ms frame overlapping the annotated phone; 10-ms hop", "probe": "nested 10-fold CV, inner L2 selection, PCA fit inside each training fold", "adaptations": ["checkpoint exposure replaces HuBERT layer axis", "causal one-layer GRU future-log-Mel objective differs from HuBERT Base masked objective", "causal states only summarize preceding audio; HuBERT representations are bidirectionally contextualized", "our model support is 25 ms with a 10-ms hop; HuBERT output frame timing differs", "initial P/T/K operationally denotes initial voiceless stop context, not an aspiration diacritic stored in MALD ARPAbet"], "baseline_backends": backends, "max_items": max_items}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
