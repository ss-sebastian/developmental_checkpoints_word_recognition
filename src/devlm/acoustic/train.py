from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..train import resolve_device
from .data import AudioItem, load_audio_manifest, split_audio_sessions
from .features import load_wav_mono, log_mel_frame_count, log_mel_spectrogram
from .model import CausalLogMelGRU


FRAME_MS = 10


@dataclass
class Counters:
    optimizer_step: int = 0
    cumulative_frames_seen: int = 0
    cumulative_audio_items_seen: int = 0


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_item_features(item: AudioItem, config: dict) -> torch.Tensor:
    waveform, source_rate = load_wav_mono(item.audio_path)
    features = log_mel_spectrogram(
        waveform, source_rate,
        target_sample_rate=int(config["sample_rate"]), n_mels=int(config["n_mels"]),
        n_fft=int(config["n_fft"]), win_length=int(config["win_length"]), hop_length=int(config["hop_length"]),
    )
    if not len(features):
        raise ValueError(f"Audio is shorter than one analysis window ({int(config['win_length'])} samples): {item.audio_path}")
    return features


def standardize(features: torch.Tensor, normalization: dict) -> torch.Tensor:
    mean = torch.tensor(normalization["mean"], dtype=features.dtype)
    std = torch.tensor(normalization["std"], dtype=features.dtype)
    return (features - mean) / std


def plan_training_exposure(items: list[AudioItem], config: dict, maximum_frames: int) -> list[tuple[AudioItem, int]]:
    """Choose the manifest-ordered audio prefix; never inspect later training audio."""
    selected: list[tuple[AudioItem, int]] = []
    remaining = maximum_frames
    for item in items:
        if remaining <= 0:
            break
        count = log_mel_frame_count(
            item.audio_path, target_sample_rate=int(config["sample_rate"]),
            n_fft=int(config["n_fft"]), win_length=int(config["win_length"]), hop_length=int(config["hop_length"]),
        )
        usable = min(count, remaining)
        if usable > 0:
            selected.append((item, usable))
            remaining -= usable
    return selected


def round_robin_validation_items(items: list[AudioItem], seed: int) -> list[AudioItem]:
    """Deterministically interleave session groups before the validation cap.

    A validation cap must not turn a manifest prefix (for example, one child's
    first recording day) into the whole evaluation set.  This is deliberately
    only an evaluation ordering: it neither changes the session split nor
    introduces any training data into validation.
    """
    grouped: dict[tuple[str, str], list[AudioItem]] = {}
    for item in items:
        grouped.setdefault(item.session_key, []).append(item)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    for key in keys:
        grouped[key].sort(key=lambda item: item.order_key)
    ordered: list[AudioItem] = []
    while keys:
        remaining: list[tuple[str, str]] = []
        for key in keys:
            if grouped[key]:
                ordered.append(grouped[key].pop(0))
            if grouped[key]:
                remaining.append(key)
        keys = remaining
    return ordered


def estimate_train_normalization(planned_items: list[tuple[AudioItem, int]], config: dict, maximum_frames: int, seed: int) -> dict:
    """Estimate fixed per-Mel train-only mean/std on a deterministic clip sample."""
    ordered = list(planned_items)
    random.Random(seed + 17_171).shuffle(ordered)
    total = torch.zeros(int(config["n_mels"]), dtype=torch.float64)
    squared = torch.zeros_like(total)
    count = 0
    for item, item_limit in tqdm(ordered, desc="Estimating train-only log-Mel normalization", unit="recording", leave=False, dynamic_ncols=True):
        if count >= maximum_frames:
            break
        features = load_item_features(item, config)[:min(item_limit, maximum_frames - count)].double()
        if not len(features):
            continue
        total += features.sum(dim=0)
        squared += features.square().sum(dim=0)
        count += len(features)
    if not count:
        raise ValueError("No usable train audio frames for log-Mel normalization")
    mean = total / count
    variance = (squared / count - mean.square()).clamp_min(1e-8)
    return {
        "method": "fixed per-frequency mean/std from deterministic training-split-only audio sample",
        "normalization_frames": count,
        "normalization_hours": count * FRAME_MS / 3_600_000,
        "mean": mean.float().tolist(),
        "std": variance.sqrt().float().tolist(),
    }


def _loss_for_chunk(
    predictions: dict[int, torch.Tensor], features: torch.Tensor, global_start: int,
    *, total_frames: int, horizons: list[int], device: torch.device,
) -> tuple[torch.Tensor | None, dict[int, tuple[float, int]]]:
    """Compute only targets strictly in the future of each causal state."""
    components: list[torch.Tensor] = []
    summaries: dict[int, tuple[float, int]] = {}
    chunk_frames = predictions[horizons[0]].shape[1]
    for horizon in horizons:
        valid = min(chunk_frames, total_frames - global_start - horizon)
        if valid <= 0:
            continue
        target = features[global_start + horizon:global_start + horizon + valid].to(device)
        value = F.mse_loss(predictions[horizon][0, :valid], target)
        components.append(value)
        summaries[horizon] = (float(value.detach()) * valid, valid)
    return (torch.stack(components).mean() if components else None), summaries


def _iter_chunks(frames: torch.Tensor, chunk_frames: int):
    for start in range(0, len(frames), chunk_frames):
        yield start, min(len(frames), start + chunk_frames)


@torch.no_grad()
def validate(model: CausalLogMelGRU, items: list[AudioItem], config: dict, maximum_frames: int, normalization: dict) -> dict:
    model.eval()
    device = next(model.parameters()).device
    horizons = list(model.horizons)
    chunk_frames = int(config["sequence_chunk_frames"])
    total_by_horizon = Counter()
    count_by_horizon = Counter()
    frames_seen = 0
    items_seen = 0
    ordered_items = round_robin_validation_items(items, int(config["seed"]) + 20_261)
    sessions_seen: set[tuple[str, str]] = set()
    for item in tqdm(ordered_items, desc="Acoustic validation", unit="recording", leave=False, dynamic_ncols=True):
        if frames_seen >= maximum_frames:
            break
        features = standardize(load_item_features(item, config), normalization)
        features = features[:maximum_frames - frames_seen]
        if len(features) <= max(horizons):
            continue
        hidden = None
        for start, end in _iter_chunks(features, chunk_frames):
            chunk = features[start:end].unsqueeze(0).to(device).contiguous()
            predictions, hidden = model(chunk, hidden)
            _, summaries = _loss_for_chunk(predictions, features, start, total_frames=len(features), horizons=horizons, device=device)
            for horizon, (loss_sum, count) in summaries.items():
                total_by_horizon[horizon] += loss_sum
                count_by_horizon[horizon] += count
            hidden = hidden.detach()
        frames_seen += len(features)
        items_seen += 1
        sessions_seen.add(item.session_key)
    if not count_by_horizon:
        raise ValueError("Validation audio has no frames after applying future horizons")
    per_horizon = {str(h): total_by_horizon[h] / count_by_horizon[h] for h in horizons if count_by_horizon[h]}
    return {
        "validation_mean_future_log_mel_mse": float(np.mean(list(per_horizon.values()))),
        "validation_future_log_mel_mse_by_horizon": per_horizon,
        "validation_frames": frames_seen,
        "validation_audio_items": items_seen,
        "validation_sessions_sampled": len(sessions_seen),
        "validation_sampling_order": "seeded round-robin across session groups before the frame cap",
    }


def train(config: dict) -> dict:
    """Train the independent audio model, never touching IPA Phase 1 code/data."""
    seed = int(config["seed"])
    set_seeds(seed)
    device = resolve_device(str(config["device"]))
    print(f"Acoustic training device: {device.type.upper()}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    manifest_path = Path(config["audio_manifest_path"])
    print("Loading adult/caregiver child-directed audio manifest...", flush=True)
    items = load_audio_manifest(manifest_path)
    train_items, validation_items = split_audio_sessions(items, float(config["validation_fraction"]), seed)
    train_limit = round(float(config["max_train_hours"]) * 3_600_000 / FRAME_MS)
    validation_limit = round(float(config["max_validation_hours"]) * 3_600_000 / FRAME_MS)
    normalization_limit = min(train_limit, round(float(config["normalization_max_hours"]) * 3_600_000 / FRAME_MS))
    print(
        f"Prepared {len(items):,} recordings from {len({item.session_key for item in items}):,} sessions: "
        f"{len(train_items):,} train, {len(validation_items):,} validation. "
        f"Training cap: {float(config['max_train_hours']):.3f} h ({train_limit:,} 10-ms frames).",
        flush=True,
    )
    planned_train_items = plan_training_exposure(train_items, config, train_limit)
    planned_train_frames = sum(frames for _, frames in planned_train_items)
    if not planned_train_items:
        raise ValueError("No training WAV has enough samples for one log-Mel frame")
    if planned_train_frames < train_limit:
        print(
            f"WARNING: eligible training split contains only {planned_train_frames * FRAME_MS / 3_600_000:.3f} h; "
            f"the requested {float(config['max_train_hours']):.3f} h cap cannot be reached.",
            flush=True,
        )
    ordering_description = (
        "child-age order from target_child_age_months, then recording order"
        if all(item.target_child_age_months is not None for item in train_items)
        else "explicit manifest exposure_order (not interpreted as child age)"
    )
    (output_dir / "audio_training_exposure.json").write_text(json.dumps({
        "selection": f"manifest-ordered prefix of the training session split: {ordering_description}",
        "planned_frames": planned_train_frames,
        "planned_hours": planned_train_frames * FRAME_MS / 3_600_000,
        "ordering_seed": seed,
        "training_order": "manifest exposure order; BabySLM cache repartitioning writes a seeded random train-clip permutation",
        "items": [{"corpus_id": item.corpus_id, "session_id": item.session_id, "audio_path": str(item.audio_path), "frames_used": frames} for item, frames in planned_train_items],
    }, indent=2) + "\n", encoding="utf-8")
    source_summary = {
        "source_corpora": dict(sorted(Counter(item.source_corpus for item in items).items())),
        "speaker_roles": dict(sorted(Counter(item.speaker_role for item in items).items())),
        "directed_to_child_evidence": dict(sorted(Counter(item.directed_to_child for item in items).items())),
        "speaker_filter": "Only adult/caregiver speaker roles are accepted; target-child/CHI speech is rejected by manifest validation.",
        "audio_format": "PCM WAV only",
        "input_labels_used_for_training": "none",
    }
    (output_dir / "audio_source_manifest.json").write_text(json.dumps(source_summary, indent=2) + "\n", encoding="utf-8")
    split = {
        "train": [[item.corpus_id, item.session_id] for item in train_items],
        "validation": [[item.corpus_id, item.session_id] for item in validation_items],
    }
    (output_dir / "audio_session_split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    normalization = estimate_train_normalization(planned_train_items, config, min(normalization_limit, planned_train_frames), seed)
    normalization["configured_maximum_hours"] = float(config["normalization_max_hours"])
    (output_dir / "log_mel_normalization.json").write_text(json.dumps(normalization, indent=2) + "\n", encoding="utf-8")
    model = CausalLogMelGRU(
        int(config["n_mels"]), int(config["hidden_size"]), int(config["num_layers"]),
        list(config["future_horizons_frames"]), float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    counters = Counters()
    checkpoint_count = int(config["target_checkpoint_count"])
    if checkpoint_count <= 0:
        raise ValueError("target_checkpoint_count must be positive")
    # Schedule across the actual capped exposure. If the supplied corpus is
    # smaller than the requested cap, retain a useful developmental sequence.
    checkpoint_interval = max(1, math.ceil(planned_train_frames / checkpoint_count))
    next_checkpoint = checkpoint_interval
    last_checkpoint_frame = -1
    last_metrics: dict | None = None
    horizons = list(model.horizons)
    chunk_frames = int(config["sequence_chunk_frames"])

    def checkpoint() -> dict:
        nonlocal last_checkpoint_frame, last_metrics
        validation = validate(model, validation_items, config, validation_limit, normalization)
        metadata = {
            **asdict(counters),
            "equivalent_input_duration_hours": counters.cumulative_frames_seen * FRAME_MS / 3_600_000,
            "training_input": "10-ms log-Mel acoustic frames only; no IPA, phoneme, word-boundary, or extra Gaussian-noise input",
            "objective": "mean MSE across causal future log-Mel horizons",
            "future_horizons_frames": horizons,
            "future_horizons_ms": [h * FRAME_MS for h in horizons],
            "checkpoint_interval_frames": checkpoint_interval,
            "requested_max_train_hours": float(config["max_train_hours"]),
            "planned_train_hours": planned_train_frames * FRAME_MS / 3_600_000,
            "log_mel_normalization": normalization,
            **validation,
        }
        checkpoint_path = output_dir / f"acoustic_checkpoint_step_{counters.optimizer_step:08d}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata, "config": config}, checkpoint_path)
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata) + "\n")
        last_checkpoint_frame = counters.cumulative_frames_seen
        last_metrics = metadata
        tqdm.write(
            f"Checkpoint step={counters.optimizer_step:,} frames={counters.cumulative_frames_seen:,} "
            f"hours={metadata['equivalent_input_duration_hours']:.3f} "
            f"val_future_mse={metadata['validation_mean_future_log_mel_mse']:.6f}"
        )
        return metadata

    model.train()
    progress = tqdm(planned_train_items, desc="Acoustic training", unit="recording", dynamic_ncols=True)
    for item, planned_frames in progress:
        if counters.cumulative_frames_seen >= planned_train_frames:
            break
        features = standardize(load_item_features(item, config)[:planned_frames], normalization)
        if len(features) <= max(horizons):
            continue
        hidden = None
        for start, end in _iter_chunks(features, chunk_frames):
            chunk = features[start:end].unsqueeze(0).to(device).contiguous()
            predictions, hidden = model(chunk, hidden)
            loss, _ = _loss_for_chunk(predictions, features, start, total_frames=len(features), horizons=horizons, device=device)
            if loss is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                optimizer.step()
                counters.optimizer_step += 1
            hidden = hidden.detach()
            counters.cumulative_frames_seen += end - start
            progress.set_postfix(
                step=counters.optimizer_step, frames=counters.cumulative_frames_seen,
                hours=f"{counters.cumulative_frames_seen * FRAME_MS / 3_600_000:.3f}",
                loss=f"{float(loss.detach()):.4f}" if loss is not None else "n/a",
            )
            if counters.cumulative_frames_seen >= next_checkpoint:
                checkpoint()
                while next_checkpoint <= counters.cumulative_frames_seen:
                    next_checkpoint += checkpoint_interval
                model.train()
            if counters.cumulative_frames_seen >= planned_train_frames:
                break
        counters.cumulative_audio_items_seen += 1
    if counters.optimizer_step == 0:
        raise ValueError("Training audio contains no usable chunks after future-horizon filtering")
    if last_checkpoint_frame != counters.cumulative_frames_seen:
        checkpoint()
    run_manifest = {
        "run_type": "independent_acoustic_log_mel_phase1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_hash(manifest_path),
        "configuration": config,
        "counters": asdict(counters),
        "planned_training_exposure_frames": planned_train_frames,
        "source": source_summary,
        "status": "complete",
    }
    (output_dir / "acoustic_run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
    return last_metrics or {}
