from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from .data import Session, load_ipa_childes, split_sessions
from .features import FeatureTable
from .model import CausalPhonemeGRU
from .stream import FrameStream, build_session_stream
from .stream import PHONEME_FRAMES, PHONEME_OVERLAP_FRAMES, UTTERANCE_PAUSE_FRAMES
from .timing import FRAME_MS


@dataclass
class Counters:
    optimizer_step: int = 0
    cumulative_frames_seen: int = 0
    cumulative_phonemes_seen: int = 0
    cumulative_utterances_seen: int = 0


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested, but CUDA is not available in this runtime")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    return torch.device(requested)


def _phonemes(session: Session, table: FeatureTable) -> tuple[str, ...]:
    return tuple(p for u in session.utterances for p in (u.phonemes or table.tokenize(u.ipa)))


def make_stream(session: Session, table: FeatureTable, vocabulary: dict[str, int], cfg: dict, rng: np.random.Generator) -> FrameStream:
    return build_session_stream(
        session, table, vocabulary, rng,
        noise_sigma=float(cfg["noise_sigma"]),
        phoneme_envelope=cfg.get("phoneme_envelope"),
    )


def estimate_input_frames(sessions: list[Session], table: FeatureTable) -> int:
    """Calculate exact exposure under fixed phoneme and utterance timing."""
    total = 0
    stride = PHONEME_FRAMES - PHONEME_OVERLAP_FRAMES
    for session in sessions:
        for utterance in session.utterances:
            phonemes = utterance.phonemes or table.tokenize(utterance.ipa)
            if phonemes:
                total += PHONEME_FRAMES + (len(phonemes) - 1) * stride
        total += max(0, len(session.utterances) - 1) * UTTERANCE_PAUSE_FRAMES
    return total


def chunk_slices(total_frames: int, chunk_frames: int):
    if chunk_frames <= 0:
        raise ValueError("sequence_chunk_frames must be positive")
    for start in range(0, total_frames, chunk_frames):
        yield start, min(total_frames, start + chunk_frames)


@torch.no_grad()
def validate(model: CausalPhonemeGRU, sessions: list[Session], table: FeatureTable, vocabulary: dict[str, int], cfg: dict, seed: int) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    total_loss = total_correct = total_targets = 0
    chunk_frames = int(cfg.get("sequence_chunk_frames", 4096))
    for session in tqdm(sessions, desc="Validation", unit="session", leave=False, dynamic_ncols=True):
        stream = make_stream(session, table, vocabulary, cfg, rng)
        if not len(stream.target_ids):
            continue
        hidden = None
        for start, end in chunk_slices(len(stream.noisy_frames), chunk_frames):
            frames = torch.from_numpy(stream.noisy_frames[start:end]).unsqueeze(0).to(device).contiguous()
            logits, hidden = model(frames, hidden)
            target_mask = (stream.target_frames >= start) & (stream.target_frames < end)
            if not target_mask.any():
                continue
            local_frames = torch.from_numpy(stream.target_frames[target_mask] - start).to(device)
            targets = torch.from_numpy(stream.target_ids[target_mask]).to(device)
            selected = logits[0, local_frames]
            total_loss += float(F.cross_entropy(selected, targets, reduction="sum"))
            total_correct += int((selected.argmax(-1) == targets).sum())
            total_targets += len(targets)
    if not total_targets:
        raise ValueError("Validation split contains no next-phoneme targets")
    return total_loss / total_targets, total_correct / total_targets


def train(config: dict) -> dict:
    seed = int(config["seed"])
    set_seeds(seed)
    device = resolve_device(str(config.get("device", "auto")))
    if device.type == "cuda":
        print(f"Training device: CUDA ({torch.cuda.get_device_name(device)})", flush=True)
    else:
        print("Training device: CPU", flush=True)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    print("Loading IPA feature table...", flush=True)
    table = FeatureTable.from_json(config["feature_table_path"])
    print("Loading and grouping IPA-CHILDES by session...", flush=True)
    sessions = load_ipa_childes(config["dataset_path"], progress=True)
    train_sessions, val_sessions = split_sessions(sessions, float(config["validation_fraction"]), seed)
    print(
        f"Prepared {len(sessions):,} sessions: {len(train_sessions):,} train, "
        f"{len(val_sessions):,} validation.",
        flush=True,
    )
    symbols = sorted({p for s in sessions for p in _phonemes(s, table)})
    vocabulary = {p: i for i, p in enumerate(symbols)}
    (output_dir / "phoneme_vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table.save(output_dir / "ipa_feature_mapping.json")
    split_manifest = {
        "train": [[s.corpus_id, s.session_id] for s in train_sessions],
        "validation": [[s.corpus_id, s.session_id] for s in val_sessions],
    }
    (output_dir / "session_split.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    model = CausalPhonemeGRU(table.width, int(config["hidden_size"]), int(config["num_layers"]), len(vocabulary), float(config["dropout"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    counters, train_rng = Counters(), np.random.default_rng(seed)
    target_checkpoint_count = int(config["target_checkpoint_count"])
    if target_checkpoint_count <= 0:
        raise ValueError("target_checkpoint_count must be positive")
    estimated_total_frames = estimate_input_frames(train_sessions, table)
    checkpoint_interval_frames = max(1, math.ceil(estimated_total_frames / target_checkpoint_count))
    print(
        f"Planned exposure: {estimated_total_frames:,} frames "
        f"({estimated_total_frames * FRAME_MS / 3_600_000:.3f} h); "
        f"checkpoint/eval approximately every {checkpoint_interval_frames:,} frames.",
        flush=True,
    )
    next_checkpoint_frame = checkpoint_interval_frames
    last_checkpoint_frame = -1
    last_metrics = None

    def checkpoint() -> dict:
        nonlocal last_metrics, last_checkpoint_frame
        val_loss, val_accuracy = validate(model, val_sessions, table, vocabulary, config, seed + 1)
        metadata = {
            **asdict(counters),
            "equivalent_input_duration_hours": counters.cumulative_frames_seen * FRAME_MS / 3_600_000,
            "validation_next_phoneme_loss": val_loss,
            "validation_next_phoneme_accuracy": val_accuracy,
            "planned_checkpoint_interval_frames": checkpoint_interval_frames,
        }
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata, "config": config, "vocabulary": vocabulary}, output_dir / f"checkpoint_step_{counters.optimizer_step:08d}.pt")
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata) + "\n")
        last_metrics = metadata
        last_checkpoint_frame = counters.cumulative_frames_seen
        tqdm.write(
            f"Checkpoint step={counters.optimizer_step:,} "
            f"frames={counters.cumulative_frames_seen:,} "
            f"hours={metadata['equivalent_input_duration_hours']:.3f} "
            f"val_loss={val_loss:.6f} val_accuracy={val_accuracy:.4f}"
        )
        return metadata

    model.train()
    chunk_frames = int(config.get("sequence_chunk_frames", 4096))
    progress = tqdm(train_sessions, desc="Training", unit="session", dynamic_ncols=True)
    for session in progress:  # developmental age order, one continuous pass
        stream = make_stream(session, table, vocabulary, config, train_rng)
        if not len(stream.target_ids):
            continue
        hidden = None
        utterance_end_frames = {
            max(span.end_frame for span in stream.spans if span.utterance_index == utterance_index)
            for utterance_index in range(len(session.utterances))
        }
        for start, end in chunk_slices(len(stream.noisy_frames), chunk_frames):
            frames = torch.from_numpy(stream.noisy_frames[start:end]).unsqueeze(0).to(device).contiguous()
            logits, hidden = model(frames, hidden)
            target_mask = (stream.target_frames >= start) & (stream.target_frames < end)
            latest_loss = None
            if target_mask.any():
                local_frames = torch.from_numpy(stream.target_frames[target_mask] - start).to(device)
                targets = torch.from_numpy(stream.target_ids[target_mask]).to(device)
                selected = logits[0, local_frames]
                loss = F.cross_entropy(selected, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
                optimizer.step()
                counters.optimizer_step += 1
                latest_loss = float(loss.detach())
            hidden = hidden.detach()
            counters.cumulative_frames_seen += end - start
            counters.cumulative_phonemes_seen += sum(start < span.end_frame <= end for span in stream.spans)
            counters.cumulative_utterances_seen += sum(start < utterance_end <= end for utterance_end in utterance_end_frames)
            postfix = {
                "step": counters.optimizer_step,
                "frames": counters.cumulative_frames_seen,
                "hours": f"{counters.cumulative_frames_seen * FRAME_MS / 3_600_000:.3f}",
            }
            if latest_loss is not None:
                postfix["loss"] = f"{latest_loss:.4f}"
            progress.set_postfix(**postfix)
            if counters.cumulative_frames_seen >= next_checkpoint_frame:
                checkpoint()
                while next_checkpoint_frame <= counters.cumulative_frames_seen:
                    next_checkpoint_frame += checkpoint_interval_frames
                model.train()
    if counters.optimizer_step == 0:
        raise ValueError("Training split contains no next-phoneme targets")
    if last_checkpoint_frame != counters.cumulative_frames_seen:
        checkpoint()
    return last_metrics or {}
