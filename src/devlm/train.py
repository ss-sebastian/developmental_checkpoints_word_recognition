from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Session, load_ipa_childes, split_sessions
from .features import FeatureTable
from .model import CausalPhonemeGRU
from .stream import FrameStream, build_session_stream
from .timing import EmpiricalPauseSampler, FRAME_MS


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
    torch.use_deterministic_algorithms(True)


def _phonemes(session: Session, table: FeatureTable) -> tuple[str, ...]:
    return tuple(p for u in session.utterances for p in (u.phonemes or table.tokenize(u.ipa)))


def make_stream(session: Session, table: FeatureTable, vocabulary: dict[str, int], cfg: dict, rng: np.random.Generator) -> FrameStream:
    pauses = EmpiricalPauseSampler.from_json(cfg["pause_durations_path"], rng)
    return build_session_stream(
        session, table, vocabulary, pauses, rng,
        noise_sigma=float(cfg["noise_sigma"]),
        phoneme_envelope=cfg.get("phoneme_envelope"),
    )


@torch.no_grad()
def validate(model: CausalPhonemeGRU, sessions: list[Session], table: FeatureTable, vocabulary: dict[str, int], cfg: dict, seed: int) -> tuple[float, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    total_loss = total_correct = total_targets = 0
    for session in sessions:
        stream = make_stream(session, table, vocabulary, cfg, rng)
        if not len(stream.target_ids):
            continue
        frames = torch.from_numpy(stream.noisy_frames).unsqueeze(0)
        logits, _ = model(frames)
        selected = logits[0, torch.from_numpy(stream.target_frames)]
        targets = torch.from_numpy(stream.target_ids)
        total_loss += float(F.cross_entropy(selected, targets, reduction="sum"))
        total_correct += int((selected.argmax(-1) == targets).sum())
        total_targets += len(targets)
    if not total_targets:
        raise ValueError("Validation split contains no next-phoneme targets")
    return total_loss / total_targets, total_correct / total_targets


def train(config: dict) -> dict:
    seed = int(config["seed"])
    set_seeds(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.jsonl").write_text("", encoding="utf-8")
    table = FeatureTable.from_json(config["feature_table_path"])
    sessions = load_ipa_childes(config["dataset_path"])
    train_sessions, val_sessions = split_sessions(sessions, float(config["validation_fraction"]), seed)
    symbols = sorted({p for s in sessions for p in _phonemes(s, table)})
    vocabulary = {p: i for i, p in enumerate(symbols)}
    (output_dir / "phoneme_vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table.save(output_dir / "ipa_feature_mapping.json")
    split_manifest = {
        "train": [[s.corpus_id, s.session_id] for s in train_sessions],
        "validation": [[s.corpus_id, s.session_id] for s in val_sessions],
    }
    (output_dir / "session_split.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    model = CausalPhonemeGRU(table.width, int(config["hidden_size"]), int(config["num_layers"]), len(vocabulary), float(config["dropout"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    counters, train_rng = Counters(), np.random.default_rng(seed)
    checkpoint_every = int(config["checkpoint_every_steps"])
    last_metrics = None

    def checkpoint() -> dict:
        nonlocal last_metrics
        val_loss, val_accuracy = validate(model, val_sessions, table, vocabulary, config, seed + 1)
        metadata = {
            **asdict(counters),
            "equivalent_input_duration_hours": counters.cumulative_frames_seen * FRAME_MS / 3_600_000,
            "validation_next_phoneme_loss": val_loss,
            "validation_next_phoneme_accuracy": val_accuracy,
        }
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata, "config": config, "vocabulary": vocabulary}, output_dir / f"checkpoint_step_{counters.optimizer_step:08d}.pt")
        with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metadata) + "\n")
        last_metrics = metadata
        return metadata

    model.train()
    for session in train_sessions:  # developmental age order, one continuous pass
        stream = make_stream(session, table, vocabulary, config, train_rng)
        if not len(stream.target_ids):
            continue
        logits, _ = model(torch.from_numpy(stream.noisy_frames).unsqueeze(0))
        selected = logits[0, torch.from_numpy(stream.target_frames)]
        loss = F.cross_entropy(selected, torch.from_numpy(stream.target_ids))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
        counters.optimizer_step += 1
        counters.cumulative_frames_seen += len(stream.noisy_frames)
        counters.cumulative_phonemes_seen += len(stream.spans)
        counters.cumulative_utterances_seen += len(session.utterances)
        if counters.optimizer_step % checkpoint_every == 0:
            checkpoint()
        model.train()
    if counters.optimizer_step == 0:
        raise ValueError("Training split contains no next-phoneme targets")
    if counters.optimizer_step % checkpoint_every:
        checkpoint()
    return last_metrics or {}
