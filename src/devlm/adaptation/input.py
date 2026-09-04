from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from devlm.data import Session, Utterance
from devlm.features import FeatureTable
from devlm.stream import FrameStream, build_session_stream

from .schema import AdaptationItem


@dataclass(frozen=True)
class AdaptationBatch:
    """Padded final-state inputs; contains no behavioral covariates."""

    frames: torch.Tensor
    lengths: torch.Tensor
    labels: torch.Tensor
    item_ids: tuple[str, ...]


def build_adaptation_stream(
    item: AdaptationItem,
    features: FeatureTable,
    phoneme_to_id: dict[str, int],
    rng: np.random.Generator,
    noise_sigma: float = 0.05,
    phoneme_envelope: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> FrameStream:
    """Route adaptation IPA through the unchanged Phase 1 frame generator."""
    utterance = Utterance(
        corpus_id="synthetic-adaptation-interface",
        session_id=item.item_id,
        target_child_age_months=0.0,
        utterance_order=1,
        ipa="",
        text="",
        phonemes=item.phonemes,
    )
    session = Session("synthetic-adaptation-interface", item.item_id, 0.0, (utterance,))
    return build_session_stream(
        session, features, phoneme_to_id, rng,
        noise_sigma=noise_sigma, phoneme_envelope=phoneme_envelope,
    )


def build_adaptation_batch(
    items: list[AdaptationItem],
    features: FeatureTable,
    phoneme_to_id: dict[str, int],
    seed: int,
    noise_sigma: float = 0.05,
    phoneme_envelope: list[float] | tuple[float, ...] | np.ndarray | None = None,
    device: torch.device | str = "cpu",
) -> AdaptationBatch:
    """Build a deterministic padded batch with the unchanged Phase 1 stream code."""
    if not items:
        raise ValueError("Cannot build an empty adaptation batch")
    rng = np.random.default_rng(seed)
    streams = [
        build_adaptation_stream(
            item, features, phoneme_to_id, rng,
            noise_sigma=noise_sigma, phoneme_envelope=phoneme_envelope,
        )
        for item in items
    ]
    lengths = np.asarray([len(stream.noisy_frames) for stream in streams], dtype=np.int64)
    frames = np.zeros((len(streams), int(lengths.max()), features.width), dtype=np.float32)
    for index, stream in enumerate(streams):
        frames[index, :lengths[index]] = stream.noisy_frames
    return AdaptationBatch(
        frames=torch.from_numpy(frames).to(device),
        lengths=torch.from_numpy(lengths).to(device),
        labels=torch.tensor([item.binary_label for item in items], dtype=torch.float32, device=device),
        item_ids=tuple(item.item_id for item in items),
    )
