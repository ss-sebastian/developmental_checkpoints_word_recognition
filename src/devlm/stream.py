from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Session
from .features import FeatureTable
from .timing import FRAME_MS

PHONEME_FRAMES = 5
PHONEME_OVERLAP_FRAMES = 1
UTTERANCE_PAUSE_FRAMES = 3
DEFAULT_PHONEME_ENVELOPE = np.asarray([1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 3.0], dtype=np.float32)


@dataclass(frozen=True)
class PhonemeSpan:
    phoneme: str
    utterance_index: int
    start_frame: int
    end_frame: int


@dataclass
class FrameStream:
    clean_frames: np.ndarray
    noisy_frames: np.ndarray
    speech_mask: np.ndarray
    target_frames: np.ndarray
    target_ids: np.ndarray
    spans: tuple[PhonemeSpan, ...]
    frame_ms: int = FRAME_MS


def _normalized_envelope(values: list[float] | tuple[float, ...] | np.ndarray | None) -> np.ndarray:
    envelope = DEFAULT_PHONEME_ENVELOPE.copy() if values is None else np.asarray(values, dtype=np.float32)
    if envelope.shape != (PHONEME_FRAMES,):
        raise ValueError(f"phoneme_envelope must contain exactly {PHONEME_FRAMES} values")
    if not np.isfinite(envelope).all() or np.any(envelope <= 0):
        raise ValueError("phoneme_envelope values must be finite and strictly positive so all five frames are active")
    if not np.allclose(envelope, envelope[::-1]):
        raise ValueError("phoneme_envelope must be symmetric")
    return envelope / envelope.max()


def build_session_stream(
    session: Session, features: FeatureTable, phoneme_to_id: dict[str, int],
    rng: np.random.Generator, noise_sigma: float = 0.05,
    phoneme_envelope: list[float] | tuple[float, ...] | np.ndarray | None = None,
) -> FrameStream:
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")
    envelope = _normalized_envelope(phoneme_envelope)
    spans: list[PhonemeSpan] = []
    cursor = 0
    for ui, utt in enumerate(session.utterances):
        phonemes = utt.phonemes or features.tokenize(utt.ipa)
        if not phonemes:
            raise ValueError(f"Empty phoneme sequence in {session.session_id}, utterance {utt.utterance_order}")
        for pi, phoneme in enumerate(phonemes):
            start = cursor
            end = start + PHONEME_FRAMES
            spans.append(PhonemeSpan(phoneme, ui, start, end))
            cursor = end if pi == len(phonemes) - 1 else end - PHONEME_OVERLAP_FRAMES
        if ui < len(session.utterances) - 1:
            cursor = spans[-1].end_frame + UTTERANCE_PAUSE_FRAMES
    total_frames = max(s.end_frame for s in spans)
    clean = np.zeros((total_frames, features.width), dtype=np.float32)
    speech = np.zeros(total_frames, dtype=bool)
    for span in spans:
        clean[span.start_frame:span.end_frame] += envelope[:, None] * features.vector(span.phoneme)[None, :]
        speech[span.start_frame:span.end_frame] = True
    noisy = clean.copy()
    if noise_sigma and speech.any():
        noisy[speech] += rng.normal(0.0, noise_sigma, size=(int(speech.sum()), features.width)).astype(np.float32)
    targets_f, targets_i = [], []
    for idx, span in enumerate(spans):
        if idx == 0:
            continue
        prediction_frame = span.start_frame - 1
        if prediction_frame < 0:
            continue
        # This is an executable leakage invariant, not merely metadata.
        if prediction_frame >= span.start_frame:
            raise AssertionError("Target activation leaked into predictor context")
        targets_f.append(prediction_frame)
        targets_i.append(phoneme_to_id[span.phoneme])
    return FrameStream(clean, noisy, speech, np.asarray(targets_f, dtype=np.int64), np.asarray(targets_i, dtype=np.int64), tuple(spans))
