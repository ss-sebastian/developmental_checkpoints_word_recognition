from __future__ import annotations

import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def load_wav_mono(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load PCM WAV as mono float32 in [-1, 1].  No labels are involved."""
    path = Path(path)
    with wave.open(str(path), "rb") as reader:
        channels, sample_width, sample_rate, frames = reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()
        compressed = reader.getcomptype() != "NONE"
        payload = reader.readframes(frames)
    if compressed:
        raise ValueError(f"Compressed WAV is unsupported: {path}")
    if channels < 1 or sample_width not in {1, 2, 3, 4}:
        raise ValueError(f"Unsupported WAV PCM format in {path}: channels={channels}, sample_width={sample_width}")
    if sample_width == 1:
        values = (np.frombuffer(payload, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 3)
        signed = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        values = signed.astype(np.float32) / 8388608.0
    else:
        values = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    waveform = torch.from_numpy(values.reshape(-1, channels).mean(axis=1).copy())
    return waveform, sample_rate


def log_mel_frame_count(path: str | Path, *, target_sample_rate: int, n_fft: int, hop_length: int, win_length: int) -> int:
    """Return the exact frame count implied by WAV metadata and extractor rules."""
    with wave.open(str(path), "rb") as reader:
        source_rate, samples = reader.getframerate(), reader.getnframes()
    resampled_samples = samples if source_rate == target_sample_rate else max(1, round(samples * target_sample_rate / source_rate))
    if win_length > n_fft:
        raise ValueError("win_length must not exceed n_fft")
    # The 800-point DFT is zero-padding *within* every explicit 25-ms frame;
    # it must never change the frame support or duration accounting.
    return max(0, 1 + (resampled_samples - win_length) // hop_length) if resampled_samples >= win_length else 0


def resample_linear(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    target_length = max(1, round(len(waveform) * target_rate / source_rate))
    return F.interpolate(waveform[None, None], size=target_length, mode="linear", align_corners=False)[0, 0]


def _hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


@lru_cache(maxsize=16)
def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    hz_points = _mel_to_hz(torch.linspace(float(_hz_to_mel(torch.tensor(0.0))), float(_hz_to_mel(torch.tensor(sample_rate / 2))), n_mels + 2))
    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long().clamp(0, n_freqs - 1)
    bank = torch.zeros(n_mels, n_freqs)
    for index in range(n_mels):
        left, center, right = (int(bins[index]), int(bins[index + 1]), int(bins[index + 2]))
        if center > left:
            bank[index, left:center] = torch.arange(left, center, dtype=torch.float32).sub(left).div(center - left)
        if right > center:
            bank[index, center:right] = torch.arange(center, right, dtype=torch.float32).sub(right).neg().div(right - center)
    if torch.any(bank.sum(dim=1) <= 0):
        empty = torch.nonzero(bank.sum(dim=1) <= 0).flatten().tolist()
        raise ValueError(f"Mel filterbank has empty bins {empty}; increase n_fft while retaining the requested win_length")
    return bank


def log_mel_spectrogram(
    waveform: torch.Tensor, sample_rate: int, *, target_sample_rate: int = 16_000,
    n_mels: int = 80, n_fft: int = 800, win_length: int = 400, hop_length: int = 160,
) -> torch.Tensor:
    """Return [frames, n_mels] natural-log power Mel features at a 10-ms hop.

    Each explicit 25-ms Hann-windowed frame is zero-padded to an 800-point DFT;
    this does not extend its temporal support or change its 10-ms hop. Model
    state t predicts only later frames. Real recording silence stays in the
    signal; no pauses are added. The longer DFT grid is intentional: with 80
    filters it avoids empty Mel bands while retaining a 25-ms analysis window.
    """
    if hop_length * 1000 != target_sample_rate * 10:
        raise ValueError("log-Mel hop must be exactly 10 ms")
    waveform = resample_linear(waveform.float(), sample_rate, target_sample_rate)
    if win_length > n_fft:
        raise ValueError("win_length must not exceed n_fft")
    if len(waveform) < win_length:
        return torch.empty(0, n_mels, dtype=torch.float32)
    frames = waveform.unfold(0, win_length, hop_length)
    window = torch.hann_window(win_length, dtype=waveform.dtype, device=waveform.device)
    spectrum = torch.fft.rfft(frames * window, n=n_fft, dim=-1).abs().square()
    mel = _mel_filterbank(target_sample_rate, n_fft, n_mels).to(spectrum)
    return torch.log(torch.clamp(spectrum @ mel.transpose(0, 1), min=1e-10)).contiguous()
