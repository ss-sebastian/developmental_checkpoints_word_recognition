from __future__ import annotations

import tomllib
from pathlib import Path

from .features import _mel_filterbank


REQUIRED = {
    "audio_manifest_path", "output_dir", "seed", "device",
    "validation_fraction", "sample_rate", "n_mels", "n_fft", "win_length", "hop_length",
    "hidden_size", "num_layers", "dropout", "learning_rate",
    "gradient_clip_norm", "future_horizons_frames", "sequence_chunk_frames",
    "max_train_hours", "max_validation_hours", "normalization_max_hours", "target_checkpoint_count",
}


def load_config(path: str | Path) -> dict:
    """Load only the independent ``[acoustic_phase1]`` TOML section."""
    path = Path(path).resolve()
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    if "acoustic_phase1" not in document:
        raise ValueError("Config must contain an [acoustic_phase1] table")
    config = dict(document["acoustic_phase1"])
    missing = REQUIRED - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(sorted(missing))}")
    for key in ("audio_manifest_path", "output_dir"):
        candidate = Path(config[key])
        config[key] = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
    if any(float(config[key]) <= 0 for key in ("max_train_hours", "max_validation_hours", "normalization_max_hours")):
        raise ValueError("max_train_hours, max_validation_hours and normalization_max_hours must be positive")
    horizons = [int(value) for value in config["future_horizons_frames"]]
    if not horizons or any(value <= 1 for value in horizons):
        raise ValueError("future_horizons_frames must contain positive horizons greater than 1; t+1-only training is disallowed")
    config["future_horizons_frames"] = sorted(set(horizons))
    if int(config["hop_length"]) * 1000 != int(config["sample_rate"]) * 10:
        raise ValueError("hop_length must represent exactly 10 ms at sample_rate")
    if int(config["win_length"]) * 1000 != int(config["sample_rate"]) * 25:
        raise ValueError("win_length must represent exactly a 25-ms analysis window at sample_rate")
    if int(config["n_fft"]) < int(config["win_length"]):
        raise ValueError("n_fft must be at least win_length")
    # Fail before an hours-long run if the requested FFT/Mel geometry would
    # silently create all-zero Mel dimensions.
    _mel_filterbank(int(config["sample_rate"]), int(config["n_fft"]), int(config["n_mels"]))
    return config
