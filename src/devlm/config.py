from __future__ import annotations

import tomllib
from pathlib import Path


REQUIRED = {
    "dataset_path", "feature_table_path", "pause_durations_path",
    "output_dir", "seed", "validation_fraction", "noise_sigma", "hidden_size", "num_layers", "dropout",
    "learning_rate", "gradient_clip_norm", "checkpoint_every_steps",
}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)["phase1"]
    missing = REQUIRED - config.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(sorted(missing))}")
    for key in ("dataset_path", "feature_table_path", "pause_durations_path", "output_dir"):
        candidate = Path(config[key])
        config[key] = str(candidate if candidate.is_absolute() else (path.parent / candidate).resolve())
    return config
