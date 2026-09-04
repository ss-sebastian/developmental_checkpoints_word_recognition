from __future__ import annotations

import tomllib
from pathlib import Path

from .schema import TASK_NAMES


REQUIRED = {
    "task_name", "manifest_path", "split_path", "hidden_dim", "learning_rate",
    "weight_decay", "optimizer_family", "batch_size", "max_epochs", "patience",
    "min_delta", "early_stopping_metric", "seed",
}
DEFAULT_ADAPTATION_SIZES = (25, 50, 100, 200, 400)
DEFAULT_INITIALIZATION_SEEDS = (1729, 2718, 3141)


def load_adaptation_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)["task_adaptation"]
    missing = REQUIRED - config.keys()
    if missing:
        raise ValueError(f"Missing task-adaptation configuration keys: {', '.join(sorted(missing))}")
    if config["task_name"] not in TASK_NAMES:
        raise ValueError(f"Unsupported task_name {config['task_name']!r}")
    if config["early_stopping_metric"] != "validation_loss":
        raise ValueError("early_stopping_metric must be validation_loss")
    for key in ("manifest_path", "split_path"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    config.setdefault("adaptation_sizes", list(DEFAULT_ADAPTATION_SIZES))
    config.setdefault("initialization_seeds", list(DEFAULT_INITIALIZATION_SEEDS))
    config.setdefault("train_fraction", 0.70)
    config.setdefault("validation_fraction", 0.15)
    config.setdefault("test_fraction", 0.15)
    if tuple(config["adaptation_sizes"]) != DEFAULT_ADAPTATION_SIZES:
        raise ValueError(f"adaptation_sizes must be {list(DEFAULT_ADAPTATION_SIZES)} in this infrastructure version")
    initialization_seeds = tuple(int(value) for value in config["initialization_seeds"])
    if not initialization_seeds or len(initialization_seeds) != len(set(initialization_seeds)):
        raise ValueError("initialization_seeds must be a non-empty list of unique integers")
    if any(value < 0 or value >= 10_000_000_000 for value in initialization_seeds):
        raise ValueError("Every initialization seed must be between 0 and 9,999,999,999")
    config["initialization_seeds"] = list(initialization_seeds)
    if int(config["hidden_dim"]) < 1 or int(config["batch_size"]) < 1:
        raise ValueError("hidden_dim and batch_size must be positive")
    if int(config["max_epochs"]) < 1 or int(config["patience"]) < 1:
        raise ValueError("max_epochs and patience must be positive")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    fractions = tuple(float(config[key]) for key in ("train_fraction", "validation_fraction", "test_fraction"))
    if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must be positive and sum to 1")
    return config
