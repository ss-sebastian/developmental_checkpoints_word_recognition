from __future__ import annotations

import tomllib
from pathlib import Path


def load_construction_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)["stimulus_construction"]
    required = {"source_root", "output_root", "seed"}
    if missing := required - config.keys():
        raise ValueError(f"Missing stimulus-construction config keys: {sorted(missing)}")
    if int(config["seed"]) != 1729:
        raise ValueError("The fixed construction seed is 1729; seed search is forbidden")
    for key in ("source_root", "output_root"):
        value = Path(str(config[key]))
        config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return config
