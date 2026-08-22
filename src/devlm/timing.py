from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FRAME_MS = 10


class EmpiricalPauseSampler:
    """Samples observed utterance-pause durations in milliseconds."""
    def __init__(self, values_ms: dict[str, list[float]], rng: np.random.Generator):
        cleaned = {k: np.asarray(v, dtype=float) for k, v in values_ms.items()}
        if not cleaned or any(len(v) == 0 or np.any(v <= 0) for v in cleaned.values()):
            raise ValueError("Empirical timing lists must be non-empty and strictly positive")
        self.values_ms, self.rng = cleaned, rng

    @classmethod
    def from_json(cls, path: str | Path, rng: np.random.Generator) -> "EmpiricalPauseSampler":
        with Path(path).open(encoding="utf-8") as handle:
            obj = json.load(handle)
        if obj.get("synthetic_fixture") and not obj.get("allow_for_tests_only", False):
            raise ValueError("Synthetic timing data are not authorized for training")
        return cls(obj["values_ms"], rng)

    def sample_frames(self, label: str = "__default__") -> int:
        candidates = self.values_ms.get(label, self.values_ms.get("__default__"))
        if candidates is None:
            raise ValueError(f"No empirical timing values for {label!r} and no __default__ fallback")
        milliseconds = float(self.rng.choice(candidates))
        return max(1, int(round(milliseconds / FRAME_MS)))

    def expected_frames(self, label: str = "__default__") -> float:
        candidates = self.values_ms.get(label, self.values_ms.get("__default__"))
        if candidates is None:
            raise ValueError(f"No empirical timing values for {label!r} and no __default__ fallback")
        frame_values = np.maximum(1, np.rint(candidates / FRAME_MS))
        return float(frame_values.mean())
