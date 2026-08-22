from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FeatureTable:
    def __init__(self, feature_names: list[str], mapping: dict[str, list[float]]):
        if not feature_names or not mapping:
            raise ValueError("Feature table cannot be empty")
        width = len(feature_names)
        if any(len(v) != width for v in mapping.values()):
            raise ValueError("All feature vectors must have the declared width")
        self.feature_names = tuple(feature_names)
        self.mapping = {p: np.asarray(v, dtype=np.float32) for p, v in mapping.items()}
        self._symbols = sorted(self.mapping, key=len, reverse=True)

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureTable":
        with Path(path).open(encoding="utf-8") as handle:
            obj = json.load(handle)
        return cls(obj["feature_names"], obj["phonemes"])

    @classmethod
    def from_panphon(cls, phonemes: list[str], save_to: str | Path | None = None) -> "FeatureTable":
        try:
            import panphon  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PanPhon is not installed; provide feature_table_path or install the optional 'panphon' dependency") from exc
        ft = panphon.FeatureTable()
        names = list(ft.names)
        mapping = {}
        for phoneme in sorted(set(phonemes)):
            vectors = ft.word_to_vector_list(phoneme, numeric=True)
            if not vectors:
                raise ValueError(f"PanPhon cannot represent IPA phoneme token {phoneme!r}")
            # IPA-CHILDES treats some diphthongs/affricates as one phoneme token.
            # Their component segment vectors are averaged into one continuous vector.
            mapping[phoneme] = np.asarray(vectors, dtype=np.float32).mean(axis=0).astype(float).tolist()
        table = cls(names, mapping)
        if save_to:
            table.save(save_to)
        return table

    def save(self, path: str | Path) -> None:
        payload = {"feature_names": list(self.feature_names), "phonemes": {k: v.tolist() for k, v in self.mapping.items()}}
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @property
    def width(self) -> int:
        return len(self.feature_names)

    def vector(self, phoneme: str) -> np.ndarray:
        try:
            return self.mapping[phoneme]
        except KeyError as exc:
            raise ValueError(f"No articulatory feature vector for IPA phoneme {phoneme!r}") from exc

    def tokenize(self, ipa: str) -> tuple[str, ...]:
        """Longest-match segmentation; whitespace and common word delimiters vanish."""
        out, i = [], 0
        while i < len(ipa):
            if ipa[i].isspace() or ipa[i] in "_|#":
                i += 1
                continue
            symbol = next((s for s in self._symbols if ipa.startswith(s, i)), None)
            if symbol is None:
                raise ValueError(f"Cannot segment IPA at {ipa[i:]!r}; use pre-segmented phonemes or extend the feature table")
            out.append(symbol)
            i += len(symbol)
        return tuple(out)
