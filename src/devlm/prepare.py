from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .features import FeatureTable


def normalize_huggingface_record(row: dict) -> dict | None:
    """Convert one IPA-CHILDES EnglishNA row to the Phase 1 loader schema."""
    age = row.get("target_child_age")
    if age is None or not math.isfinite(float(age)):
        return None
    required_ids = (row.get("corpus_id"), row.get("transcript_id"), row.get("id"))
    if any(value is None for value in required_ids):
        return None
    phonemes = tuple(
        token for token in str(row.get("ipa_transcription") or "").split()
        if token and token != "WORD_BOUNDARY"
    )
    if not phonemes:
        return None
    text = str(row.get("processed_gloss") or row.get("gloss") or "").strip()
    return {
        "corpus_id": str(row["corpus_id"]),
        "session_id": str(row["transcript_id"]),
        "target_child_age_months": float(age),
        "utterance_order": int(row["id"]),
        "ipa": " ".join(phonemes),
        "text": text,
        "phonemes": list(phonemes),
        "language": "English",
        "dialect": "North American English",
    }


def download_and_prepare(output_dataset: str | Path, output_features: str | Path) -> dict:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install the 'colab' optional dependency to download IPA-CHILDES") from exc

    output_dataset, output_features = Path(output_dataset), Path(output_features)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    output_features.parent.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(
        "phonemetransformers/IPA-CHILDES", "EnglishNA", split="train", streaming=True
    )
    inventory: set[str] = set()
    kept = skipped = 0
    with output_dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            normalized = normalize_huggingface_record(row)
            if normalized is None:
                skipped += 1
                continue
            handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
            inventory.update(normalized["phonemes"])
            kept += 1
            if kept % 100_000 == 0:
                print(f"Prepared {kept:,} utterances...", flush=True)
    if not kept:
        raise RuntimeError("No usable EnglishNA IPA-CHILDES utterances were downloaded")
    FeatureTable.from_panphon(sorted(inventory), save_to=output_features)
    return {"utterances": kept, "skipped": skipped, "phoneme_types": len(inventory)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare IPA-CHILDES EnglishNA")
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-features", required=True)
    args = parser.parse_args()
    print(json.dumps(download_and_prepare(args.output_dataset, args.output_features), indent=2))


if __name__ == "__main__":
    main()
