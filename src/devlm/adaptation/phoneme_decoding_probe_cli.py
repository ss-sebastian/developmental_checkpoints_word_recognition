from __future__ import annotations

import argparse

from .phoneme_decoding_probe import PhonemeDecodingProbeOptions, run_phoneme_decoding_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode phoneme identity from frozen Phase 1 peak-frame states")
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--session-split")
    parser.add_argument("--feature-table")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-tokens-per-phoneme", type=int, default=20)
    parser.add_argument("--max-tokens-per-phoneme", type=int, default=500)
    args = parser.parse_args()
    run_phoneme_decoding_probe(args.source_csv, args.checkpoint_root, args.output_root, args.session_split, args.feature_table,
        PhonemeDecodingProbeOptions(device=args.device, min_tokens_per_phoneme=args.min_tokens_per_phoneme, max_tokens_per_phoneme=args.max_tokens_per_phoneme))


if __name__ == "__main__":
    main()
