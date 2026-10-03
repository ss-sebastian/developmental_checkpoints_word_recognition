from __future__ import annotations

import argparse

from .segmentation_probe import SegmentationProbeOptions, run_segmentation_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the minimal Phase 1 word-segmentation probe")
    parser.add_argument("--source-csv", required=True, help="Raw IPA-CHILDES CSV retaining WORD_BOUNDARY")
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing 30 Phase 1 checkpoints")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--session-split")
    parser.add_argument("--feature-table")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-transitions", type=int, default=120_000)
    parser.add_argument("--max-validation-transitions", type=int, default=30_000)
    parser.add_argument("--max-test-transitions", type=int, default=30_000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    options = SegmentationProbeOptions(
        device=args.device,
        max_train_transitions=args.max_train_transitions,
        max_validation_transitions=args.max_validation_transitions,
        max_test_transitions=args.max_test_transitions,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    run_segmentation_probe(
        args.source_csv, args.checkpoint_root, args.output_root,
        session_split_path=args.session_split,
        feature_table_path=args.feature_table,
        options=options,
    )


if __name__ == "__main__":
    main()
