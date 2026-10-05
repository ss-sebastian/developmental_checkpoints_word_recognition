from __future__ import annotations

import argparse

from .sound_probe import DEFAULT_INITIALIZATION_SEEDS, SoundProbeOptions, prepare_sound_manifest, run_sound_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and run the Sound-only frozen-representation probe")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Create a model-blind attested Sound manifest")
    prepare.add_argument("--source", required=True, help="Existing Sound-only candidate/final manifest TSV")
    prepare.add_argument("--output", required=True, help="Dedicated primary manifest output TSV")
    prepare.add_argument("--max-word-reuse", type=int, default=40)
    prepare.add_argument("--max-component-words", type=int, default=50)

    run = commands.add_parser("run", help="Fit M00–M30 frozen Sound linear probes")
    run.add_argument("--stimuli", required=True, help="Validated Sound-only manifest TSV")
    run.add_argument("--checkpoints-dir", required=True, help="Phase 1 ZIP extraction containing M01–M30")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--feature-table")
    run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    run.add_argument("--noise-sigma", type=float, default=0.05)
    run.add_argument("--input-noise-seed", type=int, default=20261004)
    run.add_argument("--initialization-seeds", nargs="+", type=int, default=list(DEFAULT_INITIALIZATION_SEEDS))
    run.add_argument("--max-epochs", type=int, default=100)
    run.add_argument("--patience", type=int, default=10)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--bootstrap-repetitions", type=int, default=1000)
    run.add_argument("--max-word-reuse", type=int, default=40)
    run.add_argument("--max-component-words", type=int, default=50)
    args = parser.parse_args()

    if args.command == "prepare":
        report = prepare_sound_manifest(
            args.source, args.output, max_word_reuse=args.max_word_reuse,
            max_component_words=args.max_component_words,
        )
        print(report)
        return
    options = SoundProbeOptions(
        initialization_seeds=tuple(args.initialization_seeds), noise_sigma=args.noise_sigma,
        input_noise_seed=args.input_noise_seed, max_epochs=args.max_epochs,
        patience=args.patience, batch_size=args.batch_size, learning_rate=args.learning_rate,
        bootstrap_repetitions=args.bootstrap_repetitions, max_word_reuse=args.max_word_reuse,
        max_component_words=args.max_component_words, device=args.device,
    )
    run_sound_probe(
        args.stimuli, args.checkpoints_dir, args.output_dir,
        feature_table_path=args.feature_table, options=options,
    )


if __name__ == "__main__":
    main()
