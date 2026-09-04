from __future__ import annotations

import argparse

from .train import TrainingOptions, train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen-checkpoint linear task readouts")
    parser.add_argument("--stimuli", required=True)
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-table")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoding-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--noise-sigma", type=float, default=0.05)
    parser.add_argument("--input-noise-seed", type=int, default=20260904)
    parser.add_argument("--initialization-seeds", type=int, nargs="+", default=[1729, 2718, 3141])
    args = parser.parse_args()
    options = TrainingOptions(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        batch_size=args.batch_size, encoding_batch_size=args.encoding_batch_size,
        max_epochs=args.max_epochs, patience=args.patience, min_delta=args.min_delta,
        noise_sigma=args.noise_sigma, input_noise_seed=args.input_noise_seed,
        initialization_seeds=tuple(args.initialization_seeds), device=args.device,
    )
    train_all(
        args.stimuli, args.checkpoints_dir, args.output_dir,
        feature_table_path=args.feature_table, options=options,
    )


if __name__ == "__main__":
    main()
