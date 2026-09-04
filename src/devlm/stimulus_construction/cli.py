from __future__ import annotations

import argparse

from .config import load_construction_config
from .construct import construct


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct model-blind task-adaptation stimuli; performs no training")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = construct(load_construction_config(args.config))
    for task, rows in result["final"].items():
        print(f"{task}: {len(result['candidates'][task]):,} candidates -> {len(rows):,} final items", flush=True)


if __name__ == "__main__":
    main()
