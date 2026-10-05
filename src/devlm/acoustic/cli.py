from __future__ import annotations

import argparse
import json

from .config import load_config
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the independent causal log-Mel acoustic developmental LM")
    parser.add_argument("--config", required=True, help="TOML configuration with [acoustic_phase1]")
    args = parser.parse_args()
    print(json.dumps(train(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
