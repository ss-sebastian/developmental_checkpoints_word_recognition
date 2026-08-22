from __future__ import annotations

import argparse
import json

from .config import load_config
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 1 developmental next-phoneme GRU")
    parser.add_argument("--config", required=True, help="TOML configuration path")
    args = parser.parse_args()
    metrics = train(load_config(args.config))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
