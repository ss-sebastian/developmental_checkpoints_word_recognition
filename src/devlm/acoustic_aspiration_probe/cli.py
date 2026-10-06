from __future__ import annotations

import argparse

from .run import run_replication


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate Martin et al. aspiration probes on frozen acoustic checkpoints")
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mald-cache", required=True)
    parser.add_argument("--max-items", type=int, help="explicit smoke-only per-task cap; formal default is all MALD matches")
    args = parser.parse_args()
    run_replication(args.checkpoints, args.output, args.mald_cache, max_items=args.max_items)


if __name__ == "__main__":
    main()
