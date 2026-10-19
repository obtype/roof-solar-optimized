"""Command line entry point: ``python -m roofsolar --config configs/default.yaml``."""
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import load_config, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="3D roof reconstruction and solar panel placement")
    parser.add_argument("--config", default="configs/default.yaml", help="path to YAML config")
    parser.add_argument("--data-dir", default=None, help="override data directory")
    parser.add_argument("--out-dir", default=None, help="override output directory")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    metrics = run(cfg)
    print("\n=== summary ===")
    print(f"panels: {metrics['panels']['count']}  ({metrics['panels']['kwp']} kWp)")
    print(f"facets used: {len(metrics['facets'])}")
    print(f"obstructions: {metrics['obstructions']}")
    print(f"outputs written to: {Path(cfg['out_dir']).resolve()}")


if __name__ == "__main__":  # pragma: no cover
    main()
