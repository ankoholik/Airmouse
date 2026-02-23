"""
Run multiple training experiments and log results to a CSV.

This is intended to satisfy coursework requirements like:
  - >30 experiments for fast models (<5 min each), OR
  - total >30 hours for long models (>5 min each).

Example:
  python scripts/run_experiments.py --runs 30 --epochs 80 --lrs 0.001 0.0005 0.0002
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ml.train import train


def iter_lrs(lrs: list[float], runs: int) -> Iterable[float]:
    if not lrs:
        for _ in range(runs):
            yield 0.0005
        return
    for i in range(runs):
        yield lrs[i % len(lrs)]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    p = argparse.ArgumentParser(description="Run multiple training experiments and log results.")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42, help="Base seed (will be incremented per run)")
    p.add_argument("--lrs", type=float, nargs="*", default=[0.001, 0.0005, 0.0002])
    p.add_argument("--out-dir", type=Path, default=Path("logs/experiments"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.out_dir / f"experiments_{stamp}.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_id",
                "seed",
                "epochs",
                "batch_size",
                "lr",
                "device",
                "wall_time_s",
                "test_loss",
                "test_accuracy",
            ]
        )

        for run_id, lr in enumerate(iter_lrs(args.lrs, args.runs), start=1):
            seed = args.seed + run_id - 1
            set_seed(seed)

            run_log_dir = args.out_dir / f"run_{run_id:03d}"
            t0 = time.time()
            result = train(
                num_epochs=args.epochs,
                learning_rate=lr,
                batch_size=args.batch_size,
                log_dir=run_log_dir,
                device=args.device,
            )
            dt = time.time() - t0

            w.writerow(
                [
                    run_id,
                    seed,
                    args.epochs,
                    args.batch_size,
                    lr,
                    args.device,
                    f"{dt:.2f}",
                    f"{result['final']['loss']:.6f}",
                    f"{result['final']['accuracy']:.6f}",
                ]
            )
            f.flush()
            print(
                f"[{run_id}/{args.runs}] lr={lr} seed={seed} "
                f"acc={result['final']['accuracy']:.4f} time={dt/60:.2f} min"
            )

    print(f"Saved experiments log: {out_csv}")


if __name__ == "__main__":
    main()

