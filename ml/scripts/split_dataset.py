"""
Create a structured dataset split (train/test) with stratification.

Outputs:
  - data/index.csv
  - data/splits/train.csv
  - data/splits/test.csv

CSV schema: path,class_id (paths relative to data/).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

from ml.dataset import build_index_csv, read_index_csv
from airmouse.paths import DATASETS_DIR


def write_rows(rows, out_csv: Path, dataset_root: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_id"])
        for r in rows:
            w.writerow([str(r.path.relative_to(dataset_root)).replace("\\", "/"), r.class_id])


def main() -> None:
    p = argparse.ArgumentParser(description="Build dataset index and stratified train/test split.")
    p.add_argument("--data-root", type=Path, default=DATASETS_DIR)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    index_csv = build_index_csv(dataset_root=args.data_root)
    rows = read_index_csv(index_csv, dataset_root=args.data_root)
    if not rows:
        raise SystemExit(f"No samples found under {args.data_root}")

    y = [r.class_id for r in rows]
    train_rows, test_rows = train_test_split(
        rows,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    splits_dir = args.data_root / "splits"
    train_csv = splits_dir / "train.csv"
    test_csv = splits_dir / "test.csv"
    write_rows(train_rows, train_csv, dataset_root=args.data_root)
    write_rows(test_rows, test_csv, dataset_root=args.data_root)

    print(f"Index: {index_csv}")
    print(f"Train: {train_csv} ({len(train_rows)})")
    print(f"Test : {test_csv} ({len(test_rows)})")


if __name__ == "__main__":
    main()

