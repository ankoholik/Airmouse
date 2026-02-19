"""
Validate saved .npy samples under the dataset root.

Writes a CSV of invalid samples (bad shape / empty / unreadable) so they can be deleted
or regenerated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from ml.dataset import _infer_class_id_from_path, to_vec63
from airmouse.paths import DATASETS_DIR


def main() -> None:
    p = argparse.ArgumentParser(description="Validate dataset .npy samples (expect 63 floats).")
    p.add_argument("--data-root", type=Path, default=DATASETS_DIR)
    p.add_argument("--out-csv", type=Path, default=Path("logs/dataset/bad_samples.csv"))
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = p.parse_args()

    root = args.data_root
    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    bad: list[tuple[str, int | None, str]] = []
    checked = 0

    for path in root.rglob("*.npy"):
        checked += 1
        class_id = _infer_class_id_from_path(path, root)
        try:
            arr = np.load(str(path), allow_pickle=False)
            vec = to_vec63(arr)
            if vec is None:
                bad.append((str(path), class_id, f"invalid shape={getattr(arr,'shape',None)} size={getattr(arr,'size',None)}"))
        except Exception as e:
            bad.append((str(path), class_id, f"load_error: {type(e).__name__}: {e}"))

        if args.limit and checked >= args.limit:
            break

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_id", "reason"])
        w.writerows(bad)

    print(f"Checked: {checked}")
    print(f"Bad: {len(bad)}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

