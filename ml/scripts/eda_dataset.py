"""
Simple EDA for the hand-gesture dataset:
  - class balance (bar chart)
  - 3–5 engineered features with histograms by class
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml.dataset import read_index_csv
from airmouse.paths import DATASETS_DIR


def engineered_features(vec63: np.ndarray) -> dict[str, float]:
    # Be tolerant in EDA: accept both (63,) and (21,3) layouts.
    arr = np.asarray(vec63)
    if arr.shape == (21, 3):
        arr = arr.reshape(63)
    else:
        arr = arr.reshape(-1)

    if arr.size != 63:
        raise ValueError(f"Expected 63 floats, got shape={getattr(vec63, 'shape', None)} size={arr.size}")

    pts = arr.reshape(21, 3)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    # 5 interpretable numeric features
    wrist = pts[0]
    d = np.linalg.norm(pts - wrist, axis=1)

    return {
        "x_mean": float(x.mean()),
        "y_mean": float(y.mean()),
        "z_mean": float(z.mean()),
        "dist_mean": float(d.mean()),
        "dist_max": float(d.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="EDA for dataset (balance + feature histograms).")
    p.add_argument("--data-root", type=Path, default=DATASETS_DIR)
    p.add_argument("--index-csv", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=0, help="0 = use all")
    p.add_argument("--out-dir", type=Path, default=Path("logs/dataset/eda"))
    args = p.parse_args()

    index_csv = args.index_csv or (args.data_root / "index.csv")
    if not index_csv.exists():
        raise FileNotFoundError(f"index.csv not found: {index_csv}. Run scripts/split_dataset.py first.")

    rows = read_index_csv(index_csv, dataset_root=args.data_root)
    if args.max_samples and args.max_samples > 0:
        rows = rows[: args.max_samples]

    # Balance
    counts = Counter([r.class_id for r in rows])
    classes = sorted(counts.keys())
    values = [counts[c] for c in classes]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.bar([str(c) for c in classes], values)
    plt.title("Class balance (count per class_id)")
    plt.xlabel("class_id")
    plt.ylabel("count")
    plt.tight_layout()
    balance_path = args.out_dir / "class_balance.png"
    plt.savefig(balance_path, dpi=150)
    plt.close()
    print(f"Saved: {balance_path}")

    # Features by class
    by_class: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    bad_samples: list[tuple[int, str, str]] = []
    for r in rows:
        vec = np.load(str(r.path))
        try:
            feats = engineered_features(vec)
        except Exception as e:
            bad_samples.append((r.class_id, str(r.path), str(e)))
            continue

        for k, v in feats.items():
            by_class[r.class_id][k].append(v)

    if bad_samples:
        print(f"Skipped {len(bad_samples)} invalid samples (showing up to 20):")
        for class_id, path, err in bad_samples[:20]:
            print(f"  class_id={class_id} path={path} err={err}")

    feat_names = ["x_mean", "y_mean", "z_mean", "dist_mean", "dist_max"]
    for feat in feat_names:
        plt.figure(figsize=(8, 4.5))
        for c in classes:
            vals = by_class[c][feat]
            if not vals:
                continue
            plt.hist(vals, bins=30, alpha=0.5, label=f"class={c}", density=True)
        plt.title(f"Histogram: {feat}")
        plt.xlabel(feat)
        plt.ylabel("density")
        plt.legend()
        plt.tight_layout()
        out_path = args.out_dir / f"hist_{feat}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

