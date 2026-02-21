"""
Visualize random samples from the hand-gesture dataset.

Each sample is a 63D vector (21 landmarks × 3 coordinates).
We render a 2D skeleton (x,y) for quick inspection.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ml.dataset import read_index_csv
from airmouse.paths import DATASETS_DIR


# MediaPipe Hands connections (subset sufficient for visualization)
MP_HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def vec63_to_xy(vec: np.ndarray) -> np.ndarray:
    pts = vec.reshape(21, 3)
    return pts[:, :2]


def draw_sample(ax, vec: np.ndarray, title: str) -> None:
    xy = vec63_to_xy(vec)
    ax.scatter(xy[:, 0], xy[:, 1], s=12)
    for a, b in MP_HAND_EDGES:
        ax.plot([xy[a, 0], xy[b, 0]], [xy[a, 1], xy[b, 1]], linewidth=1)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize random dataset samples.")
    p.add_argument("--data-root", type=Path, default=DATASETS_DIR)
    p.add_argument("--index-csv", type=Path, default=None)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("logs/dataset/samples.png"))
    args = p.parse_args()

    random.seed(args.seed)
    index_csv = args.index_csv or (args.data_root / "index.csv")
    if not index_csv.exists():
        raise FileNotFoundError(f"index.csv not found: {index_csv}. Run scripts/split_dataset.py first.")

    rows = read_index_csv(index_csv, dataset_root=args.data_root)
    if not rows:
        raise SystemExit("No rows in index.")

    chosen = random.sample(rows, k=min(args.n, len(rows)))

    cols = 4
    rows_n = int(np.ceil(len(chosen) / cols))
    plt.figure(figsize=(cols * 3.2, rows_n * 3.2))

    for i, r in enumerate(chosen, start=1):
        vec = np.load(str(r.path))
        ax = plt.subplot(rows_n, cols, i)
        draw_sample(ax, vec, title=f"class={r.class_id}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()

