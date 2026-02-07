from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from airmouse.paths import DATASETS_DIR


@dataclass(frozen=True)
class DatasetRow:
    path: Path
    class_id: int  # 1..K (as stored on disk)


def _iter_npy_files(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.npy")


def _infer_class_id_from_path(p: Path, root: Path) -> int | None:
    """
    Supports:
      - legacy: data/<class_id>/<file>.npy
      - structured: data/raw/<class_id>/<shard>/<file>.npy
    """
    rel = p.relative_to(root)
    parts = rel.parts
    if not parts:
        return None

    if parts[0] == "raw" and len(parts) >= 2:
        maybe = parts[1]
    else:
        maybe = parts[0]

    try:
        return int(maybe)
    except Exception:
        return None


def build_index_csv(
    dataset_root: Path = DATASETS_DIR,
    out_csv: Path | None = None,
) -> Path:
    """
    Build an index.csv with columns: path,class_id.
    Paths are stored relative to dataset_root for portability.
    """
    if out_csv is None:
        out_csv = dataset_root / "index.csv"

    rows: list[DatasetRow] = []
    for p in _iter_npy_files(dataset_root):
        class_id = _infer_class_id_from_path(p, dataset_root)
        if class_id is None:
            continue
        rows.append(DatasetRow(path=p, class_id=class_id))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "class_id"])
        for r in rows:
            w.writerow([str(r.path.relative_to(dataset_root)).replace("\\", "/"), r.class_id])

    return out_csv


def read_index_csv(index_csv: Path, dataset_root: Path = DATASETS_DIR) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with index_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = dataset_root / Path(row["path"])
            rows.append(DatasetRow(path=p, class_id=int(row["class_id"])))
    return rows


def to_vec63(arr: np.ndarray) -> np.ndarray | None:
    """
    Convert loaded .npy content into a flat (63,) float vector.
    Returns None if the sample is invalid/corrupted.
    """
    try:
        a = np.asarray(arr)
        if a.shape == (21, 3):
            a = a.reshape(63)
        else:
            a = a.reshape(-1)

        if a.size != 63:
            return None

        # Ensure float32 for torch friendliness
        return a.astype(np.float32, copy=False)
    except Exception:
        return None


class HandGestureDataset(Dataset):
    """
    Object: one captured hand pose (.npy)
    Features: 63 floats (21 landmarks × 3 coordinates)
    Target: gesture class_id (1..K on disk), mapped to 0..K-1 for training.
    """

    def __init__(self, rows: list[DatasetRow], strict: bool = False):
        self.rows = rows
        self.strict = strict
        # Stable label mapping: class_id 1..K -> 0..K-1
        class_ids = sorted({r.class_id for r in rows})
        self.class_to_idx = {cid: i for i, cid in enumerate(class_ids)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        data = np.load(str(r.path), allow_pickle=False)
        vec = to_vec63(data)
        if vec is None:
            if self.strict:
                raise ValueError(f"Invalid sample (expected 63 floats), got shape={getattr(data,'shape',None)} at {r.path}")
            return None

        x = torch.from_numpy(vec)
        y = torch.tensor(self.class_to_idx[r.class_id], dtype=torch.long)
        return x, y


def collate_drop_invalid(batch):
    """
    Collate function that drops invalid samples (dataset returns None).
    Returns None if the whole batch is invalid.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def create_dataloaders_from_split_csv(
    train_csv: Path,
    test_csv: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    dataset_root: Path = DATASETS_DIR,
) -> tuple[DataLoader, DataLoader, dict[int, int]]:
    """
    Returns train_loader, test_loader and class_to_idx mapping (class_id -> idx).
    """
    train_rows = read_index_csv(train_csv, dataset_root=dataset_root)
    test_rows = read_index_csv(test_csv, dataset_root=dataset_root)

    train_ds = HandGestureDataset(train_rows, strict=False)
    test_ds = HandGestureDataset(test_rows, strict=False)

    # Ensure same mapping in both splits
    test_ds.class_to_idx = train_ds.class_to_idx

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_drop_invalid,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_drop_invalid,
    )
    return train_loader, test_loader, train_ds.class_to_idx
