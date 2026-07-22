#!/usr/bin/env python3
"""Convert HuggingFace `uoft-cs/cifar10` into the canonical CIFAR-10 pickle
format that torchvision.datasets.CIFAR10 expects in
``data/cifar-10-batches-py/``.

This bypasses the slow Toronto download host entirely.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image


def _batch_dict(batch_label: str, images: np.ndarray, labels: list[int]) -> dict:
    """Build a canonical CIFAR batch dict.

    ``images`` is (N, 32, 32, 3) uint8 -> reshape to (N, 3072) uint8.
    """
    flat = images.reshape(images.shape[0], -1)
    return {
        "batch_label": batch_label,
        "data": flat,
        "labels": labels,
        "filenames": [f"{batch_label}_{i}.png" for i in range(images.shape[0])],
    }


def main() -> None:
    out_dir = Path("data/cifar-10-batches-py")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("uoft-cs/cifar10")
    train = ds["train"]
    test = ds["test"]

    # Convert to numpy arrays (N, 32, 32, 3) uint8
    def to_arrays(split, n):
        imgs = np.zeros((n, 32, 32, 3), dtype=np.uint8)
        labels = []
        for i, row in enumerate(split):
            img = row["img"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            imgs[i] = np.asarray(img)
            labels.append(row["label"])
        return imgs, labels

    # Split train (50000) into 5 batches of 10000
    train_imgs, train_labels = to_arrays(train, len(train))
    for b in range(5):
        start, end = b * 10000, (b + 1) * 10000
        d = _batch_dict(
            f"training batch {b + 1} of 5",
            train_imgs[start:end],
            train_labels[start:end],
        )
        with open(out_dir / f"data_batch_{b + 1}", "wb") as f:
            pickle.dump(d, f)
        print(f"wrote data_batch_{b + 1}  ({end - start} samples)")

    # Test batch
    test_imgs, test_labels = to_arrays(test, len(test))
    d = _batch_dict("testing batch 1 of 1", test_imgs, test_labels)
    with open(out_dir / "test_batch", "wb") as f:
        pickle.dump(d, f)
    print(f"wrote test_batch ({len(test)} samples)")

    # batches.meta
    label_names = train.features["label"].names
    meta = {
        "num_cases_per_batch": 10000,
        "label_names": label_names,
        "num_vis": 3072,
    }
    with open(out_dir / "batches.meta", "wb") as f:
        pickle.dump(meta, f)
    print(f"wrote batches.meta (labels={label_names})")

    # Write a dummy tar.gz so torchvision sees the archive (it only checks
    # the extracted dir exists, but harmless).
    print("Done. Verifying...")
    for fn in ["data_batch_1", "data_batch_2", "data_batch_3",
               "data_batch_4", "data_batch_5", "test_batch", "batches.meta"]:
        p = out_dir / fn
        assert p.exists(), f"missing {fn}"
        with open(p, "rb") as f:
            d = pickle.load(f)
        if fn != "batches.meta":
            print(f"  {fn}: data={d['data'].shape} labels={len(d['labels'])}")
    print("All batches verified.")


if __name__ == "__main__":
    main()