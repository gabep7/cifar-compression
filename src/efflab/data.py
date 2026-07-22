"""CIFAR-10 dataloaders."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _maybe_skip_integrity(skip: bool) -> None:
    """Optionally bypass torchvision MD5 checks for local smoke datasets."""
    if not skip:
        return
    import torchvision.datasets.utils as utils
    import torchvision.datasets.cifar as cifar_mod

    def _ok(*_args, **_kwargs) -> bool:
        return True

    utils.check_integrity = _ok  # type: ignore[assignment]
    cifar_mod.check_integrity = _ok  # type: ignore[attr-defined]


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders for CIFAR-10."""
    data_cfg = cfg.get("data", cfg)
    name = data_cfg.get("name", "cifar10")
    if name != "cifar10":
        raise ValueError(f"Only cifar10 supported in v1, got {name!r}")

    root = data_cfg.get("root", "data")
    batch_size = int(data_cfg.get("batch_size", 128))
    num_workers = int(data_cfg.get("num_workers", 2))
    download = bool(data_cfg.get("download", True))
    skip_integrity = bool(data_cfg.get("skip_integrity_check", False))
    _maybe_skip_integrity(skip_integrity)

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_set = datasets.CIFAR10(
        root=root, train=True, transform=train_tf, download=download
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, transform=test_tf, download=download
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, test_loader
