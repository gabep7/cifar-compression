"""Model builders for CIFAR experiments."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(
    name: str, num_classes: int = 10, pretrained: bool = False
) -> nn.Module:
    """Build a supported model for CIFAR-10 experiments."""
    name = name.lower()
    weights = "DEFAULT" if pretrained else None

    if name == "resnet18":
        model = models.resnet18(weights=weights)
        # CIFAR stem: 3x3 stride-1, drop maxpool so 32x32 features survive.
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name in {"mobilenet_v3_small", "mobilenetv3_small", "mobilenet_v3"}:
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unknown model {name!r}. Supported: resnet18, mobilenet_v3_small"
    )
