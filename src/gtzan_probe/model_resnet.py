"""Second CNN architecture: ResNet-style model for architecture robustness check.

A lightweight ResNet with 1-channel input (grayscale mel spectrogram).
Same preprocessing and splits as MusicCNN. Used to verify that the
SHAP/LIME disagreement is not specific to one architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class MusicResNet(nn.Module):
    """Lightweight ResNet for 128×128 single-channel mel spectrograms.

    Architecturally different from MusicCNN (residual connections vs. plain
    conv blocks) but uses identical preprocessing, splits and training protocol.
    ~800K parameters to match MusicCNN scale.
    """

    def __init__(self, n_classes: int = 10, dropout: float = 0.4) -> None:
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Residual stages with downsampling between them
        self.stage1 = nn.Sequential(
            ResidualBlock(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            ResidualBlock(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            ResidualBlock(128),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flat = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.head(x)
