"""MusicCNN model used as the subject model for XAI research."""

import torch
import torch.nn as nn


class MusicCNN(nn.Module):
    """
    3-block CNN for mel spectrogram classification.
    Designed for interpretability: no residual connections,
    explicit spatial feature maps accessible for XAI.
    """

    def __init__(self, n_classes: int = 10, dropout: float = 0.4):
        super().__init__()

        def conv_block(in_ch, out_ch, pool=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(pool),
                nn.Dropout2d(p=0.15),
            )

        self.block1 = conv_block(1, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flat = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = self.flat(x)
        return self.head(x)

    def get_activations(self, x):
        """Return intermediate feature maps for XAI."""
        acts = {}
        x = self.block1(x)
        acts["block1"] = x.detach()
        x = self.block2(x)
        acts["block2"] = x.detach()
        x = self.block3(x)
        acts["block3"] = x.detach()
        return acts
