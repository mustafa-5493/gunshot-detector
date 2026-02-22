import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool block."""
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class GunshotCNN(nn.Module):
    """
    Lightweight CNN for binary gunshot detection from mel-spectrograms.
    Designed to fit comfortably within 4GB VRAM on NVIDIA GTX 1050.

    Input:  (batch, 1, 64, 173)  — (channels, mel_bins, time_frames)
    Output: (batch, 2)           — logits for [non-gunshot, gunshot]
    """
    def __init__(self):
        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),   # -> (32, 32, 86)
            ConvBlock(32, 64, pool=True),   # -> (64, 16, 43)
            ConvBlock(64, 128, pool=True),  # -> (128, 8, 21)
            ConvBlock(128, 128, pool=False), # -> (128, 8, 21)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # -> (128, 4, 4)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        # AdaptiveAvgPool2d with non-divisible sizes unsupported on MPS yet.
        # Temporarily move to CPU for this op, then return to original device.
        device = x.device
        if device.type == "mps":
            x = self.adaptive_pool(x.cpu()).to(device)
        else:
            x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GunshotCNN().to(DEVICE)
    dummy = torch.randn(8, 1, 64, 173).to(DEVICE)
    out   = model(dummy)
    print(f"Output shape:     {out.shape}")
    print(f"Parameters:       {count_parameters(model):,}")
    print(f"Device:           {next(model.parameters()).device}")
