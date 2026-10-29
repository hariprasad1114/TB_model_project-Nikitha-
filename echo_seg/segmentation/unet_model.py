"""Compact U-Net for echocardiogram LV segmentation.

A small, CPU-friendly convolutional baseline.  It complements the ViT-based
`ViTSegmentation` (seg_model.py): the ViT needs pretrained encoder weights,
while this U-Net can be trained from scratch on the synthetic frames shipped
with the repo so the dashboard demo produces real segmentation output.

Checkpoints are plain dicts: {"model": state_dict, "base": int, "img_size": int,
"normalize": bool}.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """4-level U-Net (down 3x, bottleneck, up 3x)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 1, base: int = 16):
        super().__init__()
        b = base
        self.enc1 = DoubleConv(in_ch, b)          # H
        self.enc2 = DoubleConv(b, b * 2)          # H/2
        self.enc3 = DoubleConv(b * 2, b * 4)      # H/4
        self.enc4 = DoubleConv(b * 4, b * 8)      # H/8
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(b * 2, b)
        self.head = nn.Conv2d(b, out_ch, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.head(d1)
        if out.shape[-2:] != x.shape[-2:]:  # non-power-of-two inputs
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear",
                                align_corners=False)
        return out


def dice_loss(logits, target, eps: float = 1e-6):
    prob = torch.sigmoid(logits)
    num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + eps
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1.0 - (num / den).mean()


@torch.no_grad()
def dice_coef(logits, target, threshold: float = 0.5, eps: float = 1e-6):
    pred = (torch.sigmoid(logits) > threshold).float()
    num = 2.0 * (pred * target).sum(dim=(1, 2, 3)) + eps
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (num / den).mean().item()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return (self.bce_weight * self.bce(logits, target)
                + self.dice_weight * dice_loss(logits, target))
