"""
Phase 4: Shared U-Net architecture used by Dehaze / Derain / Deblur.

A compact 3-level U-Net (not the full 4-5 level version) -- kept small on
purpose since we're training on a few hundred image pairs, not millions;
a bigger network would just overfit faster without more data to justify it.

3c change: Residual learning -- the network predicts the *correction* to add
back to the degraded input, not the full clean image. This is the same design
used in DnCNN and FFDNet: the residual is nearly all-zero for clean regions,
so the optimizer starts from a much better point and converges to sharper
outputs faster than learning the full pixel grid from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """3-down / 3-up U-Net. Input/output: 3-channel image in [0, 1] range."""

    def __init__(self, in_ch=3, out_ch=3, base=32, residual=False):
        super().__init__()
        self.residual = residual
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.up1 = Up(base * 8, base * 4)
        self.up2 = Up(base * 4, base * 2)
        self.up3 = Up(base * 2, base)
        if residual:
            self.outc_raw = nn.Conv2d(base, out_ch, kernel_size=1)
        else:
            self.outc = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, x):
        inp = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        if self.residual:
            residual = self.outc_raw(x)
            return torch.clamp(inp + residual, 0.0, 1.0)
        out = self.outc(x)
        return torch.sigmoid(out)