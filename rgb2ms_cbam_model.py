"""RGB-to-multispectral CBAM model extracted from RGB_to_MS_Model.ipynb."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(),
            nn.Linear(ch // r, ch, bias=False)
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.fc(self.avg(x).view(b, c))
        m = self.fc(self.max(x).view(b, c))
        return x * self.sig((a + m).view(b, c, 1, 1))

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.cbam = CBAM(ch)
        self.act  = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.cbam(self.block(x)) + x)

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class RGB2MS_UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=4, base=64):
        super().__init__()
        b = base
        self.enc1 = EncoderBlock(in_ch, b)
        self.enc2 = EncoderBlock(b,     b * 2)
        self.enc3 = EncoderBlock(b * 2, b * 4)
        self.enc4 = EncoderBlock(b * 4, b * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(b * 8, b * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(b * 16),
            nn.LeakyReLU(0.1, inplace=True),
            ResBlock(b * 16),
            ResBlock(b * 16),
        )
        self.dec4 = DecoderBlock(b * 16, b * 8, b * 8)
        self.dec3 = DecoderBlock(b * 8,  b * 4, b * 4)
        self.dec2 = DecoderBlock(b * 4,  b * 2, b * 2)
        self.dec1 = DecoderBlock(b * 2,  b,     b)
        self.head = nn.Sequential(
            nn.Conv2d(b, b // 2, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(b // 2, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        b      = self.bottleneck(x4)
        d4     = self.dec4(b,  s4)
        d3     = self.dec3(d4, s3)
        d2     = self.dec2(d3, s2)
        d1     = self.dec1(d2, s1)
        return self.head(d1)

