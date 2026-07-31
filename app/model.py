import torch
import torch.nn as nn

class EfficientUNetGenerator(nn.Module):
    def __init__(self, base_channels=40):
        super().__init__()
        bc = base_channels
        
        # Encoder
        self.enc1 = self._down_block(3, bc)
        self.enc2 = self._down_block(bc, bc*2)
        self.enc3 = self._down_block(bc*2, bc*4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bc*4, bc*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(bc*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(bc*4, bc*4, 3, padding=1, bias=False),
            nn.BatchNorm2d(bc*4),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(bc*4, bc*2, 4, 2, 1)
        self.dec3 = self._up_block(bc*4, bc*2)
        
        self.up2 = nn.ConvTranspose2d(bc*2, bc, 4, 2, 1)
        self.dec2 = self._up_block(bc*2, bc)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(bc, bc//2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bc//2, 1, 3, padding=1),
            nn.Tanh()
        )

    def _down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        b = self.bottleneck(e3)
        
        d3 = self.up3(b)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        return out



class PatchGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)
        )
    def forward(self, x):
        return self.model(x)

   