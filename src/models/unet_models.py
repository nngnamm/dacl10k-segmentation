import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class AttentionGate(nn.Module):
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)

class UNetArchitecture(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 19, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        f = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool2d(2, 2)

        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])
        self.bottleneck = DoubleConv(f[3], f[4])

        self.up4 = nn.ConvTranspose2d(f[4], f[3], 2, 2)
        self.att4 = AttentionGate(f[3], f[3], f[2]) if use_attention else nn.Identity()
        self.dec4 = DoubleConv(f[4], f[3])

        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.att3 = AttentionGate(f[2], f[2], f[1]) if use_attention else nn.Identity()
        self.dec3 = DoubleConv(f[3], f[2])

        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.att2 = AttentionGate(f[1], f[1], f[0]) if use_attention else nn.Identity()
        self.dec2 = DoubleConv(f[2], f[1])

        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.att1 = AttentionGate(f[0], f[0], f[0] // 2) if use_attention else nn.Identity()
        self.dec1 = DoubleConv(f[1], f[0])

        self.out_conv = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        s4 = self.att4(d4, e4) if self.use_attention else e4
        d4 = self.dec4(torch.cat([s4, d4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(d3, e3) if self.use_attention else e3
        d3 = self.dec3(torch.cat([s3, d3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(d2, e2) if self.use_attention else e2
        d2 = self.dec2(torch.cat([s2, d2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(d1, e1) if self.use_attention else e1
        d1 = self.dec1(torch.cat([s1, d1], dim=1))

        return self.out_conv(d1)