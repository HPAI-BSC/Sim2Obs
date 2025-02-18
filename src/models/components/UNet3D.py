import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv3d(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.residual_to_3d = ResidualTo3D()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2 = self.residual_to_3d(x2)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class BottleneckTo3D(nn.Module):
    """
    Convert (bs, c, 8, 8) to (bs, c, 8, 8, 8) so that 3D convolutions work.
    """
    def __init__(self, ):
        super(BottleneckTo3D, self).__init__()
        self.up_to_3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        pass

    def forward(self, x):
        resolution = x.shape[-1]
        x = x.view(x.shape[0], x.shape[1], resolution // 2, resolution // 2, resolution // 2)
        x = self.up_to_3d(x)
        return x


class ResidualTo3D(nn.Module):
    """
    Convert (bs, c, h, h) to (bs, c, h, h, h) so that 3D convolutions work.
    It works as follows:
        1. Find a,b such that a*b = h/8
        2. (bs, c, h, h) to (bs, c, a*h, b*h)
        3. (bs, c, a*h, b*h) to (bs, c, h/2, h/2, h/2)
        4. (bs, c, h/2, h/2, h/2) to (bs, c, h, h, h)
    """
    def __init__(self, ):
        super(ResidualTo3D, self).__init__()
        self.up_to_3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        pass

    def _balanced_factors(self, x):
        for a in range(math.isqrt(abs(int(x))), 0, -1):  # Start from the square root and go down
            if x % a == 0:  # Check if it's a factor
                b = x // a
                return (int(a), int(b)) if a > 0 else (-int(a), -int(b))  # Ensure consistent sign

    def forward(self, x):
        # Step 0: Find a,b such that a*b = h/8
        resolution = x.shape[-1]
        if resolution % 8 != 0:
            # Resolution is not divisible by 8!!!
            raise ValueError(f"Resolution ({resolution}) is not divisible by 8!")
        else:
            a, b = self._balanced_factors(resolution / 8)

        # Step 1: (bs, c, h, h) to (bs, c, a*h, b*h)
        x = F.interpolate(x, size=(a * resolution, b * resolution), mode='bilinear', align_corners=False)
        # Step 2: (bs, c, a*h, b*h) to (bs, c, h/2, h/2, h/2)
        x = x.view(x.shape[0], x.shape[1], resolution // 2, resolution // 2, resolution // 2)
        # Step 3: (bs, c, h/2, h/2, h/2) to (bs, c, h, h, h)
        x = self.up_to_3d(x)
        return x


class UNet3D(nn.Module):
    def __init__(
            self,
            n_channels: int = 1,
            n_classes: int = 1,
            bilinear: bool = False
    ) -> None:
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.bottleneck = BottleneckTo3D()
        self.up1 = Up3D(1024, 512 // factor, bilinear)
        self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear)
        self.up4 = Up3D(128, 64, bilinear)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.bottleneck(x5)
        x = self.up1(x6, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    _ = UNet3D()