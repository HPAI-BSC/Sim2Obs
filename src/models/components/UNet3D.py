import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        :param in_channels: Input channel count.
        :param out_channels: Output channel count.
        :param mid_channels: Channel count between the two convs
            (defaults to `out_channels`).
        """
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
        """Apply the double-conv block to `x`."""
        return self.double_conv(x)


class DoubleConv3D(nn.Module):
    """(3D convolution => [BN3D] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        :param in_channels: Input channel count.
        :param out_channels: Output channel count.
        :param mid_channels: Channel count between the two convs
            (defaults to `out_channels`).
        """
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
        """Apply the 3D double-conv block to `x`."""
        return self.double_conv3d(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: Input channel count.
        :param out_channels: Output channel count.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Halve the spatial resolution (2x2 maxpool) then apply a double-conv block."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """2D upscaling (bilinear or transposed-conv) then double conv, with the corresponding encoder skip connection concatenated in."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        :param in_channels: Input channel count (of the tensor being upsampled).
        :param out_channels: Output channel count.
        :param bilinear: If True, upsample via bilinear interpolation
            (cheaper); if False, use a learned `ConvTranspose2d`.
        """
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Upsample `x1`, pad it to match `x2`'s spatial size, concatenate with the `x2` skip connection, and apply the double-conv block.

        :param x1: Decoder feature map to be upsampled.
        :param x2: Corresponding encoder skip-connection feature map.
        :return: The convolved, concatenated feature map.
        """
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
    """3D upscaling (bilinear or transposed-conv) then double conv, with the corresponding 2D encoder skip connection lifted to 3D (via `ResidualTo3D`) and concatenated in."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        :param in_channels: Input channel count (of the tensor being upsampled).
        :param out_channels: Output channel count.
        :param bilinear: If True, upsample via trilinear interpolation
            (cheaper); if False, use a learned `ConvTranspose3d`.
        """
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
        """
        Upsample `x1` (3D), lift `x2` to 3D via `ResidualTo3D`, pad to match spatial size, concatenate, and apply the 3D double-conv block.

        :param x1: Decoder feature map (already 3D) to be upsampled.
        :param x2: Corresponding 2D encoder skip-connection feature map.
        :return: The convolved, concatenated 3D feature map.
        """
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
    """Final 1x1 2D convolution mapping the decoder's channels to the output class/channel count."""
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: Input channel count.
        :param out_channels: Output channel count.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Apply the 1x1 output convolution to `x`."""
        return self.conv(x)


class OutConv3D(nn.Module):
    """Final 1x1 3D convolution mapping the decoder's channels to the output class/channel count."""
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: Input channel count.
        :param out_channels: Output channel count.
        """
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Apply the 1x1 3D output convolution to `x`."""
        return self.conv(x)


class BottleneckTo3D(nn.Module):
    """
    Convert (bs, c, 8, 8) to (bs, c, 8, 8, 8) so that 3D convolutions work.
    """
    def __init__(self, ):
        """Build the trilinear upsampler used to restore the resolution halved by the reshape in `forward`."""
        super(BottleneckTo3D, self).__init__()
        self.up_to_3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        pass

    def forward(self, x):
        """
        Reshape a `(B, C, H, W)` bottleneck tensor into a 3D volume by halving spatial resolution and folding it into a depth axis, then upsample back.

        :param x: Bottleneck tensor, shape `(B, C, res, res)`.
        :return: 3D tensor, shape `(B, C, res, res, res)`.
        """
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
        """Build the trilinear upsampler used in the final reshaping step."""
        super(ResidualTo3D, self).__init__()
        self.up_to_3d = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        pass

    def _balanced_factors(self, x):
        """Find integers `(a, b)` with `a * b == x` and `a` as close as possible to `sqrt(x)`, for reshaping `x` spatial positions into a roughly-square `(a*res, b*res)` grid."""
        for a in range(math.isqrt(abs(int(x))), 0, -1):  # Start from the square root and go down
            if x % a == 0:  # Check if it's a factor
                b = x // a
                return (int(a), int(b)) if a > 0 else (-int(a), -int(b))  # Ensure consistent sign

    def forward(self, x):
        """
        Reshape a `(B, C, H, H)` 2D feature map into a `(B, C, H, H, H)` 3D volume (see class docstring for the 4-step method).

        :param x: 2D feature map, shape `(B, C, res, res)`, with `res`
            divisible by 8.
        :return: 3D volume, shape `(B, C, res, res, res)`.

        :raises ValueError: If `res` is not divisible by 8.
        """
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
    """
    Hybrid 2D-encoder / 3D-decoder U-Net: a standard 2D U-Net encoder,
    whose bottleneck is lifted to a 3D volume (`BottleneckTo3D`) and
    decoded with 3D convolutions (`Up3D`), with each 2D encoder skip
    connection lifted to 3D (`ResidualTo3D`) before being concatenated in.
    """
    def __init__(
            self,
            n_channels: int = 1,
            n_classes: int = 1,
            bilinear: bool = False
    ) -> None:
        """
        :param n_channels: Number of input image channels.
        :param n_classes: Number of output channels/classes.
        :param bilinear: If True, decoder upsampling uses bilinear/trilinear
            interpolation (and the bottleneck/decoder channel counts are
            halved accordingly); if False, uses transposed convolutions.
        """
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
        """
        Run the 2D encoder, lift the bottleneck to 3D, then the 3D skip-connected decoder and output projection.

        :param x: Input tensor, shape `(B, n_channels, H, W)`.
        :return: Logits, shape `(B, n_classes, D, H', W')` (3D volume).
        """
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