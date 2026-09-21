import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Upscaling (bilinear or transposed-conv) then double conv, with the corresponding encoder skip connection concatenated in."""

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


class OutConv(nn.Module):
    """Final 1x1 convolution mapping the decoder's channels to the output class/channel count."""
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


class UNet(nn.Module):
    """
    Standard 2D U-Net (milesial/Pytorch-UNet-style): a 5-level encoder
    (DoubleConv + Down blocks, doubling channels each level) and a
    4-level decoder (Up blocks with skip connections), ending in a 1x1
    OutConv.
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
        :param bilinear: If True, decoder upsampling uses bilinear
            interpolation (and the bottleneck/decoder channel counts are
            halved accordingly); if False, uses transposed convolutions.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Run the encoder, then the skip-connected decoder, then the output projection.

        :param x: Input tensor, shape `(B, n_channels, H, W)`.
        :return: Logits, shape `(B, n_classes, H, W)`.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    _ = UNet()