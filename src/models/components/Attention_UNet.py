import torch
import torch.nn as nn

class Attention_block(nn.Module):
    """Attention gate: reweights skip-connection features `x` by a sigmoid attention map computed from the gating signal `g` and `x` (Attention U-Net, Oktay et al. 2018)."""
    def __init__(self, F_g, F_l, F_int):
        """
        Build the gating (`W_g`), skip-connection (`W_x`), and attention-coefficient (`psi`) projections.

        :param F_g: Number of channels in the gating signal `g`.
        :param F_l: Number of channels in the skip-connection input `x`.
        :param F_int: Number of intermediate channels the gate/skip
            projections are reduced to before combining.
        """
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Compute the attention map from `g`/`x` and apply it to `x`.

        :param g: Gating signal (typically the decoder's upsampled features).
        :param x: Skip-connection features to be gated.
        :return: `x` scaled elementwise by the learned attention
            coefficients (shape unchanged).
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class conv_block(nn.Module):
    """Two 3x3 conv + BatchNorm + ReLU layers in sequence (a standard U-Net double-conv block)."""
    def __init__(self,ch_in,ch_out):
        """
        :param ch_in: Input channel count.
        :param ch_out: Output channel count (used for both conv layers).
        """
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        """Apply the double-conv block to `x`."""
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """2x nearest-neighbour upsample followed by a 3x3 conv + BatchNorm + ReLU."""
    def __init__(self,ch_in,ch_out):
        """
        :param ch_in: Input channel count.
        :param ch_out: Output channel count.
        """
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )
    def forward(self,x):
        """Upsample and convolve `x`."""
        x = self.up(x)
        return x

class Attention_UNet(nn.Module):
    """
    Attention U-Net: a standard U-Net encoder/decoder where each skip
    connection is passed through an `Attention_block`, gated by the
    decoder's upsampled features, before being concatenated.
    """
    def __init__(self, img_ch=1, output_ch=1):
        """
        Build the 5-level encoder, 4-level attention-gated decoder, and final 1x1 output conv.

        :param img_ch: Number of input image channels.
        :param output_ch: Number of output channels.
        """
        super(Attention_UNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Run the encoder, attention-gated decoder, and final projection.

        :param x: Input tensor, shape `(B, img_ch, H, W)`.
        :return: Output tensor, shape `(B, output_ch, H, W)`.
        """
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1