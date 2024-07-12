import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.in_proj = nn.Linear(d_model, d_model * expand)
        self.conv1d = nn.Conv1d(d_model * expand, d_model * expand, kernel_size=d_conv, padding=d_conv - 1,
                                groups=d_model * expand)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_model * expand, d_model)

        # Initialize A, B, C, D matrices for selective SSM
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model * expand))
        self.C = nn.Parameter(torch.randn(d_model * expand, d_state))
        self.D = nn.Parameter(torch.randn(d_model * expand))

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        x = self.in_proj(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        # Selective SSM
        u = x
        delta = F.softplus(u)
        A_bar = torch.diag_embed(torch.exp(-delta.unsqueeze(-1) * torch.exp(self.A)))
        B_bar = torch.exp(self.B) * delta.unsqueeze(-1)
        x = torch.einsum('bsd,dn->bsn', u, B_bar)
        x = torch.einsum('bsn,bsm->bnm', x, A_bar)
        x = torch.einsum('bnm,md->bsd', x, self.C)
        x = x + u * self.D

        x = self.act(x)
        x = self.out_proj(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_state, d_conv, expand):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.mamba = MambaBlock(out_channels, d_state, d_conv, expand)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.mamba(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x, self.pool(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_state, d_conv, expand):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.mamba = MambaBlock(out_channels, d_state, d_conv, expand)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.mamba(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x



class MambaUNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        for feature in features:
            self.downs.append(DownBlock(in_channels, feature, d_state=16, d_conv=4, expand=2))
            in_channels = feature

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(UpBlock(feature*2, feature, d_state=16, d_conv=4, expand=2))

        self.bottleneck = DownBlock(features[-1], features[-1]*2, d_state=16, d_conv=4, expand=2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x, pooled = down(x)
            skip_connections.append(x)
            x = pooled

        x, _ = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.ups)):
            x = self.ups[idx](x, skip_connections[idx])

        return self.final_conv(x)
