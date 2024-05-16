import torch.nn as nn
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt

#ToDo: Add image size

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class TransEncDec(nn.Module):
    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            n_heads: int = 4,
            n_layers: int = 6,
            dim_feedforward: int = 2048,
            patch_size: int = 16,
            dropout: float = 0.1):
        super(TransEncDec, self).__init__()

        input_resolution = 128
        scale = dim_feedforward ** -0.5

        # Encoder
        self.encoder = TransformerEncoder(in_channels, n_heads, n_layers, dim_feedforward, dropout)

        # Decoder
        #self.decoder = TransformerDecoder(out_channels, n_heads, n_layers, dim_feedforward, dropout)
        #self.decoder = CNNDecoder(dim_feedforward, out_channels)
        self.decoder = ViTDecoder(dim_feedforward, patch_size=patch_size, embed_dim=768, num_heads=n_heads, num_layers=n_layers, dropout=dropout)

        # Input and output projections
        self.input_proj = nn.Conv2d(in_channels, dim_feedforward, kernel_size=patch_size, stride=patch_size, bias=False)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2, dim_feedforward))
        self.ln_pre = LayerNorm(dim_feedforward)

        self.transformer = Transformer(dim_feedforward, n_layers, n_heads)

        self.ln_post = LayerNorm(dim_feedforward)

        self.output_proj = nn.Conv2d(dim_feedforward, out_channels, kernel_size=1)



    def forward(self, x):
        # Encode input image
        print(self.n_heads, self.n_layers)
        x = self.input_proj(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        #x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])

        #enc_out = self.encoder(x)

        # Decode latent representation
        dec_out = self.decoder(x)

        # Project output
        #out = self.output_proj(dec_out)

        return dec_out


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels, n_heads, n_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()

        # Convolutional embedding
        # self.embedding = nn.Conv2d(in_channels, dim_feedforward, kernel_size=1)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(dim_feedforward, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self, x):
        # Convolutional embedding
        # x = self.embedding(x)

        # Transformer encoder
        out = self.transformer_encoder(x.flatten(2).permute(2, 0, 1))

        return out.permute(1, 2, 0).view(x.shape)



class CNNDecoder(nn.Module):
    def __init__(self, input_dim, output_channels=1):
        super(CNNDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 16 * 16 * 256),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x)
        x = x.view(batch_size, 256, 16, 16)
        x = self.decoder(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, out_channels, n_heads, n_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.dim_feedforward = dim_feedforward

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(dim_feedforward, n_heads, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers)

    def forward(self, x):
        tgt = torch.zeros_like(x, device=x.device)
        print(tgt.shape)

        # Transformer decoder
        out = self.transformer_decoder(tgt, memory=x)
        print(out.shape)

        return out.permute(1, 2, 0).view(x.shape)


import torch
import torch.nn as nn


class ViTDecoder(nn.Module):
    def __init__(self, n_features, patch_size=16, embed_dim=768, num_heads=12, num_layers=12, dropout=0.1):
        super(ViTDecoder, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (128 // patch_size) ** 2

        self.embedding = nn.Linear(n_features, embed_dim)
        #self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        decoder_layers = nn.TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward=1024, dropout=dropout,
                                                    batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers)

        self.output = nn.Linear(embed_dim, 128 ** 2)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), 1, -1)

        #pos_emb = self.pos_embedding[:, :x.size(1) + 1]
        #x = x + pos_emb

        x = self.dropout(x)

        x = self.decoder(x, x)
        x = self.output(x)

        x = x.view(x.size(0), 1, 128, 128)
        return x

"""
import torch
import torch.nn as nn
from einops import rearrange

class ViTDecoder(nn.Module):
    def __init__(self, input_dim, output_channels=1, patch_size=16, num_heads=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(ViTDecoder, self).__init__()
        self.input_dim = input_dim
        self.output_channels = output_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.encoder = nn.Linear(input_dim, 1024)

        self.patch_embedding = nn.Conv2d(1, 1024 // patch_size ** 2, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, (128 // patch_size) ** 2 + 1, 1024))

        decoder_layer = nn.TransformerDecoderLayer(d_model=1024, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Conv2d(1024, output_channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        encoded = self.encoder(x).unsqueeze(1)
        patches = self.patch_embedding(encoded.permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        positional_encoding = self.positional_encoding[:, :patches.shape[1]].expand(batch_size, -1, -1)
        output = self.transformer_decoder(tgt=patches + positional_encoding, memory=encoded)
        output = rearrange(output, 'b (h w) c -> b c h w', h=128 // self.patch_size, w=128 // self.patch_size)
        output = self.output_projection(output)
        return output
"""

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
