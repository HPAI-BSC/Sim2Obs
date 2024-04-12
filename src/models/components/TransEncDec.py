import torch.nn as nn
import torch


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
        self.decoder = TransformerDecoder(out_channels, n_heads, n_layers, dim_feedforward, dropout)

        # Input and output projections
        self.input_proj = nn.Conv2d(in_channels, dim_feedforward, kernel_size=patch_size, stride=patch_size, bias=False)
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, dim_feedforward))
        self.ln_pre = LayerNorm(dim_feedforward)

        self.transformer(dim_feedforward, n_layers, n_heads)

        self.ln_post = LayerNorm(dim_feedforward)

        self.output_proj = nn.Conv2d(dim_feedforward, out_channels, kernel_size=1)



    def forward(self, x):
        # Encode input image
        x = self.input_proj(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        print(f'After ln_pre')
        print(x.size())

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])

        print(x.size())

        #enc_out = self.encoder(x)

        # Decode latent representation
        dec_out = self.decoder(enc_out)

        # Project output
        out = self.output_proj(dec_out)

        return out


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


class TransformerDecoder(nn.Module):
    def __init__(self, out_channels, n_heads, n_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.dim_feedforward = dim_feedforward

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(dim_feedforward, n_heads, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, n_layers)

    def forward(self, x):
        batch_size, _, height, width = x.shape
        tgt = torch.zeros((height * width, batch_size, self.dim_feedforward), device=x.device)

        # Transformer decoder
        out = self.transformer_decoder(tgt, memory=x.flatten(2).permute(2, 0, 1))

        return out.permute(1, 2, 0).view(x.shape)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
