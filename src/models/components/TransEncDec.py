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
            dropout: float = 0.1):
        super(TransEncDec, self).__init__()

        # Encoder
        self.encoder = TransformerEncoder(in_channels, n_heads, n_layers, dim_feedforward, dropout)

        # Decoder
        self.decoder = TransformerDecoder(out_channels, n_heads, n_layers, dim_feedforward, dropout)

        # Input and output projections
        self.input_proj = nn.Conv2d(in_channels, dim_feedforward, kernel_size=1)
        self.output_proj = nn.Conv2d(dim_feedforward, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode input image
        x = self.input_proj(x)
        enc_out = self.encoder(x)

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
