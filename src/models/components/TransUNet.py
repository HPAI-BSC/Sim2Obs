import math
import copy

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from torch.nn.modules.utils import _pair
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from os.path import join as pjoin


from src.models.components.TransUNet_resnet import ResNetV2, np2th

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

'''
Changes from the original class :
https://github.com/Beckschen/TransUNet/blob/main/networks/vit_seg_modeling.py 

- Config class removed
- Docstrings added
- Head_channels parametrized
'''


class Embeddings(nn.Module):
    """
    Embeds input images into patch embeddings and adds positional encodings.

    Parameters:
    -----------
    hidden_dim : int
        The dimensionality of the hidden layer.
    img_size : int, optional (default=256)
        Size of the input image.
    in_channels : int, optional (default=3)
        Number of input channels.
    patch_size : int, optional (default=16)
        Size of the patches.
    patch_grid : tuple, optional
        Grid for patch extraction in hybrid mode.
    resnet_n_layers : int, optional
        Number of layers in ResNet (if hybrid is used).
    resnet_width_factor : int, optional
        Width factor for ResNet.
    dropout_rate : float, optional (default=0.1)
        Dropout rate.

    Returns:
    --------
    Tensor
        Embedded patches with positional encoding.
    Tensor
        Features extracted from ResNet (if applicable).
    """

    def __init__(self, hidden_dim, img_size=256, in_channels=3, patch_size=16, patch_grid=None,
                 resnet_n_layers=None, resnet_width_factor=None, dropout_rate=0.1):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)
        patch_size = _pair(patch_size)
        if patch_grid is not None:  # ResNet
            patch_size = (img_size[0] // 16 // patch_grid[0], img_size[1] // 16 // patch_grid[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=resnet_n_layers, width_factor=resnet_width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=hidden_dim,
                                       kernel_size=patch_size,
                                       stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_dim))
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        # x = [2, 3, 256, 256]
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2)), [2, 512, 16, 16]
        x = x.flatten(2)    # [2, 512, 256]
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings) # x [2, 256, 512]
        return embeddings, features

# ----------------------- Encoder Block -----------------------


def swish(x):
    """
    Swish activation function.

    Parameters:
    -----------
    x : Tensor
        Input tensor.

    Returns:
    --------
    Tensor
        Activated tensor.
    """

    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    """
    Multi-head self-attention mechanism.

    Parameters:
    -----------
    hidden_dim : int
        Hidden dimension.
    n_heads : int, optional (default=8)
        Number of attention heads.
    vis : bool, optional (default=True)
        Whether to return attention weights.
    attention_dropout_rate : float, optional (default=0.1)
        Dropout rate for attention.

    Returns:
    --------
    Tensor
        Output of attention mechanism.
    Tensor
        Attention weights if `vis` is True.
    """
    def __init__(self, hidden_dim, n_heads, vis, attention_dropout_rate):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = n_heads

        # Each attention head only has access to a subset of the original hidden_size features.
        self.attention_head_size = int(hidden_dim / self.num_attention_heads)   # attention_head_size = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size        # all_head_size = 512

        # Learnable Weight matrices W_q, W_k, W_v, [hidden_size, all_head_size]
        self.query = Linear(hidden_dim, self.all_head_size)     # query = [512, 512]
        self.key = Linear(hidden_dim, self.all_head_size)       # key = [512, 512]
        self.value = Linear(hidden_dim, self.all_head_size)     # value = [512, 512]

        self.out = Linear(hidden_dim, hidden_dim)       # out = [512, 512]
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # new_x_shape = [2, 10, 8, 64]
        x = x.view(*new_x_shape)                                                            # from x = [2, 10, 512] to x = [2, 10, 8, 64]
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # Multiply the embedding by the weight matrices to get Q, K, V
        mixed_query_layer = self.query(hidden_states)   # mixed_query_layer = [2, 10, 512]
        mixed_key_layer = self.key(hidden_states)       # mixed_key_layer = [2, 10, 512]
        mixed_value_layer = self.value(hidden_states)   # mixed_value_layer = [2, 10, 512]

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # softmax(Q * K_transposed / sqrt(heads))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        # attention weights * V
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # Projection back to the original size
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) module used in Transformer blocks.

    Args:
        hidden_dim (int): Dimensionality of the input features.
        mlp_dim (int): Dimensionality of the hidden layer.
        dropout_rate (float): Dropout rate applied after activations.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        act_fn (function): Activation function (GELU).
        dropout (nn.Dropout): Dropout layer.
    """
    def __init__(self, hidden_dim, mlp_dim, dropout_rate):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_dim, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_dim)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Transformer Block consisting of Multi-Head Self-Attention (MHSA) and MLP layers.

    Args:
        hidden_dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        mlp_dim (int): Dimensionality of the hidden layer in the MLP.
        dropout_rate (float): Dropout rate applied after MLP layers.
        attention_dropout_rate (float): Dropout rate applied in the attention mechanism.
        vis (bool): Whether to return attention weights for visualization.

    Attributes:
        attention_norm (nn.LayerNorm): Layer normalization before attention.
        ffn_norm (nn.LayerNorm): Layer normalization before the MLP.
        ffn (Mlp): Feedforward network (MLP).
        attn (Attention): Multi-Head Self-Attention module.
    """
    def __init__(self, hidden_dim, n_heads, mlp_dim, dropout_rate, attention_dropout_rate, vis):
        super(Block, self).__init__()
        self.hidden_size = hidden_dim
        self.attention_norm = LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = Mlp(hidden_dim, mlp_dim, dropout_rate)
        self.attn = Attention(hidden_dim, n_heads, vis, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    # TODO are we gonna use this ?
    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    """
    Encoder module composed of multiple Transformer blocks.

    Args:
        hidden_dim (int): Dimensionality of the hidden layer.
        n_layers (int): Number of Transformer blocks.
        n_heads (int): Number of attention heads.
        mlp_dim (int): Dimensionality of the feed-forward network.
        dropout_rate (float): Dropout rate for feed-forward layers.
        attention_dropout_rate (float): Dropout rate for attention layers.
        vis (bool): Whether to return attention weights for visualization.
    """
    def __init__(self, hidden_dim, n_layers, n_heads, mlp_dim, dropout_rate, attention_dropout_rate, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_dim, eps=1e-6)
        for _ in range(n_layers):
            layer = Block(hidden_dim, n_heads, mlp_dim, dropout_rate, attention_dropout_rate, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


# ----------------------- Transformer -----------------------

class Transformer(nn.Module):
    """
    Transformer model consisting of an embedding layer and an encoder.

    Args:
        img_size (int): Size of the input image.
        hidden_dim (int): Dimensionality of the hidden layer.
        n_layers (int): Number of Transformer blocks.
        n_heads (int): Number of attention heads.
        mlp_dim (int): Dimensionality of the feed-forward network.
        dropout_rate (float): Dropout rate for feed-forward layers.
        attention_dropout_rate (float): Dropout rate for attention layers.
        vis (bool): Whether to return attention weights for visualization.
        in_channels (int, optional): Number of input channels. Defaults to 3.
        patch_size (int, optional): Size of image patches. Defaults to 16.
        patch_grid (tuple, optional): Grid structure of patches.
        resnet_n_layers (int, optional): Number of ResNet layers.
        resnet_width_factor (int, optional): Width factor for ResNet.
    """
    def __init__(self, img_size, hidden_dim, n_layers, n_heads, mlp_dim, dropout_rate, attention_dropout_rate, vis,
                 in_channels, patch_size, patch_grid, resnet_n_layers, resnet_width_factor):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(hidden_dim, img_size, in_channels, patch_size, patch_grid, resnet_n_layers,
                                     resnet_width_factor, dropout_rate)
        self.encoder = Encoder(hidden_dim, n_layers, n_heads, mlp_dim, dropout_rate, attention_dropout_rate, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


# ----------------------- Decoder part  -----------------------

class Conv2dReLU(nn.Sequential):
    """
    Convolutional block with ReLU activation and optional Batch Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        padding (int, optional): Padding size. Defaults to 0.
        stride (int, optional): Stride size. Defaults to 1.
        use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    """
    Decoder block with two convolutional layers and optional skip connections.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        skip_channels (int, optional): Number of skip connection channels. Defaults to 0.
        use_batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        # This forces to consider skip connections always
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    """
    DecoderCup is the decoder module used in TransUNet for reconstructing the
    segmented image from the encoded transformer representations.

    Args:
        hidden_dim (int): Dimensionality of the hidden state from the transformer encoder.
        head_channels (int): Number of output channels after the initial convolution layer.
        decoder_channels (list[int]): List specifying the number of channels at each decoder stage.
        skip_channels (list[int]): List specifying the number of channels for skip connections.
        n_skip (int): Number of skip connections to use in the decoder.

    Attributes:
        conv_more (nn.Sequential): A convolutional block that adjusts the transformer
                                   output to match the decoder's expected input.
        blocks (nn.ModuleList): List of `DecoderBlock` modules used for upsampling and refinement.
        n_skip (int): Stores the number of skip connections used.
    """
    def __init__(self, hidden_dim, head_channels, decoder_channels, skip_channels, n_skip):
        super().__init__()
        # head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_dim,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        in_channels = [head_channels] + list(decoder_channels[:-1])
        #print(f"in_channels: {in_channels}")
        out_channels = decoder_channels
        #print(f"out_channels: {out_channels}")
        self.n_skip = n_skip

        if n_skip != 0:
            # Put the last one to 0 ?
            for i in range(4 - n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3 - i] = 0

        else:
            skip_channels = [0, 0, 0, 0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        #print(f'-decoder-')
        #print(x.shape)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        #print(x.shape)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
            #print(f'-decoder block-')
            #print(x.shape)
        #print(x.shape)
        return x


# ----------------------- Segmentation Head  -----------------------

class SegmentationHead(nn.Sequential):
    """
    A class that defines a segmentation head for the neural network.

    The SegmentationHead class is a part of the decoder in a segmentation network. It consists of a convolutional
    layer followed by an optional upsampling layer. The convolutional layer is used to map the input feature
    maps to the desired number of output channels, while the upsampling layer resizes the output feature maps
    if necessary for the segmentation task.

    Args:
        in_channels (int): The number of input channels (depth of the feature maps).
        out_channels (int): The number of output channels (depth of the output segmentation maps).
        kernel_size (int, optional): The size of the kernel for the convolutional layer. Defaults to 3.
        upsampling (int, optional): The scaling factor for upsampling. If greater than 1, upsampling is applied.
                                     Defaults to 1 (no upsampling).

    Attributes:
        conv2d (nn.Conv2d): The convolutional layer that reduces the depth of the feature maps.
        upsampling (nn.UpsamplingBilinear2d or nn.Identity): The upsampling layer to resize the feature maps,
                                                           or identity if no upsampling is needed.

    """
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


# ----------------------- TransUNet  -----------------------

class TransUNet(nn.Module):
    """
    A Transformer-based U-Net model designed for image segmentation tasks.
    This model integrates a Transformer architecture with a U-Net style decoder
    for high-resolution predictions. The architecture consists of a Transformer encoder,
    a decoder with skip connections, and a final segmentation head.

    Args:
        img_size (int, optional): The input image size. Defaults to 224.
        in_channels (int, optional): Number of input channels (e.g., 3 for RGB). Defaults to 3.
        hidden_dim (int, optional): Dimension of hidden layers in the Transformer. Defaults to 512.
        n_layers (int, optional): Number of Transformer layers. Defaults to 2.
        n_heads (int, optional): Number of attention heads in the Transformer. Defaults to 8.
        mlp_dim (int, optional): Dimension of the MLP layers in the Transformer. Defaults to 3072.
        zero_head (bool, optional): If True, initializes the final layer of the Transformer with zeros. Defaults to False.
        vis (bool, optional): If True, visualizes attention maps during training. Defaults to False.
        dropout_rate (float, optional): Dropout rate in Transformer layers. Defaults to 0.1.
        attention_dropout_rate (float, optional): Dropout rate in the attention mechanism. Defaults to 0.1.
        resnet_n_layers (int, optional): Number of layers in the ResNet backbone. Defaults to None.
        resnet_width_factor (float, optional): Factor to scale ResNet width. Defaults to None.
        patch_grid (tuple, optional): Grid size for the patches in the Transformer. Defaults to None.
        patch_size (int, optional): Size of each patch. Defaults to 16.
        classifier (str, optional): The classifier type ('seg' for segmentation). Defaults to 'seg'.
        head_channels (int, optional): Number of channels in the head of the decoder. Defaults to 512.
        decoder_channels (tuple, optional): Number of channels for each layer in the decoder. Defaults to (256, 128, 64, 16).
        skip_channels (tuple, optional): Number of channels for each skip connection layer. Defaults to (256, 128, 64, 16).
        n_skip (int, optional): Number of skip connections to use. Defaults to 0.
        n_classes (int, optional): Number of output classes for segmentation. Defaults to 2.

    """
    def __init__(self, img_size=224, n_channels=3, hidden_dim=512, n_layers=2, n_heads=8, mlp_dim=3072,
                 zero_head=False, vis=False, dropout_rate=0.1, attention_dropout_rate=0.1,
                 resnet_n_layers=None, resnet_width_factor=None, patch_grid=None, patch_size=16, classifier=None,
                 head_channels=512, decoder_channels=(256, 128, 64, 16), skip_channels=(256, 128, 64, 16), n_skip=0,
                 n_classes=1, segmentation_head_kernel_size=3):
        super().__init__()
        if skip_channels is None:
            skip_channels = [512, 256, 64, 16]
        self.num_classes = n_classes
        self.zero_head = zero_head

        # Not sure about this
        self.classifier = classifier
        self.transformer = Transformer(img_size, hidden_dim, n_layers, n_heads, mlp_dim, dropout_rate,
                                       attention_dropout_rate, vis, n_channels, patch_size, patch_grid,
                                       resnet_n_layers, resnet_width_factor)
        self.decoder = DecoderCup(hidden_dim, head_channels, decoder_channels, skip_channels, n_skip)
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=n_classes,
            kernel_size=segmentation_head_kernel_size,
        )

    def forward(self, x):
        """
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        """
        #print(x.shape)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        #print(x.shape)
        x = self.decoder(x, features)
        #print(x.shape)
        logits = self.segmentation_head(x)
        #print(x.shape)
        return logits

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"],
                                                                                      conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


if __name__ == "__main__":
    _ = TransUNet()