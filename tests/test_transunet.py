import os
import unittest

import math
import torch
from torch.nn import LayerNorm

from src.models.components.TransUNet import Attention, Mlp, Block, Encoder, Embeddings, DecoderBlock, DecoderCup


class TestTransUNet(unittest.TestCase):
    def setUp(self):
        os.environ['PROJECT_ROOT'] = '..'

        # Embeddings
        self.hidden_dim = 512
        self.img_size = 256
        self.in_channels = 3
        self.patch_size = 16
        self.patch_grid = None
        self.resnet_n_layers = None
        self.resnet_width_factor = None
        self.dropout_rate = 0.1

        # MSA
        self.n_heads = 8
        self.attention_dropout_rate = 0.1

        # MLP
        self.mlp_dim = 3072

        self.vis = False
        self.batch_size = 2

        # Encoder
        self.n_layers = 2

        # Decoder
        self.head_channels = 512
        self.skip_channels = [512, 256, 64, 16]
        self.n_skip = 0
        self.decoder_channels = [256, 128, 64, 16]

    def test_embeddings(self):
        embeddings = Embeddings(self.hidden_dim, self.img_size, self.in_channels, self.patch_size, self.patch_grid,
                                self.resnet_n_layers, self.resnet_width_factor, self.dropout_rate)

        # Dummy input tensor: batch size = 2, 256x256 image, 3 channels
        input_tensor = torch.randn(self.batch_size, self.in_channels, self.img_size, self.img_size)
        embeddings, features = embeddings(input_tensor)

        # Check the output shape of the embeddings
        self.assertEqual(embeddings.shape, (self.batch_size, self.img_size, 512), "Output embeddings shape should match.")
        # Check that features are returned for hybrid model (should be None if not hybrid)
        self.assertIsNone(features, "Features should be None when using non-hybrid model.")

    def test_attention_forward(self):
        attention = Attention(self.hidden_dim, self.n_heads, self.vis, self.attention_dropout_rate)
        # Mock hidden states (e.g., batch size of 2, sequence length of 10, hidden size of 512)
        hidden_states = torch.randn(self.batch_size, 10, self.hidden_dim)

        # Test forward pass
        attention_output, weights = attention(hidden_states)

        # Check if the output has the correct shape
        self.assertEqual(attention_output.shape, (self.batch_size, 10, self.hidden_dim))

        # If visualization is enabled, weights should be returned
        if self.vis:
            self.assertEqual(weights.shape, (self.batch_size, self.n_heads, 10, 10))

    def test_attention_scores(self):
        attention = Attention(self.hidden_dim, self.n_heads, self.vis, self.attention_dropout_rate)

        # Mock hidden states (e.g., batch size of 2, sequence length of 10, hidden size of 512)
        hidden_states = torch.randn(self.batch_size, 10, self.hidden_dim)

        # Extract attention weights manually to verify the attention computation
        mixed_query_layer = attention.query(hidden_states)
        mixed_key_layer = attention.key(hidden_states)

        query_layer = attention.transpose_for_scores(mixed_query_layer)
        key_layer = attention.transpose_for_scores(mixed_key_layer)

        # Calculate attention scores manually
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(attention.attention_head_size)
        self.assertEqual(attention_scores.shape, (self.batch_size, self.n_heads, 10, 10))  # Expected shape

    def test_mlp(self):
        # Instantiate the MLP model
        model = Mlp(self.hidden_dim, self.mlp_dim, self.dropout_rate)

        # Create a dummy input tensor of size (batch_size, hidden_size)
        dummy_input = torch.randn(self.batch_size, self.hidden_dim)

        # Forward pass through the model
        output = model(dummy_input)

        # Check the output shape
        assert output.shape == (self.batch_size, self.hidden_dim), f"Output shape is incorrect: {output.shape}"

        # Print details of the test to check layer outputs
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

        # Test the behavior of individual layers
        # Forward pass through the first linear layer
        fc1_output = model.fc1(dummy_input)
        print(f"fc1 output shape: {fc1_output.shape}")

        # Apply activation and dropout after fc1
        activated_output = model.act_fn(fc1_output)
        print(f"After activation (GELU) output shape: {activated_output.shape}")

        dropout_output = model.dropout(activated_output)
        print(f"After dropout output shape: {dropout_output.shape}")

        # Forward pass through the second linear layer (fc2)
        fc2_output = model.fc2(dropout_output)
        print(f"fc2 output shape: {fc2_output.shape}")

        # Apply dropout after fc2
        final_output = model.dropout(fc2_output)
        print(f"Final output shape after second dropout: {final_output.shape}")

    def test_block(self):
        self.vis = True

        block = Block(self.hidden_dim, self.n_heads, self.mlp_dim, self.dropout_rate, self.attention_dropout_rate,
                      self.vis)

        # Create a dummy input tensor of shape [batch_size, seq_len, hidden_size]
        x = torch.randn(self.batch_size, 10, self.hidden_dim)

        # Forward pass through the block
        output, attention_weights = block(x)

        # Check the output shapes
        print("Output shape:", output.shape)
        print("Attention weights shape:", attention_weights.shape)

    def test_encoder(self):
        self.vis = True
        encoder = Encoder(self.hidden_dim, self.n_layers, self.n_heads, self.mlp_dim, self.dropout_rate,
                          self.attention_dropout_rate, self.vis)

        # Dummy input tensor (batch size = 2, sequence length = 10, hidden size = 512)
        hidden_states = torch.randn(self.batch_size, 10, self.hidden_dim)

        self.assertEqual(len(encoder.layer), self.n_layers, "Number of encoder layers should be correct.")
        self.assertTrue(isinstance(encoder.encoder_norm, LayerNorm), "encoder_norm should be a LayerNorm.")

        # Test the forward pass of the Encoder
        encoded, attn_weights = encoder(hidden_states)

        # Ensure the output shape is correct
        self.assertEqual(encoded.shape, hidden_states.shape,
                         "The shape of encoded output should match the input hidden_states.")

        # If vis is True, ensure attention weights are returned
        self.assertIsInstance(attn_weights, list, "attn_weights should be a list.")
        self.assertEqual(len(attn_weights), len(encoder.layer),
                         "There should be one attention weights list per layer.")

    def test_decoder_block(self):
        in_channels = 512
        out_channels = 256
        skip_channels = 512
        self.decoder = DecoderBlock(in_channels, out_channels, skip_channels, use_batchnorm=True)
        x = torch.randn(2, in_channels, 32, 32)
        skip = torch.randn(2, skip_channels, 64, 64)

        output = self.decoder(x, skip)

        expected_shape = (2, out_channels, 64, 64)  # Upsampled from 32x32 → 64x64
        self.assertEqual(output.shape, expected_shape)

    def test_decoder_cup(self):
        decoder = DecoderCup(self.hidden_dim, self.head_channels, self.decoder_channels, self.skip_channels,
                             self.n_skip)

        B, n_patch, hidden = 2, 64, self.hidden_dim  # Example batch size and patches
        hidden_states = torch.rand(B, n_patch, hidden)

        # Mock features (skip connections)
        features = [
            torch.rand(B, c, 8, 8) for c in self.skip_channels  # Simulating different feature sizes
        ]

        output = decoder(hidden_states, features)

        # Check output shape (should be upsampled)
        assert output.shape == (B, self.decoder_channels[-1], 128, 128), f"Unexpected output shape: {output.shape}"




