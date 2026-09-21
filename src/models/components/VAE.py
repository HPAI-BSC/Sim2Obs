import torch
import torch.nn as nn

"""
    See https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb
    A simple implementation of Gaussian MLP Encoder and Decoder

    We apply some changes: add batch normalization and xavier initialization
"""


class Encoder(nn.Module):
    """Gaussian MLP encoder: two BatchNorm'd hidden layers, then separate linear heads for the latent mean and log-variance."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Build the encoder's linear/BatchNorm layers and apply Xavier initialization.

        :param input_dim: Flattened input feature dimension.
        :param hidden_dim: Width of the two hidden layers.
        :param latent_dim: Dimension of the latent mean/log-variance outputs.
        """
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Apply Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        """Apply Xavier-uniform init to Linear weights (zeroed bias) and constant init (weight=1, bias=0) to BatchNorm1d layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Encode `x` into the approximate posterior's mean and log-variance.

        :param x: Input tensor, shape `(B, input_dim)`.
        :return: `(mean, log_var)`, each shape `(B, latent_dim)`.
        """
        h_ = self.LeakyReLU(self.bn1(self.FC_input(x)))
        h_ = self.LeakyReLU(self.bn2(self.FC_input2(h_)))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    """MLP decoder: two hidden layers, then a linear + sigmoid output head."""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Build the decoder's linear layers.

        :param latent_dim: Latent input dimension.
        :param hidden_dim: Width of the two hidden layers.
        :param output_dim: Flattened output (reconstruction) dimension.
        """
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Decode a latent sample `x` into a reconstruction in `[0, 1]` (sigmoid output).

        :param x: Latent tensor, shape `(B, latent_dim)`.
        :return: Reconstruction, shape `(B, output_dim)`.
        """
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAE(nn.Module):
    """Simple Gaussian-MLP Variational Autoencoder: an `Encoder` producing a latent mean/log-variance, and a `Decoder` reconstructing from a reparameterized latent sample."""
    def __init__(self, x_dim, hidden_dim, latent_dim):
        """
        Build the encoder and decoder.

        :param x_dim: Flattened input/output feature dimension.
        :param hidden_dim: Hidden layer width, shared by encoder and decoder.
        :param latent_dim: Latent space dimension.
        """
        super(VAE, self).__init__()
        self.Encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.Decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)

    def reparameterization(self, mean, var):
        """Sample `z = mean + var * epsilon` with `epsilon ~ N(0, 1)` (the reparameterization trick), keeping the sample differentiable w.r.t. `mean`/`var`."""
        #epsilon = torch.randn_like(var).to(DEVICE)  # sampling epsilon
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        """Encode, reparameterize, and decode `x`.

        :param x: Input tensor, shape `(B, x_dim)`.
        :return: `(x_hat, mean, log_var)` — the reconstruction and the
            latent distribution parameters.
        """
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var
