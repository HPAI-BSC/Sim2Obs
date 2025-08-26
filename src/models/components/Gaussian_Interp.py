import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


# Define the Conditional UNet for velocity prediction
class ConditionalUNet(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(input_channels * 2, hidden_channels, 3, padding=1),
                                  nn.GroupNorm(8, hidden_channels), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
                                  nn.GroupNorm(8, hidden_channels * 2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
                                  nn.GroupNorm(8, hidden_channels * 4), nn.SiLU())

        # Bottleneck
        print('->', hidden_channels * 4, hidden_channels * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 4, hidden_channels * 8, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 8),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 8, hidden_channels * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU()
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, hidden_channels * 2), nn.SiLU())
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, hidden_channels), nn.SiLU())
        self.dec3 = nn.Sequential(nn.Conv2d(hidden_channels, input_channels, 3, padding=1))

    def forward(self, x, t, cond):
        # Embed time
        t_embed = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)

        # Concatenate input with condition
        x = torch.cat([x, cond], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck with time embedding
        print(e3.shape)
        print(t_embed.shape)
        b = self.bottleneck(e3 + t_embed)

        # Decoder
        d1 = self.dec1(b)
        d2 = self.dec2(d1 + e2)
        d3 = self.dec3(d2 + e1)

        return d3




# Gaussian Interpolation Flow Model
class GaussianInterpolationFlow(nn.Module):
    def __init__(self, input_channels, device='cuda'):
        super().__init__()
        self.v_net = ConditionalUNet(input_channels).to(device)
        self.device = device

    def forward(self, x, t=None, cond=None):
        if t is None or cond is None:
            print(x)
            print("t is None or cond is none")
            return torch.tensor([0.0])
        return self.v_net(x, t, cond)

    def compute_interpolant(self, x, y, t, epsilon):
        alpha_t = (1 - t).view(-1, 1, 1, 1)
        beta_t = t.view(-1, 1, 1, 1)
        sigma_t = (t * (1 - t)).view(-1, 1, 1, 1)

        return alpha_t * x + beta_t * y + sigma_t * epsilon

    def compute_velocity(self, x, y, epsilon, t):
        alpha_t = (1 - t).view(-1, 1, 1, 1)
        beta_t = t.view(-1, 1, 1, 1)
        d_alpha = -1
        d_beta = 1
        d_sigma = (1 - 2 * t).view(-1, 1, 1, 1)

        return d_alpha * x + d_beta * y + d_sigma * epsilon

    def train_step(self, x, y, return_preds=False):
        """
        No need to compute optimizer step, lightning takes care of that.
        """
        self.v_net.train()

        # Sample random time and noise
        t = torch.rand(x.shape[0], device=self.device)
        epsilon = torch.randn_like(x)

        # Compute interpolant and ground truth velocity
        I_t = self.compute_interpolant(x, y, t, epsilon)
        v_true = self.compute_velocity(x, y, epsilon, t)

        # Predict velocity
        v_pred = self.v_net(I_t, t.unsqueeze(1), x)

        # Compute loss
        loss = nn.MSELoss()(v_pred, v_true)

        if return_preds:
            preds = self.sample(x, num_steps=10)
            return loss.item(), preds
        else:
            return loss.item(),

    def sample(self, x, num_steps=100):
        self.v_net.eval()
        with torch.no_grad():
            I = x.clone()
            dt = 1.0 / num_steps

            for s in range(num_steps):
                t = torch.full((x.shape[0],), s / num_steps, device=self.device)
                v_pred = self.v_net(I, t.unsqueeze(1), x)
                I = I + v_pred * dt

            return I


if __name__ == "__main__":
    _ = GaussianInterpolationFlow()