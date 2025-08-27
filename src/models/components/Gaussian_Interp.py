import torch.nn as nn


# Define the Conditional UNet for velocity prediction
class BasicConditionalUNet(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_channels * 4),
            nn.SiLU(),
            nn.Linear(hidden_channels * 4, hidden_channels * 4)
        )

        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(input_channels * 2, hidden_channels, 3, padding=1),
                                  nn.GroupNorm(8, hidden_channels), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels * 2, 3, stride=2, padding=1),
                                  nn.GroupNorm(8, hidden_channels * 2), nn.SiLU())
        self.enc3 = nn.Sequential(nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, stride=2, padding=1),
                                  nn.GroupNorm(8, hidden_channels * 4), nn.SiLU())

        # Bottleneck
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

    def forward(self, x, t, x0):
        # Embed time
        t_embed = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)

        # Concatenate input with condition
        x = torch.cat([x, x0], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck with time embedding
        b = self.bottleneck(e3 + t_embed)

        # Decoder
        d1 = self.dec1(b)
        d2 = self.dec2(d1 + e2)
        d3 = self.dec3(d2 + e1)

        return d3


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, norm_groups=8):
        super().__init__()
        self.norm_groups = norm_groups
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(norm_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb=None):
        print('-->', x.shape, self.in_channels)
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            if time_emb.ndim == 3:
                time_emb = time_emb.squeeze(1)  # Should be # [1,64]
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)

        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, norm_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x)
        h = h.view(batch, channels, height * width).transpose(1, 2)
        h, _ = self.attention(h, h, h)
        h = h.transpose(1, 2).view(batch, channels, height, width)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class ConditionalUNetOld(nn.Module):
    def __init__(
            self,
            input_channels,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            norm_groups=8,
            time_emb_dim=128
    ):
        super().__init__()
        #norm_groups = input_channels
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks=num_res_blocks

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial projection (concatenates input and condition)
        self.init_conv = nn.Conv2d(input_channels * 2, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [base_channels * mult for mult in channel_mults]  # [base_channels] +
        #in_out_channels = list(zip(channels[:-1], channels[1:]))

        res = 1  # Track resolution for attention

        """
        for i, (in_ch, out_ch) in enumerate(in_out_channels):
            for _ in range(num_res_blocks):
                print('--->',in_ch,out_ch)
                self.down_blocks.append(nn.ModuleList([
                    ResidualBlock(in_ch, out_ch, time_emb_dim, norm_groups),
                    AttentionBlock(out_ch, norm_groups=norm_groups) if res in attention_resolutions else nn.Identity()
                ]))
                in_ch = out_ch

            if i < len(in_out_channels) - 1:  # Don't downsample after last block
                self.down_samples.append(Downsample(out_ch))
                res *= 2
        """
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # Add residual blocks for this level
            for _ in range(num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    ResidualBlock(in_ch, out_ch, time_emb_dim, norm_groups),
                    AttentionBlock(out_ch, norm_groups=norm_groups) if res in attention_resolutions else nn.Identity()
                ]))
                in_ch = out_ch  # Subsequent blocks in same level have same channels

            # Add downsampling (except after last level)
            if i < len(channels) - 2:
                self.down_samples.append(Downsample(out_ch))
                res *= 2

        # Middle blocks
        out_ch = channels[-1]
        self.mid_block1 = ResidualBlock(out_ch, out_ch, time_emb_dim, norm_groups)
        self.mid_attn = AttentionBlock(out_ch, norm_groups=norm_groups)
        self.mid_block2 = ResidualBlock(out_ch, out_ch, time_emb_dim, norm_groups)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        skip_ch = channels[-1]
        #for i, (in_ch, out_ch) in enumerate(reversed(in_out_channels)):

        """
        skips_channels = list(reversed([64, 64, 128, 128, 128, 256, 256, 256, 512, 512]))
        count = 0
        for i in range(len(channels) -1, -1, -1):
            in_ch = channels[i-1]
            out_ch = channels[i]
            for j in range(num_res_blocks + 1):  # +1 because of skip connections
                try:
                    skip_ch = skips_channels[count]
                except:
                    skip_ch = 0
                print(i, j, in_ch, out_ch, skip_ch, '-', out_ch + skip_ch)
                self.up_blocks.append(nn.ModuleList([
                    # skip_ch + out_ch because skip connections are just concatenated
                    ResidualBlock(skip_ch + out_ch, out_ch, time_emb_dim, norm_groups),
                    AttentionBlock(out_ch, norm_groups=norm_groups) if res in attention_resolutions else nn.Identity()
                ]))
                count += 1
        """
        # ToDo: This is hardcoded, implement logic
        #skips_n_outs_channels = [1024, 1024, 768, 768, 512, 384, 256, 128, 128, 128, 64, 64, 64]
        skips_n_outs_channels = [1024, 1024, 768, 512, 256, 256, 256, 128, 128, 128, 64, 64, 64]
        count = 0
        for i in range(len(channels) - 1, -1, -1):
            out_ch = channels[i]
            in_ch = channels[i]
            for j in range(num_res_blocks + 1):  # +1 because of skip connections
                try:
                    skip_n_outs_ch = skips_n_outs_channels[count]
                except:
                    skip_n_outs_ch = 0
                self.up_blocks.append(nn.ModuleList([
                    # skip_ch + out_ch because skip connections are just concatenated
                    ResidualBlock(skip_n_outs_ch, out_ch, time_emb_dim, norm_groups),
                    AttentionBlock(out_ch, norm_groups=norm_groups) if res in attention_resolutions else nn.Identity()
                ]))
                count += 1

            if i > len(channels) - 1:  # Don't upsample after last block
                self.up_samples.append(Upsample(out_ch))
                res //= 2

        # Final layers
        self.final_norm = nn.GroupNorm(norm_groups, base_channels)
        self.final_conv = nn.Conv2d(base_channels, input_channels, 3, padding=1)

    def forward(self, x, t, x0):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))

        # Concatenate input with condition
        x = torch.cat([x, x0], dim=1)
        x = self.init_conv(x)

        # Skip connections
        skips = [x]

        # Downsample
        down_block_idx = 0
        for i in range(len(self.down_samples)):
            # Process residual blocks for this resolution
            for _ in range(self.num_res_blocks):
                res_block, attn = self.down_blocks[down_block_idx]
                x = res_block(x, t_emb)
                x = attn(x)
                skips.append(x)
                down_block_idx += 1

            # Downsample
            x = self.down_samples[i](x)
            skips.append(x)

        # Process the final blocks at the bottom resolution
        for i in range(down_block_idx, len(self.down_blocks)):
            res_block, attn = self.down_blocks[i]
            x = res_block(x, t_emb)
            x = attn(x)
            skips.append(x)

        # Middle blocks
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Upsample
        up_block_idx = 0
        for i in range(len(self.up_samples)):
            # Process residual blocks for this resolution
            for _ in range(self.num_res_blocks + 1):
                res_block, attn = self.up_blocks[up_block_idx]
                x = torch.cat([x, skips.pop()], dim=1)
                x = res_block(x, t_emb)
                x = attn(x)
                up_block_idx += 1

            # Upsample
            x = self.up_samples[i](x)

        # Process the final blocks at the top resolution
        for i in range(up_block_idx, len(self.up_blocks)):
            res_block, attn = self.up_blocks[i]
            try:
                x = torch.cat([x, skips.pop()], dim=1)
            except:
                pass
            x = res_block(x, t_emb)
            x = attn(x)

        # Final
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return

class ConditionalUNet(nn.Module):
    def __init__(
            self,
            input_channels,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            num_res_blocks=2,
            attention_resolutions=(8, 16),
            norm_groups=8,
            time_emb_dim=128
    ):
        super().__init__()
        # norm_groups = input_channels
        self.time_emb_dim = time_emb_dim
        self.num_res_blocks = num_res_blocks

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Initial projection (concatenates input and condition)
        self.init_conv = nn.Conv2d(input_channels * 2, base_channels, 3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        channels = [base_channels * mult for mult in channel_mults]  # [base_channels] +

        res = 1  # Track resolution for attention
        for i in range(len(channels) - 1):
            in_ch = channels[i]
            out_ch = channels[i + 1]

            # Add residual blocks for this level
            for _ in range(num_res_blocks):
                self.down_blocks.append(nn.ModuleList([
                    ResidualBlock(in_ch, out_ch, time_emb_dim, norm_groups),
                    AttentionBlock(out_ch,
                                   norm_groups=norm_groups) if res in attention_resolutions else nn.Identity()
                ]))
                in_ch = out_ch  # Subsequent blocks in same level have same channels

            # Add downsampling (except after last level)
            if i < len(channels) - 2:
                self.down_samples.append(Downsample(out_ch))
                res *= 2

        # Middle blocks
        out_ch = channels[-1]
        self.mid_block1 = ResidualBlock(out_ch, out_ch, time_emb_dim, norm_groups)
        self.mid_attn = AttentionBlock(out_ch, norm_groups=norm_groups)
        self.mid_block2 = ResidualBlock(out_ch, out_ch, time_emb_dim, norm_groups)

        # Upsampling
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        count = 0
        for i in range(len(channels) - 1, -1, -1):
            out_ch = channels[i]
            in_ch = channels[i-1]
            for j in range(num_res_blocks):  # +1 because of skip connections
                print('a', out_ch, in_ch)
                self.up_blocks.append(nn.ModuleList([
                    # skip_ch + out_ch because skip connections are just concatenated
                    ResidualBlock(out_ch, out_ch, time_emb_dim, norm_groups),
                    AttentionBlock(out_ch,
                                   norm_groups=norm_groups) if res in attention_resolutions else nn.Identity()
                ]))
                count += 1

            if i > len(channels) - 1:  # Don't upsample after last block
                self.up_samples.append(Upsample(in_ch))
                res //= 2

        # Final layers
        self.final_norm = nn.GroupNorm(norm_groups, base_channels)
        self.final_conv = nn.Conv2d(base_channels, input_channels, 3, padding=1)

    def forward(self, x, t, x0):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))

        print('*0', x.shape)
        # Concatenate input with condition
        x = torch.cat([x, x0], dim=1)
        print('*', x.shape)
        x = self.init_conv(x)
        print('*', x.shape)

        # Skip connections
        skips = [x]

        print('downsample')

        # Downsample
        down_block_idx = 0
        for i in range(len(self.down_samples)):
            # Process residual blocks for this resolution
            print('*', x.shape)
            for _ in range(self.num_res_blocks):
                res_block, attn = self.down_blocks[down_block_idx]
                x = res_block(x, t_emb)
                x = attn(x)
                down_block_idx += 1

            # Downsample
            x = self.down_samples[i](x)
            skips.append(x)

        print('last blocks')

        # Process the final blocks at the bottom resolution
        for i in range(down_block_idx, len(self.down_blocks)):
            print(i, x.sahpe)
            res_block, attn = self.down_blocks[i]
            x = res_block(x, t_emb)
            x = attn(x)
        skips.append(x)

        print('---')
        print([item.shape for item in skips])
        print('---')

        print('middle_block')

        # Middle blocks
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)
        print(x.shape)

        print('up')

        # Upsample
        up_block_idx = 0
        for i in range(len(self.up_samples)):
            # Process residual blocks for this resolution
            x = torch.cat([x, skips.pop()], dim=1)
            print('*', x.shape)
            for _ in range(self.num_res_blocks + 1):
                res_block, attn = self.up_blocks[up_block_idx]
                x = res_block(x, t_emb)
                x = attn(x)
                up_block_idx += 1

            # Upsample
            x = self.up_samples[i](x)

        print('last blocks')

        # Process the final blocks at the top resolution
        for i in range(up_block_idx, len(self.up_blocks)):
            print(i, x.shape)
            res_block, attn = self.up_blocks[i]
            x = res_block(x, t_emb)
            x = attn(x)

        print('final')

        # Final
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        print('-->', x.shape)
        print('_______' * 3)

        return x




# Gaussian Interpolation Flow Model
class GaussianInterpolationFlow(nn.Module):
    def __init__(self, input_channels, base_channels, channel_mults, num_res_blocks,
                 attention_resolutions, norm_groups, time_emb_dim, device='cuda'):
        super().__init__()
        self.v_net = ConditionalUNet(input_channels,
                                     base_channels,
                                     channel_mults,
                                     num_res_blocks,
                                     attention_resolutions,
                                     norm_groups,
                                     time_emb_dim).to(device)
        self.device = device

    def forward(self, x, t=None, x0=None):
        if t is None or x0 is None:
            raise NotImplementedError('t and/or x0 are None.')
        return self.v_net(x, t, x0)

    def compute_interpolant(self, x, y, t, epsilon):
        alpha_t = (1 - t).view(-1, 1, 1, 1)
        beta_t = t.view(-1, 1, 1, 1)
        sigma_t = (t * (1 - t)).view(-1, 1, 1, 1)

        return alpha_t * x + beta_t * y + sigma_t * epsilon

    def compute_velocity(self, x, y, epsilon, t):
        # Here we just compute the derivative of the interp path: I_t = (1-t) x + (t) y + t(1-t) ε
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

        print('v_true', v_true.shape)
        print('v_pred', v_pred.shape)
        # Compute loss
        loss = nn.MSELoss()(v_pred, v_true)

        if return_preds:
            preds = self.sample(x, num_steps=100)
            return loss, preds
        else:
            return loss

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