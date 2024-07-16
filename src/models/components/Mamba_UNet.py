import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

"""
class GetDevice(pl.lighting):
    def __init__(self):
        super(GetDevice, self).__init__()

    def get_device(self):
        return self.device


gd = GetDevice()
DEVICE = GetDevice.get_device()

print(DEVICE)
"""


class SimpleSSM(nn.Module):
    def __init__(self, in_dim, d_model, d_conv):
        super().__init__()
        self.n = d_model
        self.e = in_dim

        self.in_proj = nn.Linear(in_dim, d_model)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=d_conv, padding=d_conv - 1,
                                groups=d_model)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(d_model, in_dim)

    def forward(self, A, B, C, x):
        # x shape: (batch_size, seq_len, in_channels, d_model)
        batch_size, seq_len, _ = x.shape

        x = self.in_proj(x)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)

        # initialize hidden state
        h = torch.zeros(batch_size, self.n).to('cuda:0')

        # h_prime = torch.einsum('bn,n->bn', h, A) + torch.einsum('bn,n->bn', u, B)
        # y = torch.einsum('bn,n -> bn', h, C)
        # y = self.act(y)
        # y = self.out_proj(y)

        y_seq = []
        # Recursive along seq_len; selective ssm (A,B already discretized)
        for i in range(seq_len):
            u = x[:, i, :]
            # h' = A h + Bu
            #
            A_u_bar = A[:, i, :]
            B_u_bar = B[:, i, :]
            C_u = C[:, i, :]
            h = torch.einsum('bn,bn->bn', h, A_u_bar) + torch.einsum('bn,bn->bn', u, B_u_bar)
            # y = C h
            y = torch.einsum('bn,bn -> bn', h, C_u)
            y = self.act(y)
            y = self.out_proj(y)
            y_seq.append(y)

        y_seq = torch.stack(y_seq, dim=1)

        return y_seq


def patch_image(image, patch_size=16):
    # Images of size (batch,channels,h,w)
    # Returns j flattened patches of size (batch, j, p^2 * c)
    B, C, H, W = image.shape

    # Ensure the image dimensions are divisible by the patch size
    assert H % patch_size == 0 and W % patch_size == 0, \
        f"Image dimensions ({H}, {W}) must be divisible by patch size {patch_size}"

    # Number of patches in each dimension
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    # Reshape the image into patches
    patches = image.reshape(B, C, n_patches_h, patch_size, n_patches_w, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    patches = patches.reshape(B, n_patches_h * n_patches_w, C * patch_size * patch_size)
    return patches


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


def unpatch_image(token_sequence, original_size, patch_size=16):
    # token_sequence of size (batch, j, p^2 * c)
    # Returns image of size (batch, channels, h, w)
    B, J, _ = token_sequence.shape
    C, H, W = original_size

    n_patches_h = H // patch_size
    n_patches_w = W // patch_size

    # Reshape the token sequence back into patches
    patches = token_sequence.reshape(B, n_patches_h, n_patches_w, C, patch_size, patch_size)

    # Permute and reshape to get the original image shape
    image = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)

    return image


class VimBlock(nn.Module):
    def __init__(self, D, E, N):
        super().__init__()
        self.D = D  # Input dimension
        self.E = E  # Hidden dimension
        self.N = N  # SSM state dimension

        # Normalization and linear layers
        self.norm = nn.LayerNorm(D)
        self.linear_x = nn.Linear(D, E)
        self.linear_z = nn.Linear(D, E)

        # Convolutional layers
        self.conv1d_forward = nn.Conv1d(E, E, kernel_size=1)
        self.conv1d_backward = nn.Conv1d(E, E, kernel_size=1)

        # Linear layers for SSM
        self.linear_B_forward = nn.Linear(E, N)
        self.linear_C_forward = nn.Linear(E, N)
        self.linear_Delta_forward = nn.Linear(E, 1)
        self.linear_B_backward = nn.Linear(E, N)
        self.linear_C_backward = nn.Linear(E, N)
        self.linear_Delta_backward = nn.Linear(E, 1)

        # Learnable parameters
        self.parameter_Delta_forward = nn.Parameter(torch.randn(1))
        self.parameter_Delta_backward = nn.Parameter(torch.randn(1))
        self.parameter_A_forward = nn.Parameter(torch.randn(E, N))
        self.parameter_A_backward = nn.Parameter(torch.randn(E, N))

        # Output linear layer
        self.linear_T = nn.Linear(E, D)

        # SSM
        self.SSM = SimpleSSM(E, N, d_conv=3)

    def forward(self, Tl_1):
        B, M, D = Tl_1.shape

        # Normalize input
        T_prime_l_1 = self.norm(Tl_1)

        # Linear transformations
        x = self.linear_x(T_prime_l_1)
        z = self.linear_z(T_prime_l_1)

        # Process in both directions
        outputs = []
        for direction in ['forward', 'backward']:
            if direction == 'backward':
                x = x.flip(1)

            x_prime = F.silu(getattr(self, f'conv1d_{direction}')(x.transpose(1, 2)).transpose(1, 2))
            B_o = getattr(self, f'linear_B_{direction}')(x_prime)
            C_o = getattr(self, f'linear_C_{direction}')(x_prime)

            Delta_o = torch.log(1 + torch.exp(getattr(self, f'linear_Delta_{direction}')(x_prime) +
                                              getattr(self, f'parameter_Delta_{direction}')))

            # A_o = Delta_o.unsqueeze(-1) * getattr(self, f'parameter_A_{direction}') / self.N

            A_o = torch.diagonal(torch.exp(-Delta_o.unsqueeze(-1) * getattr(self, f'parameter_A_{direction}')), dim1=-2,
                                 dim2=-1)

            print(f'A_o: {A_o.size()}')
            print(f'B_o: {B_o.size()}')

            B_o = Delta_o * B_o

            y_o = self.SSM(A_o, B_o, C_o, x_prime)

            if direction == 'backward':
                y_o = y_o.flip(1)

            outputs.append(y_o)

        # Gated outputs
        y_prime_forward = outputs[0] * F.silu(z)
        y_prime_backward = outputs[1] * F.silu(z)

        # Residual connection
        Tl = self.linear_T(y_prime_forward + y_prime_backward) + Tl_1

        return Tl


class DownscaleBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, ssm_dim, patch_size, img_original_size):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.img_original_size = img_original_size
        self.vim_block = VimBlock(input_dim, hidden_dim, ssm_dim) 
        self.downscale = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.patch_linear_proj = nn.Linear(patch_size ** 2, input_dim)

    def forward(self, x, unpatch=True):
        if unpatch:
            x = unpatch_image(x, self.img_original_size, self.patch_size)
        x = self.downscale(x)
        patches = patch_image(x, self.patch_size)
        proj_patches = self.patch_linear_proj(patches)
        E = torch.from_numpy(getPositionEncoding(seq_len=proj_patches.size()[1], d=self.input_dim)).to('cuda:0').to(torch.float32)
        token_sequence = proj_patches + E
        x = self.vim_block(token_sequence)
        return x


class UpscaleBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, ssm_dim, patch_size, img_original_size):
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.img_original_size = img_original_size
        self.vim_block = VimBlock(input_dim, hidden_dim, ssm_dim)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.downscale = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.patch_linear_proj = nn.Linear(patch_size ** 2, input_dim)

    def forward(self, x, unpatch=True):
        if unpatch:
            x = unpatch_image(x, self.img_original_size, self.patch_size)

        upscaled = self.upsample(x)
        patches = patch_image(upscaled, self.patch_size)
        proj_patches = self.patch_linear_proj(patches)
        E = torch.from_numpy(getPositionEncoding(seq_len=proj_patches.size()[1], d=self.input_dim)).to('cuda:0').to(torch.float32)
        token_sequence = proj_patches + E
        x = self.vim_block(token_sequence)
        return x


def half_size(size, times=1):
    # img_size is a tuple (1,128,128). Out: (1,64,64)
    assert len(size) == 3
    for i in range(times):
        size = (size[0], int(size[1]/2), int(size[2]/2))
    return size


class MambaUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, ssm_dim, patch_size=16, img_original_size=(1, 128, 128)):
        super().__init__()
        self.patch_size = patch_size
        self.original_size = img_original_size
        self.downscale1 = DownscaleBlock(input_dim=input_dim, hidden_dim=hidden_dim, ssm_dim=ssm_dim, patch_size=patch_size,
                                         img_original_size=img_original_size)
        self.downscale2 = DownscaleBlock(input_dim=input_dim, hidden_dim=hidden_dim, ssm_dim=ssm_dim, patch_size=patch_size,
                                         img_original_size=half_size(img_original_size, times=1))
        self.downscale3 = DownscaleBlock(input_dim=input_dim, hidden_dim=hidden_dim, ssm_dim=ssm_dim, patch_size=patch_size,
                                         img_original_size=half_size(img_original_size, times=2))

        self.bottleneck = VimBlock(D=input_dim, E=hidden_dim, N=ssm_dim)

        self.upscale1 = UpscaleBlock(input_dim=input_dim, hidden_dim=hidden_dim, ssm_dim=ssm_dim, patch_size=patch_size,
                                     img_original_size=half_size(img_original_size, times=3))
        self.upscale2 = UpscaleBlock(input_dim=input_dim, hidden_dim=hidden_dim, ssm_dim=ssm_dim, patch_size=patch_size,
                                     img_original_size=half_size(img_original_size, times=2))
        self.upscale3 = UpscaleBlock(input_dim=input_dim, hidden_dim=hidden_dim, ssm_dim=ssm_dim, patch_size=patch_size,
                                     img_original_size=half_size(img_original_size, times=1))

    def forward(self, x):
        x = self.downscale1(x, False)
        x = self.downscale2(x, True)
        x = self.downscale3(x, True)
        x = self.bottleneck(x)
        x = self.upscale1(x, True)
        x = self.upscale2(x, True)
        x = self.upscale3(x, True)
        x = unpatch_image(x, original_size=self.original_size, patch_size=16)
        return x


if __name__ == "__main__":
    _ = MambaUNet()
