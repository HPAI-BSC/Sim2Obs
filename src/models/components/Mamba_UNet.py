"""
Code adapted from: https://github.com/ziyangwang007/Mamba-UNet
"""


# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import copy
from torch.utils.checkpoint import checkpoint
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import flop_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

def _selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                        return_last_state=False):
    """Pure-PyTorch reference implementation of the Mamba selective scan (no custom CUDA kernels),
    numerically equivalent to `mamba_ssm.ops.selective_scan_interface.selective_scan_ref`. Used as a
    last-resort fallback when neither `mamba_ssm` nor `selective_scan_cuda` is importable, which also
    makes the model runnable on CPU (e.g. for tests and debugging)."""
    dtype_in = u.dtype
    # Compute in at least float32 (matching mamba_ssm's reference), preserving float64 if given.
    compute_dtype = torch.float64 if u.dtype == torch.float64 else torch.float32
    u = u.to(compute_dtype)
    delta = delta.to(compute_dtype)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].to(compute_dtype)
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    B = B.to(compute_dtype)
    C = C.to(compute_dtype)
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        ys.append(y)
    y = torch.stack(ys, dim=2)  # (batch dim L)
    out = y if D is None else y + u * rearrange(D.to(compute_dtype), "d -> d 1")
    if z is not None:
        out = out * F.silu(z.to(compute_dtype))
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


# import mamba_ssm.selective_scan_fn (in which causal_conv1d is needed)
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    _cuda_scan_available = False
except ImportError:
    selective_scan_fn = None
    try:
        import selective_scan_cuda
        _cuda_scan_available = True
    except ImportError:
        selective_scan_cuda = None
        _cuda_scan_available = False

if _cuda_scan_available and selective_scan_fn is None:
    class SelectiveScanFn(torch.autograd.Function):
        """CUDA-kernel-backed autograd Function for the Mamba selective scan, used as a fallback when `mamba_ssm`'s own implementation isn't importable."""

        @staticmethod
        def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                        return_last_state=False):
            """Run the selective scan forward pass via `selective_scan_cuda.fwd`, saving tensors needed for backward.

            :param ctx: Autograd context.
            :param u: Input sequence, shape `(B, D, L)`.
            :param delta: Per-timestep discretization step, shape `(B, D, L)`.
            :param A: State transition matrix, shape `(D, N)`.
            :param B: Input matrix, shape `(B, N, L)` or `(B, G, N, L)`.
            :param C: Output matrix, shape `(B, N, L)` or `(B, G, N, L)`.
            :param D: Optional skip-connection term, shape `(D,)`.
            :param z: Optional gating tensor, shape `(B, D, L)`.
            :param delta_bias: Optional bias added to `delta`.
            :param delta_softplus: Whether to apply softplus to `delta` (+bias).
            :param return_last_state: If True, also return the final SSM state.
            :return: The scan output (and last state, if requested).
            """
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if z is not None and z.stride(-1) != 1:
                z = z.contiguous()
            if B.dim() == 3:
                B = rearrange(B, "b dstate l -> b 1 dstate l")
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = rearrange(C, "b dstate l -> b 1 dstate l")
                ctx.squeeze_C = True
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, z, delta_bias, delta_softplus)
            ctx.delta_softplus = delta_softplus
            ctx.has_z = z is not None
            last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
            if not ctx.has_z:
                ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
                return out if not return_last_state else (out, last_state)
            else:
                ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
                out_z = rest[0]
                return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
            """Compute gradients for the selective scan via `selective_scan_cuda.bwd`, using the tensors saved in `forward`.

            :param ctx: Autograd context (with saved tensors from `forward`).
            :param dout: Gradient of the loss w.r.t. the scan output.
            :return: Gradients for each `forward` input, in order.
            """
            if not ctx.has_z:
                u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
                z = None
                out = None
            else:
                u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
            # backward of selective_scan_cuda with the backward of chunk).
            # Here we just pass in None and dz will be allocated in the C++ code.
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, z, delta_bias, dout, x, out, None, ctx.delta_softplus,
                False  # option to recompute out_z, not used here
            )
            dz = rest[0] if ctx.has_z else None
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC,
                    dD if D is not None else None,
                    dz,
                    ddelta_bias if delta_bias is not None else None,
                    None,
                    None)

    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                          return_last_state=False):
        """if return_last_state is True, returns (out, last_state)
        last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
        not considered in the backward pass.
        """
        return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


# Fall back to the pure-PyTorch reference scan when no fast implementation is available.
if selective_scan_fn is None:
    selective_scan_fn = _selective_scan_ref
if "selective_scan_ref" not in globals():
    selective_scan_ref = _selective_scan_ref


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        """Estimate multiply-add (MAC) FLOPs for an einsum `equation` over zero-filled arrays of `input_shapes`, via numpy's optimal contraction path."""
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


def selective_scan_flop_jit(inputs, outputs):
    """
    `fvcore` JIT flop-count handle for the selective scan op, inferring shapes/flags from tensor positions and sizes (rather than debug names).

    :param inputs: The traced op's input tensors, in
        `[x, dt, A, B, C, D, dt_bias]` order.
    :param outputs: Unused; required by the `fvcore` handle signature.
    :return: Estimated FLOPs, via `flops_selective_scan_ref`.
    """
    # Inputs are expected in the order: [x, dt, A, B, C, D, dt_bias]
    # Use tensor sizes and positions instead of debug names
    x = inputs[0]  # Should be the main input tensor (B, D, L)
    A = inputs[2]  # Should be the A parameter (D, N)
    B = inputs[3]  # Should be the B parameter (with_Group=True if 4D)

    # Extract dimensions from tensor sizes
    x_sizes = x.type().sizes()
    A_sizes = A.type().sizes()
    B_sizes = B.type().sizes()

    B_dim, D, L = x_sizes  # Batch, Dimension, Sequence length
    N = A_sizes[1]  # State dimension from A matrix

    # Determine model configuration based on tensor dimensions
    with_Group = len(B_sizes) == 4  # Check if B is 4-dimensional

    # Check for D and z parameters based on input count and position
    with_D = len(inputs) > 5 and inputs[5].type().sizes() is not None
    if with_D:
        with_z = len(inputs) > 6
    else:
        with_z = len(inputs) > 5

    # Calculate FLOPs
    flops = flops_selective_scan_ref(B=B_dim, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        """See class docstring for parameters."""
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Patchify `x` via a strided conv, to channels-last `(B, H', W', embed_dim)`, applying normalization if configured."""
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        """See class docstring for parameters."""
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Halve `x`'s spatial resolution by concatenating each 2x2 block of channels, then a 4C->2C linear reduction.

        :param x: Input tensor, shape `(B, H, W, dim)`.
        :return: Output tensor, shape `(B, H//2, W//2, 2*dim)`.
        """
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # Use the (possibly SHAPE_FIX-truncated) spatial dims, not the original H, W — which would
        # mismatch the view for odd input sizes.
        x = x.view(B, x0.shape[1], x0.shape[2], 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    """2D-channels-last (`B,H,W,C`) 2x spatial upsampling: linear expand to `2*dim` channels, then rearrange every 2x2 block into 4 spatial positions at `dim // 2` channels."""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        """
        :param dim: Input channel dimension.
        :param dim_scale: If 2, applies a linear expansion to `2*dim`
            channels before rearranging; otherwise the identity.
        :param norm_layer: Normalization layer applied to the output.
        """
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """Expand channels then rearrange into a 2x upsampled spatial layout.

        :param x: Input tensor, shape `(B, H, W, dim)`.
        :return: Output tensor, shape `(B, 2H, 2W, dim // 2)`.
        """
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    """4x spatial upsampling with a channel compression step: linear expand to `16*dim` channels, rearrange into a 4x-upsampled spatial layout, then a linear `compress` back down to `dim` channels."""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        """
        :param dim: Input/output channel dimension.
        :param dim_scale: Spatial upsampling factor.
        :param norm_layer: Normalization layer applied to the output.
        """
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        after_rearrange_features = (16 * dim) // (dim_scale * dim_scale)
        self.compress = nn.Linear(after_rearrange_features, dim, bias=False)
        #self.compress = nn.Linear(16 * dim, dim, bias=False)

    def forward(self, x):
        """Expand channels, rearrange into a `dim_scale`x upsampled spatial layout, then compress back to `dim` channels.

        :param x: Input tensor, shape `(B, H, W, dim)`.
        :return: Output tensor, shape `(B, dim_scale*H, dim_scale*W, dim)`.
        """
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.compress(x)
        x = self.norm(x)
        return x


class SS2D(nn.Module):
    """
    2D Selective Scan (SS2D) block — the core VMamba operator behind
    `VSSBlock`. Runs 4 directional 1D selective scans over the flattened
    `(B, d_inner, H, W)` feature map (row-major, column-major, and both
    reversed — see `forward_corev0`), lets each direction/channel have
    its own learned SSM parameters (`A_logs`/`Ds`/`dt_projs_*`, `K=4`
    copies), and sums the 4 scan outputs back into one spatial feature
    map. Wrapped by an input projection + depthwise conv + SiLU on the
    way in, and a SiLU-gated output projection on the way out (a
    Mamba-style gated block).
    """
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        """
        :param d_model: Input/output feature dimension.
        :param d_state: SSM state dimension.
        :param d_conv: Depthwise conv kernel size applied before the scan.
        :param expand: Expansion factor for the inner (scan) dimension
            (`d_inner = expand * d_model`).
        :param dt_rank: Rank of the low-rank `dt` projection ("auto" ->
            `ceil(d_model / 16)`).
        :param dt_min: Minimum initial `dt` value (after softplus).
        :param dt_max: Maximum initial `dt` value (after softplus).
        :param dt_init: `dt` projection weight init mode, `"constant"` or `"random"`.
        :param dt_scale: Scale applied to the `dt` projection's init std.
        :param dt_init_floor: Minimum clamp on initial `dt` (pre-inverse-softplus).
        :param dropout: Output dropout probability (disabled if 0).
        :param conv_bias: Whether the depthwise conv has a bias.
        :param bias: Whether the in/out linear projections have a bias.
        :param device: Device for newly created parameters.
        :param dtype: Dtype for newly created parameters.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0
        # self.forward_core = self.forward_corev0_seq
        # self.forward_core = self.forward_corev1

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        """
        Build and initialize the `dt` (discretization step) linear projection.

        Weight init keeps activation variance controlled at
        initialization; bias is set (via inverse-softplus) so that
        `softplus(bias)` starts within `[dt_min, dt_max]`, and is marked
        `_no_reinit` so a later global reinit pass doesn't zero it out.

        :param dt_rank: Input (rank) dimension of the projection.
        :param d_inner: Output dimension of the projection.
        :param dt_scale: Scale applied to the weight init std.
        :param dt_init: `"constant"` or `"random"` weight init mode.
        :param dt_min: Minimum initial `softplus(dt)` value.
        :param dt_max: Maximum initial `softplus(dt)` value.
        :param dt_init_floor: Minimum clamp on the pre-inverse-softplus `dt`.
        :param factory_kwargs: `device`/`dtype` forwarded to `nn.Linear`.
        :return: The initialized `nn.Linear(dt_rank, d_inner)`.
        """
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        """
        Build the learned `log(A)` state-transition parameter, S4D-real initialized (`A[d, n] = n`, for `n` in `[1, d_state]`).

        :param d_state: SSM state dimension.
        :param d_inner: Inner (scan) feature dimension.
        :param copies: Number of independent copies to create (one per
            scan direction, `K=4` in `SS2D`).
        :param device: Device for the created parameter.
        :param merge: If True and `copies > 1`, flatten the copies into
            the leading dimension rather than keeping them separate.
        :return: An `nn.Parameter` of shape `(d_inner, d_state)`, or
            `(copies * d_inner, d_state)`/`(copies, d_inner, d_state)`
            depending on `copies`/`merge`.
        """
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        """
        Build the learned `D` skip-connection parameter, initialized to all ones.

        :param d_inner: Inner (scan) feature dimension.
        :param copies: Number of independent copies to create (one per
            scan direction, `K=4` in `SS2D`).
        :param device: Device for the created parameter.
        :param merge: If True and `copies > 1`, flatten the copies into
            the leading dimension rather than keeping them separate.
        :return: An `nn.Parameter` of shape `(d_inner,)`, or
            `(copies * d_inner,)`/`(copies, d_inner)` depending on
            `copies`/`merge`.
        """
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        """
        Run the 4-directional selective scan core: flatten `x` in row-major and column-major order (plus their reversals, `K=4` total), run one batched selective scan over all 4, then sum the (un-reversed) results back into a single spatial feature map.

        This is the default `forward_core` implementation (set in
        `__init__`); `forward_corev0_seq` and `forward_corev1` are
        alternative (looped, or fp16-oriented) implementations of the
        same computation.

        :param x: Input feature map, shape `(B, d_inner, H, W)`.
        :return: Output feature map, shape `(B, H, W, d_inner)`.
        """
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev0_seq(self, x: torch.Tensor):
        """
        Same computation as `forward_corev0`, but running the selective scan once per direction (a Python loop over `K=4`) instead of one batched call — an alternative core implementation, not used by default.

        :param x: Input feature map, shape `(B, d_inner, H, W)`.
        :return: Output feature map, shape `(B, H, W, d_inner)`.
        """
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)

        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = []
        for i in range(4):
            yi = self.selective_scan(
                xs[:, i], dts[:, i],
                As[i], Bs[:, i], Cs[:, i], Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward_corev1(self, x: torch.Tensor):
        """
        Same computation as `forward_corev0`, but keeping tensors in their native (e.g. fp16) dtype rather than casting to float32 before the scan — an alternative core implementation, not used by default (expects the scan output in float16).

        :param x: Input feature map, shape `(B, d_inner, H, W)`.
        :return: Output feature map, shape `(B, H, W, d_inner)`.
        """
        #self.selective_scan = selective_scan_fn_v1
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.view(B, K, -1, L)  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        Ds = self.Ds.view(-1)  # (k * d)
        dt_projs_bias = self.dt_projs_bias.view(-1)  # (k * d)

        # print(self.Ds.dtype, self.A_logs.dtype, self.dt_projs_bias.dtype, flush=True) # fp16, fp16, fp16

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float16

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0].float() + inv_y[:, 0].float() + wh_y.float() + invwh_y.float()
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        """
        Run the gated SS2D block: input projection + split into (scan input, gate), depthwise conv + SiLU, the 4-directional selective scan (`forward_core`), SiLU-gating, then output projection.

        :param x: Input tensor, shape `(B, H, W, d_model)`.
        :return: Output tensor, shape `(B, H, W, d_model)`.
        """
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        if x.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = self.forward_core(x)
        else:
            y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """Vision-State-Space block: pre-norm residual `SS2D` self-attention (`x + drop_path(SS2D(norm(x)))`) — the Mamba-based analogue of a Swin transformer block."""
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        """
        :param hidden_dim: Input/output feature dimension.
        :param drop_path: Stochastic depth drop probability.
        :param norm_layer: Normalization layer constructor.
        :param attn_drop_rate: Dropout probability inside `SS2D`.
        :param d_state: SSM state dimension for `SS2D`.
        :param kwargs: Forwarded to `SS2D`.
        """
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        """Apply pre-norm residual SS2D self-attention to `input`."""
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        """See class docstring for parameters. Builds `depth` `VSSBlock` instances, re-initializes each block's `SS2D.out_proj.weight` (kaiming-uniform), and adds an optional `downsample` at the end."""
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                """Re-apply kaiming-uniform init to any `out_proj.weight` parameter found in `module` (fake-clone-then-init, to keep the RNG state consistent with the reference implementation)."""
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """Run `x` through all `VSSBlock` instances (optionally with gradient checkpointing) then the downsample layer, if any."""
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        Upsample (nn.Module | None, optional): Upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        """See class docstring for parameters. Builds `depth` `VSSBlock` instances, re-initializes each block's `SS2D.out_proj.weight` (kaiming-uniform), and adds an optional `PatchExpand` upsample at the end."""
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                """Re-apply kaiming-uniform init to any `out_proj.weight` parameter found in `module` (fake-clone-then-init, to keep the RNG state consistent with the reference implementation)."""
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, dim_scale=2, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None

    def forward(self, x):
        """Run `x` through all `VSSBlock` instances (optionally with gradient checkpointing) then the upsample layer, if any."""
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class VSSM(nn.Module):
    """
    Visual State Space Model: a Swin-U-Net-shaped hierarchical
    architecture (`PatchEmbed2D` -> `VSSLayer`/`PatchMerging2D` encoder
    stages -> mirrored `VSSLayer_up`/`PatchExpand` decoder stages with
    skip connections -> `FinalPatchExpand_X4` + output conv) but with
    `VSSBlock` (Mamba selective-scan) attention in place of Swin's
    windowed self-attention.
    """
    def __init__(self, patch_size=4, in_chans=1, num_classes=4, depths=[2, 2, 9, 2],
                 embed_dim=96, dims=None, d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        """
        :param patch_size: Patch token size.
        :param in_chans: Number of input image channels.
        :param num_classes: Number of output classes/channels.
        :param depths: Per-stage `VSSBlock` counts (encoder; mirrored for
            the decoder).
        :param embed_dim: Base (stage-0) embedding dimension; each deeper
            stage doubles it. Ignored if `dims` is given.
        :param dims: Explicit per-stage embedding dimensions (defaults to
            `[embed_dim * 2**i for i in range(len(depths))]`).
        :param d_state: SSM state dimension for every `SS2D` block
            (`ceil(dims[0] / 6)` if None).
        :param drop_rate: Dropout rate.
        :param attn_drop_rate: Dropout rate inside `SS2D`.
        :param drop_path_rate: Stochastic depth rate (linearly scaled
            across all blocks).
        :param norm_layer: Normalization layer constructor.
        :param patch_norm: Whether to normalize after patch embedding.
        :param use_checkpoint: Whether to use gradient checkpointing.
        :param final_upsample: If `"expand_first"`, use
            `FinalPatchExpand_X4` + a conv for the final upsample;
            otherwise the identity.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if dims is None:
            dims = [int(embed_dim * 2 ** i_layer) for i_layer in range(self.num_layers)]
        elif isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.patch_size = patch_size
        self.final_upsample = final_upsample

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * dims[self.num_layers - 1 - i_layer],
                                      dims[self.num_layers - 1 - i_layer]) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(dim=dims[self.num_layers - 1 - i_layer], dim_scale=2,
                                       norm_layer=norm_layer)
            else:
                layer_up = VSSLayer_up(
                    dim=dims[self.num_layers - 1 - i_layer],
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                    norm_layer=norm_layer,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        #print(f'--> {self.final_upsample}')
        if self.final_upsample == "expand_first":
            #print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(dim_scale=self.patch_size, dim=self.embed_dim)
            self.output = nn.Conv2d(in_channels=self.embed_dim, out_channels=self.num_classes, kernel_size=1,
                                    bias=False)
        else:
            #print("No final upsample")
            self.up = lambda x: x

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # Encoder and Bottleneck
    def forward_features(self, x):
        """
        Run the patch embedding and encoder (`VSSLayer`) stages, collecting a skip-connection feature map before each stage.

        :param x: Input image tensor.
        :return: `(x, x_downsample)` — the final (bottleneck) encoded
            features, and the list of per-stage skip-connection features
            (shallowest first).
        """
        x = self.patch_embed(x)

        x_downsample = []
        for i,layer in enumerate(self.layers):
            x_downsample.append(x)
            x = layer(x)
        x = self.norm(x) # B H W C
        return x, x_downsample

    # def forward_backbone(self, x):
    #     x = self.patch_embed(x)

    #     for layer in self.layers:
    #         x = layer(x)
    #     return x

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        """
        Run the decoder (`VSSLayer_up`) stages, concatenating each stage's matching encoder skip connection before processing (except the first stage, which only upsamples).

        :param x: Bottleneck features from `forward_features`.
        :param x_downsample: Per-encoder-stage skip features from
            `forward_features`.
        :return: The decoded features, normalized.
        """
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B H W C

        return x

    def up_x4(self, x):
        """
        Apply the final upsample and output projection, converting decoded features back to a spatial image.

        :param x: Decoded features from `forward_up_features`, shape
            `(B, H, W, C)`.
        :return: Output image tensor, shape `(B, num_classes, H', W')`
            (unchanged if `final_upsample != "expand_first"`).
        """
        if self.final_upsample == "expand_first":
            x = self.up(x)
            # FinalPatchExpand already restores the spatial dims, so no reshape is needed here.
            x = x.permute(0, 3, 1, 2)  # B,C,H,W
            x = self.output(x)

        return x

    def forward(self, x):
        """
        Run the full encoder-decoder VSSM: features, decoded skip-connected features, then the final upsample.

        :param x: Input image tensor.
        :return: Output logits, shape `(B, num_classes, H, W)`.
        """
        _, _, H, W = x.shape
        divisor = self.patch_size * (2 ** (self.num_layers - 1))
        assert H % divisor == 0 and W % divisor == 0, (
            f"Input spatial dims ({H}, {W}) must be divisible by {divisor} "
            f"(patch_size={self.patch_size} x 2^(num_layers-1) downsamplings)."
        )
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        return x

    def flops(self, shape=(3, 224, 224)):
        """Estimate total FLOPs for a forward pass at the given input `shape`, via `fvcore`'s `FlopCountAnalysis` (with custom handles for the selective-scan op)."""
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,  # latter
        }

        model = copy.deepcopy(self)
        device = next(model.parameters()).device
        model.to(device).eval()

        input = torch.randn((1, *shape), device=device)
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9



class MambaUNet(nn.Module):
    """Thin wrapper around `VSSM`, built from a config object (mirroring `SwinUnet`'s wrapper pattern)."""
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        """
        Build the underlying `VSSM` from `config`.

        :param config: Config object with `patch_size`, `in_chans`,
            `embed_dim`, `depths`, `drop_rate`, `drop_path_rate`,
            `patch_norm`, `use_checkpoint`, `final_upsample`.
        :param img_size: Unused (kept for interface parity with
            `SwinUnet`).
        :param num_classes: Number of output classes/channels.
        :param zero_head: Stored but not otherwise used here.
        :param vis: Unused (kept for interface parity with `SwinUnet`).
        """
        super(MambaUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.mamba_unet = VSSM(
                                patch_size=config.patch_size,
                                in_chans=config.in_chans,
                                num_classes=self.num_classes,
                                embed_dim=config.embed_dim,
                                depths=config.depths,
                                drop_rate=config.drop_rate,
                                drop_path_rate=config.drop_path_rate,
                                patch_norm=config.patch_norm,
                                use_checkpoint=config.use_checkpoint,
                                final_upsample=config.final_upsample)

    def forward(self, x):
        """Run the underlying `VSSM`. Unlike `SwinUnet.forward`, does not repeat a single-channel input to 3 channels.

        :param x: Input image tensor.
        :return: Output logits, shape `(B, num_classes, H, W)`.
        """
        logits = self.mamba_unet(x)
        return logits


if __name__ == "__main__":
    from types import SimpleNamespace

    config = SimpleNamespace(patch_size=4, in_chans=1, embed_dim=24, depths=[1, 1, 1, 1],
                             drop_rate=0., drop_path_rate=0., patch_norm=True,
                             use_checkpoint=False, final_upsample="expand_first")
    model = MambaUNet(config, num_classes=1)
    out = model(torch.randn(1, 1, 64, 64))
    print(out.shape)
