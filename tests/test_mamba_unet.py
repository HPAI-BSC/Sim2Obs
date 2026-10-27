from types import SimpleNamespace

import hydra
import pytest
import torch
from hydra import compose, initialize

from src.models.components.Mamba_UNet import (
    MambaUNet,
    PatchMerging2D,
    _selective_scan_ref,
    selective_scan_fn,
)


def _config(**overrides):
    """A minimal, CPU-friendly MambaUNet config (small embed_dim, single block per stage)."""
    base = dict(
        patch_size=4,
        in_chans=1,
        embed_dim=16,
        depths=[1, 1, 1, 1],
        drop_rate=0.0,
        drop_path_rate=0.0,
        patch_norm=True,
        use_checkpoint=False,
        final_upsample="expand_first",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_forward_shape_cpu():
    """The model runs on CPU (via the pure-PyTorch scan fallback) and restores the input resolution."""
    model = MambaUNet(_config(), num_classes=1)
    out = model(torch.randn(1, 1, 32, 32))
    assert out.shape == (1, 1, 32, 32)


def test_grad_flow_cpu():
    """A backward pass produces finite gradients for all parameters."""
    model = MambaUNet(_config(), num_classes=1)
    loss = model(torch.randn(1, 1, 32, 32)).square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads), "some parameters did not receive gradients"
    assert all(torch.isfinite(g).all() for g in grads)


def test_input_divisibility_assert():
    """Inputs whose spatial dims are not divisible by patch_size * 2^(num_layers-1) raise a clear error."""
    model = MambaUNet(_config(), num_classes=1)
    with pytest.raises(AssertionError, match="divisible"):
        model(torch.randn(1, 1, 48, 48))


def test_patch_merging_even_and_odd():
    """PatchMerging2D halves even dims exactly, and truncates (without crashing) on odd dims."""
    pm = PatchMerging2D(dim=8)
    assert pm(torch.randn(1, 6, 6, 8)).shape == (1, 3, 3, 16)
    assert pm(torch.randn(1, 5, 7, 8)).shape == (1, 2, 3, 16)


def test_selective_scan_fn_matches_reference():
    """`selective_scan_fn` (fast kernel when available, pure-PyTorch ref otherwise) matches the
    reference implementation numerically, forward and backward."""
    torch.manual_seed(0)
    batch, dim, seqlen, dstate = 2, 8, 16, 4
    kwargs = dict(dtype=torch.float64, requires_grad=True)
    u = torch.randn(batch, dim, seqlen, **kwargs)
    delta = torch.randn(batch, dim, seqlen, **kwargs)
    A = -torch.rand(dim, dstate, dtype=torch.float64)
    B = torch.randn(batch, dstate, seqlen, **kwargs)
    C = torch.randn(batch, dstate, seqlen, **kwargs)
    D = torch.rand(dim, dtype=torch.float64)
    delta_bias = torch.randn(dim, dtype=torch.float64)

    args = (u, delta, A, B, C)
    out_fast = selective_scan_fn(*args, D=D, delta_bias=delta_bias, delta_softplus=True)
    out_fast.sum().backward()
    grads = {name: p.grad.clone() for name, p in [("u", u), ("delta", delta), ("B", B), ("C", C)]}

    for leaf in (u, delta, B, C):
        leaf.grad = None
    out_ref, last_state = _selective_scan_ref(
        *args, D=D, delta_bias=delta_bias, delta_softplus=True, return_last_state=True
    )
    assert out_fast.shape == (batch, dim, seqlen)
    assert last_state.shape == (batch, dim, dstate)
    assert torch.allclose(out_fast.detach(), out_ref, atol=1e-10)

    out_ref.sum().backward()
    for name in grads:
        assert torch.allclose(grads[name], locals()[name].grad, atol=1e-10)


def test_mamba_unet_config_instantiates():
    """The Hydra model config builds a model whose architecture actually reflects `embed_dim`."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="model/mamba_unet")
    net_cfg = cfg.model.net
    model = hydra.utils.instantiate(net_cfg)
    assert isinstance(model, MambaUNet)
    assert model.mamba_unet.embed_dim == net_cfg.config.embed_dim
    assert model.mamba_unet.dims == [net_cfg.config.embed_dim * 2**i for i in range(len(net_cfg.config.depths))]
