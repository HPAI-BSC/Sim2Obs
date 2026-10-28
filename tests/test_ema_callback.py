import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from src.callbacks.ema import EMACallback


class ToyModule(LightningModule):
    """Minimal regression module for exercising the EMA callback."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return F.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), y)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _dataloader() -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(32, 2, generator=g)
    y = torch.randn(32, 1, generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_ema_update_and_swap_restore() -> None:
    """EMA shadow is initialized on the first batch and blends subsequent updates; validation swaps EMA weights in and restores the live weights after."""
    module = nn.Linear(2, 2)
    cb = EMACallback(decay=0.5)
    with torch.no_grad():
        init_weight = module.weight.clone()

    # First update: shadow is a plain copy of the initial weights.
    cb.on_train_batch_end(None, module, None, None, 0)
    assert cb.ema_params is not None
    assert torch.equal(cb.ema_params[0], init_weight)

    # Simulate an optimizer step, then blend: ema = 0.5 * init + 0.5 * new.
    with torch.no_grad():
        module.weight.add_(1.0)
        new_weight = module.weight.clone()
    cb.on_train_batch_end(None, module, None, None, 1)
    expected_ema = 0.5 * init_weight + 0.5 * new_weight
    assert torch.allclose(cb.ema_params[0], expected_ema)

    # Validation runs on the EMA weights...
    cb.on_validation_start(None, module)
    assert torch.allclose(module.weight, expected_ema)
    # ...and the live weights are restored afterwards.
    cb.on_validation_end(None, module)
    assert torch.allclose(module.weight, new_weight)
    assert cb._backup is None


def test_ema_checkpoint_contains_ema_weights() -> None:
    """Checkpoints saved while the EMA is swapped in carry the EMA weights in `state_dict`."""
    module = nn.Linear(2, 2)
    cb = EMACallback(decay=0.5)
    cb.on_train_batch_end(None, module, None, None, 0)
    with torch.no_grad():
        module.weight.add_(1.0)
    cb.on_train_batch_end(None, module, None, None, 1)

    cb.on_validation_start(None, module)
    checkpoint = {"state_dict": module.state_dict()}
    cb.on_save_checkpoint(None, module, checkpoint)
    cb.on_validation_end(None, module)

    for name, ema_p in zip([n for n, _ in module.named_parameters()], cb.ema_params):
        assert torch.allclose(checkpoint["state_dict"][name], ema_p)


def test_ema_state_dict_roundtrip() -> None:
    """The EMA shadow survives the callback state_dict roundtrip used for checkpoint resume."""
    module = nn.Linear(2, 2)
    cb = EMACallback(decay=0.5)
    cb.on_train_batch_end(None, module, None, None, 0)

    restored = EMACallback(decay=0.5)
    restored.load_state_dict(cb.state_dict())
    for ema_p, restored_p in zip(cb.ema_params, restored.ema_params):
        assert torch.equal(ema_p, restored_p)


def test_ema_end_to_end_trainer(tmp_path) -> None:
    """A short Trainer run validates with EMA weights and saves EMA weights into checkpoints."""
    module = ToyModule()
    cb = EMACallback(decay=0.5)
    ckpt_dir = tmp_path / "ckpts"
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        logger=False,
        enable_progress_bar=False,
        callbacks=[cb, ModelCheckpoint(dirpath=ckpt_dir, save_last=True)],
    )
    trainer.fit(module, train_dataloaders=_dataloader(), val_dataloaders=_dataloader())

    assert cb.ema_params is not None
    ckpt = torch.load(ckpt_dir / "last.ckpt", weights_only=False)
    for name, ema_p in zip([n for n, _ in module.named_parameters()], cb.ema_params):
        assert torch.allclose(ckpt["state_dict"][name], ema_p)
    # Callback state (the shadow itself) is also persisted for resume.
    assert "callbacks" in ckpt and cb.state_key in ckpt["callbacks"]
