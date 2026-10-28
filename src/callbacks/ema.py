from typing import Any, Dict, List, Optional

import torch
from lightning import Callback, LightningModule, Trainer


class EMACallback(Callback):
    """Exponential moving average (EMA) of the model weights.

    Maintains a shadow copy of the trainable parameters, updated after every
    optimizer step as ``ema = decay * ema + (1 - decay) * param``.

    - Validation (and therefore the monitored ``val/loss``, early stopping and
      checkpoint selection) runs with the EMA weights swapped in, and the
      original weights are restored afterwards so training is unaffected.
    - Checkpoints are saved with the EMA weights baked into ``state_dict``, so
      the ``best``/``last`` checkpoints (and the post-fit test, which loads the
      best checkpoint) all evaluate the EMA model.
    - The EMA shadow itself is stored in the checkpoint under the callback's
      state, so resuming training restores it.
    """

    def __init__(self, decay: float = 0.999) -> None:
        """
        :param decay: EMA decay factor. With ``decay=0.999`` the shadow has an
            effective averaging window of ~1000 optimizer steps.
        """
        super().__init__()
        self.decay = decay
        self.ema_params: Optional[List[torch.Tensor]] = None

    @property
    def state_key(self) -> str:
        return f"ema_decay_{self.decay}"

    def state_dict(self) -> Dict[str, Any]:
        return {"ema_params": self.ema_params}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ema_params = state_dict.get("ema_params")

    @staticmethod
    def _tracked_params(pl_module: LightningModule) -> List[torch.Tensor]:
        return [p for p in pl_module.parameters() if p.requires_grad]

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: Any,
        batch: Any, batch_idx: int,
    ) -> None:
        """Update the EMA shadow after the optimizer step."""
        params = self._tracked_params(pl_module)
        if self.ema_params is None:
            self.ema_params = [p.detach().clone() for p in params]
            return
        with torch.no_grad():
            for ema_p, p in zip(self.ema_params, params):
                if ema_p.device != p.device:
                    ema_p = ema_p.to(p.device)
                ema_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def _swap_in_ema(self, pl_module: LightningModule) -> None:
        """Save the live weights and overwrite them with the EMA shadow."""
        params = self._tracked_params(pl_module)
        self._backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, ema_p in zip(params, self.ema_params):
                p.copy_(ema_p)

    def _restore_backup(self, pl_module: LightningModule) -> None:
        """Restore the live weights saved by `_swap_in_ema`."""
        with torch.no_grad():
            for p, backup in zip(self._tracked_params(pl_module), self._backup):
                p.copy_(backup)
        self._backup = None

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Evaluate validation with the EMA weights (no-op until EMA exists)."""
        if self.ema_params is None:
            return
        self._swap_in_ema(pl_module)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if getattr(self, "_backup", None) is None:
            return
        self._restore_backup(pl_module)

    def on_save_checkpoint(self, trainer: Trainer, pl_module: LightningModule, checkpoint: Dict[str, Any]) -> None:
        """Save checkpoints with the EMA weights in place of the live weights."""
        if self.ema_params is None:
            return
        state = checkpoint["state_dict"]
        param_names = [n for n, p in pl_module.named_parameters() if p.requires_grad]
        for name, ema_p in zip(param_names, self.ema_params):
            if name in state:
                state[name] = ema_p.detach().clone().to(state[name].dtype)
