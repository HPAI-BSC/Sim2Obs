from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric, MeanAbsoluteError
from torchmetrics.image import RelativeAverageSpectralError, RootMeanSquaredErrorUsingSlidingWindow
import torch.nn as nn
#from ptflops import get_model_complexity_info
from src.models.utils import MaskedMAE

torch.set_float32_matmul_precision('medium')


"""
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, yhat, y):
        return torch.sqrt(torch.mean((yhat - y) ** 2))
"""


class MambaLitModule(LightningModule):
    """
    A ``LightningModule`` implements 8 key methods::

        def __init__(self):
        # Define initialization code here.

        def setup(self, stage):
        # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
        # This hook is called on every process when using DDP.

        def training_step(self, batch, batch_idx):
        # The complete training step.

        def validation_step(self, batch, batch_idx):
        # The complete validation step.

        def test_step(self, batch, batch_idx):
        # The complete test step.

        def predict_step(self, batch, batch_idx):
        # The complete predict step.

        def configure_optimizers(self):
        # Define and configure optimizers and LR schedulers.

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None:
        """Initialize a `LitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function -- computed below
        #self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = RMSELoss()

        # metric objects for calculating and averaging accuracy across batches
        # We choose Relative Average Spectral Error (RASE)
        # and Root Mean Squared Error Using Sliding Window
        self.train_RASE = RelativeAverageSpectralError(window_size=8)
        self.val_RASE = RelativeAverageSpectralError(window_size=8)
        self.test_RASE = RelativeAverageSpectralError(window_size=8)

        self.train_rmse = RootMeanSquaredErrorUsingSlidingWindow(window_size=8)
        self.val_rmse = RootMeanSquaredErrorUsingSlidingWindow(window_size=8)
        self.test_rmse = RootMeanSquaredErrorUsingSlidingWindow(window_size=8)

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        # We introduce MaskedMAE to homogenize evals across datasets with different ICL/BCG levels
        self.train_masked_mae = MaskedMAE()
        self.val_masked_mae = MaskedMAE()
        self.test_masked_mae = MaskedMAE()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation RASE
        self.val_rase_best = MinMetric()
        self.val_rmse_best = MinMetric()
        self.val_mae_best = MinMetric()

        self.val_masked_mae_epch = []

        self._estimate_flops()


    def _estimate_flops(self):
        """
        print(f"On train init")
        macs, params = get_model_complexity_info(
            self.net,
            (1, 128, 128),  # Input shape (C, H, W)
            as_strings=False,
            print_per_layer_stat=True,
            verbose=True
        )
        # Clean memory
        torch.cuda.empty_cache()
        print('_' * 25)
        print(f"Computational complexity: {macs} MACs")
        print(f"Forward FLOPs: {macs * 2 * 1e-9:.4f} GFLOPs")
        # Estimate backward pass (typically 2-3× forward)
        backward_flops = macs * 2 * 2.5
        print(f"Backward FLOPs: {backward_flops * 1e-9:.4f} GFLOPs")
        print(f"Number of parameters: {params}")
        print('_' * 25)
        """
        import torch
        from torch.profiler import profile, record_function, ProfilerActivity

        if not torch.cuda.is_available():
            print('_'*25)
            print("Skipping FLOPs estimation: CUDA is not available.")
            print('_'*25)
            return

        # Example input
        x = torch.randn(1, 1, 128, 128).cuda()

        # Move model to CUDA
        model = self.net.cuda()
        model.eval()

        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                record_shapes=True,
        ) as prof:
            with torch.no_grad():
                with record_function("model_inference"):
                    model(x)

        # Print total FLOPs
        #print(prof.key_averages().table(sort_by="flops", row_limit=10))
        print('_'*25)
        total_flops = sum([e.flops for e in prof.key_averages() if e.flops is not None])
        print(f"Total forward FLOPs: {total_flops / 1e9:.2f} GFLOPs")
        print(f"Backward FLOPs: {total_flops * 2.5 / 1e9:.2f} GFLOPs")
        print('_'*25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_RASE.reset()
        self.val_rmse.reset()
        self.val_mae.reset()
        self.val_rase_best.reset()
        self.val_rmse_best.reset()
        self.val_mae_best.reset()

    def model_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        preds = self.forward(x)
        if torch.isnan(preds).any():
            preds[torch.isnan(preds)] = 0.0
        #loss = self.criterion(preds, y)
        # We apply sigmoid both in loss computation and RASE etc computation. BCEwithlogits for better num stability.
        #loss = nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')
        y, preds = torch.sigmoid(y), torch.sigmoid(preds)
        loss = nn.functional.binary_cross_entropy(preds, y, reduction='mean')
        #loss = nn.functional.mse_loss(preds, y, reduction='mean')
        #loss = nn.functional.binary_cross_entropy_with_logits(preds, y, reduction='mean')

        return loss, preds, y

    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        #self.trainer.optimizers[0].zero_grad(set_to_none=True)
        if batch_idx % 25 == 0:
            torch.cuda.empty_cache()

        if batch_idx % 1000 == 0:
            self.train_loss.reset()
            self.train_RASE.reset()
            self.train_rmse.reset()
            self.train_mae.reset()
            self.val_loss.reset()
            self.val_RASE.reset()
            self.val_mae.reset()
            self.val_rmse.reset()

        """
        # Use gradient clipping to prevent memory spikes
        self.clip_gradients(
            optimizer=self.trainer.optimizers[0],
            gradient_clip_val=0.5,
            gradient_clip_algorithm="norm"
        )
        """

        loss, preds, targets = self.model_step(batch)

        if batch_idx % 2000 == 0:
            import matplotlib.pyplot as plt
            plt.imshow(preds[0].permute(1, 2, 0).detach().to(torch.float32).cpu().numpy())
            plt.savefig(f"/path/to")

            plt.imshow(targets[0].permute(1, 2, 0).detach().to(torch.float32).cpu().numpy())
            plt.savefig(f"/path/to")
            plt.close()

        # update and log metrics
        self.train_loss(loss.detach())
        self.train_RASE(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.train_rmse(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.train_mae(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        masked_mae = self.train_masked_mae(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.log("train/masked_mae", masked_mae, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/RASE", self.train_RASE, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/rmse", self.train_rmse, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mae", self.train_mae, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        torch.cuda.empty_cache()
        self.train_loss.reset()
        self.train_RASE.reset()
        self.train_rmse.reset()
        self.train_mae.reset()
        self.val_loss.reset()
        self.val_RASE.reset()
        self.val_rmse.reset()
        self.val_mae.reset()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_RASE(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.val_rmse(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.val_mae(preds.detach().to(torch.float32), targets.detach().to(torch.float32))

        val_masked_mae = self.val_masked_mae(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.val_masked_mae_epch.append(val_masked_mae.item())
        self.log("val/masked_mae", val_masked_mae, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/RASE", self.val_RASE, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/rmse", self.val_rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        torch.cuda.empty_cache()
        rase = self.val_RASE.compute()  # get current val rase
        rmse = self.val_rmse.compute()  # get current val rsme
        mae = self.val_mae.compute()
        self.val_rase_best(rase)  # update best so far val rase
        self.val_rmse_best(rmse)
        self.val_mae_best(mae)

        val_masked_mae = torch.mean(torch.tensor(self.val_masked_mae_epch))
        best_val_masked_mae = torch.min(torch.tensor(self.val_masked_mae_epch))
        self.val_masked_mae_epch = []
        self.log("val/masked_mae_best", best_val_masked_mae, sync_dist=True, prog_bar=True)

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/rase_best", self.val_rase_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/rmse_best", self.val_rmse_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/mae_best", self.val_mae_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_RASE(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.test_rmse(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.test_mae(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.test_mae(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        test_masked_mae = self.test_masked_mae(preds.detach().to(torch.float32), targets.detach().to(torch.float32))
        self.log("test/masked_mae", test_masked_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/RASE", self.test_RASE, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/rmse", self.test_rmse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = MambaLitModule(None, None, None, None)
