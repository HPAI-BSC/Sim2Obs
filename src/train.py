import os
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict
from lightning.pytorch.tuner import Tuner
#from ptflops import get_model_complexity_info


#rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
rootutils.setup_root(__file__, pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)


class MetricsPlotter(Callback):
    """Tracks training/validation metric curves and saves plots at the end of training."""

    PAIRS = {
        "RASE": ("train/RASE", "val/RASE"),
        "MAE": ("train/mae", "val/mae"),
        "Masked_MAE": ("train/masked_mae", "val/masked_mae"),
    }

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.history: Dict[str, Dict[str, List[float]]] = {
            name: {"train": [], "val": []} for name in self.PAIRS
        }

    def _record(self, trainer: Trainer, split: str) -> None:
        metrics = trainer.callback_metrics
        for name, (train_key, val_key) in self.PAIRS.items():
            key = train_key if split == "train" else val_key
            if key in metrics:
                self.history[name][split].append(float(metrics[key]))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._record(trainer, "train")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._record(trainer, "val")

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        for name in self.PAIRS:
            train_curve = self.history[name]["train"]
            val_curve = self.history[name]["val"]
            if not train_curve and not val_curve:
                log.warning(f"No {name} values recorded, skipping plot.")
                continue
            fig, ax = plt.subplots()
            if train_curve:
                ax.plot(range(1, len(train_curve) + 1), train_curve, label="train")
            if val_curve:
                ax.plot(range(1, len(val_curve) + 1), val_curve, label="val")
            ax.set_title(f"{name} curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(name.replace("_", " "))
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"{name}_curve.png"), dpi=150)
            plt.close(fig)
            log.info(f"Saved {name} curve plot to {self.output_dir}")

import torch

# Ensure CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.tensor([1.0], device="cuda")
    print(torch.cuda.memory_summary(device=0, abbreviated=False))
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Set the number of epochs dynamically (for example, 50 epochs)
    #with open_dict(cfg):
    #    cfg.trainer.max_epochs = 1  # Modify this value based on your requirement

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))
    callbacks.append(MetricsPlotter(output_dir=cfg.paths.output_dir))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, profiler='simple')

    """
    # Find optimal batch size
    tuner = Tuner(trainer)

    # Auto-scale batch size by growing it exponentially (default) or binary search
    optimal_bs = tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
    #tuner.scale_batch_size(model, mode="binsearch")

    # Fit as normal with new batch size
    trainer.fit(model)

    print(f"--> Optimal batch size2: {optimal_bs}")
    print(f"--> Optimal batch size: {model.hparams.batch_size }")
    """

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        start_time = time.time()
        start_epoch = trainer.fit_loop.epoch_progress.current.completed
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        train_time = time.time() - start_time
        epochs_completed = trainer.fit_loop.epoch_progress.current.completed - start_epoch
        if epochs_completed > 0:
            log.info(
                f"Total training time: {timedelta(seconds=int(train_time))} "
                f"({epochs_completed} epochs)"
            )
            log.info(f"Avg time per epoch: {timedelta(seconds=int(train_time / epochs_completed))}")
        else:
            log.info(f"Total training time: {timedelta(seconds=int(train_time))} (no epoch completed)")
        """
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False)
            flops = 2 * macs  # MACs -> FLOPs
            print(f"Forward pass FLOPs per sample: {flops:,}")
        """

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        print(trainer.checkpoint_callback)
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
