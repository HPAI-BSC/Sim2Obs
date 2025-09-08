from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import torch
import numpy as np
import random

import matplotlib.pyplot as plt

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


def plot_predictions(predictions, ground_truths, output_path):
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")

    # Create a grid of subplots
    fig, axes = plt.subplots(2, len(predictions), figsize=(len(predictions) * 4, 2 * 4))

    # Plot each image in a subplot
    i = 0
    for pred, gt in zip(predictions, ground_truths):
        if isinstance(pred, torch.Tensor):
            #pred = pred.permute(1, 2, 0).detach().cpu().numpy()
            pred = pred.permute(1, 2, 0).detach().cpu()
            pred = np.array(pred.tolist())
        else:
            pred = np.transpose(pred, (1, 2, 0))
        if isinstance(gt, torch.Tensor):
            #gt = gt.permute(1, 2, 0).detach().cpu().numpy()
            gt = gt.permute(1, 2, 0).detach().cpu()
            gt = np.array(gt.tolist())
        else:
            gt = np.transpose(pred, (1, 2, 0))
        axes[0, i].imshow(pred)
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Prediction {i}')

        axes[1, i].imshow(gt)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Ground truth {i}')
        i += 1

    plt.tight_layout()

    fig.savefig(f'{output_path}/prediction_sample.png')
    plt.close(fig)


def plot_single_pred(pred, i, output_path):
    if isinstance(pred, torch.Tensor):
        #pred = pred.permute(1, 2, 0).detach().cpu().numpy()
        pred = pred.permute(1, 2, 0).detach().cpu()
        pred = np.array(pred.tolist())
    else:
        pred = np.transpose(pred, (1, 2, 0))

    plt.imshow(pred, cmap="viridis")
    plt.tight_layout()

    plt.savefig(f'{output_path}/img_{i}.png')
    plt.close()


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    if cfg.save_predictions:
        log.info("Predicting...")
        predictions = trainer.predict(model=model, dataloaders=datamodule.predict_dataloader(), ckpt_path=cfg.ckpt_path)

        """
        # Plot the first 3
        pred0, pred1, pred2 = predictions[0][0], predictions[0][1], predictions[0][2]
        preds = [pred0, pred1, pred2]

        for imgs, labels in datamodule.test_dataloader():
            gts = [labels[0], labels[1], labels[2]]  # Get the first two in the batch
            break
        """
        output_path = cfg.paths.output_dir
        random_idxs = random.sample(list(range(len(predictions[0]))), 3)

        pred0, pred1, pred2 = predictions[0][random_idxs[0]], predictions[0][random_idxs[1]], predictions[0][random_idxs[2]]
        preds = [pred0, pred1, pred2]

        for imgs, labels in datamodule.test_dataloader():
            gts = [labels[random_idxs[0]], labels[random_idxs[1]], labels[random_idxs[2]]]  # Get the first two in the batch
            break
        plot_predictions(preds, gts, output_path)

        for i in range(len(gts)):
            pred_i,gts_i = preds[i], gts[i]

            plt.figure(figsize=(8, 6))
            diff = pred_i - gts_i
            diff = diff.permute(1, 2, 0).detach().cpu()
            print(f'sample {i}: {torch.mean(torch.abs(diff))}')
            diff = np.array(diff.tolist())
            im = plt.imshow(diff, cmap="bwr", vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
            plt.colorbar(im, label="ICL frac difference")
            plt.axis("off")
            plt.savefig(f"{output_path}/icl_diff_sample_{i}.png")

        """
        print('plotting ICL frac difference')
        gts0 = gts[0]
        print(gts0.shape)
        print(torch.max(gts0), torch.min(gts0))
        diff_abs = torch.abs(pred0 - gts0)
        diff_abs = diff_abs.permute(1, 2, 0).detach().cpu()
        diff_abs = np.array(diff_abs.tolist())
        plt.imshow(diff_abs)
        plt.savefig(f"{output_path}/icl_diff_sample.png")

        plt.figure(figsize=(8, 6))
        diff = pred0-gts0
        diff = diff.permute(1, 2, 0).detach().cpu()
        diff = np.array(diff.tolist())
        im = plt.imshow(diff, cmap="bwr", vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        plt.colorbar(im, label="ICL frac difference")
        plt.axis("off")
        plt.savefig(f"{output_path}/icl_diff_sample2.png")
        """


        """
        log.info("Saving predictions...")
        output_path = cfg.paths.output_dir
        for batch_idx, batch in enumerate(predictions):
            for i, pred in enumerate(batch):
                global_idx = batch_idx * len(batch) + i
                plot_single_pred(pred, global_idx, output_path)
        """
        log.info(f"Predictions saved to {output_path}")



    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
