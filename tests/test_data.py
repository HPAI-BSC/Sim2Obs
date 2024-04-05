from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule
from src.data.utils import split_dataset


def abs_diff(array, expected_length):
    return abs(len(array) - expected_length)


def test_mnist_dataset_split():
    data_dir = "local/data/raw"
    split_mode = "random"
    num_samples = 5_000
    train_val_test_split = [0.85, 0.05, 0.10]
    train_split, val_split, test_split = train_val_test_split
    train_val_test_data = split_dataset(data_dir, split_mode, train_val_test_split)

    train_images = train_val_test_data["train_data"][0]
    val_images = train_val_test_data["val_data"][0]
    test_images = train_val_test_data["test_data"][0]
    train_labels = train_val_test_data["train_data"][1]
    val_labels = train_val_test_data["val_data"][1]
    test_labels = train_val_test_data["test_data"][1]

    expected_train_samples = int(num_samples * train_split)
    expected_val_samples = int(num_samples * val_split)
    expected_test_samples = int(num_samples * test_split)

    assert abs_diff(train_images, expected_train_samples) <= 1
    assert abs_diff(val_images, expected_val_samples) <= 1
    assert abs_diff(test_images, expected_test_samples) <= 1
    assert abs_diff(train_labels, expected_train_samples) <= 1
    assert abs_diff(val_labels, expected_val_samples) <= 1
    assert abs_diff(test_labels, expected_test_samples) <= 1


def test_mnist_dataset_split_startified():
    data_dir = "local/data/raw"
    split_mode = "stratified"
    train_val_test_split = [0.85, 0.05, 0.10]
    train_val_test_data = split_dataset(data_dir, split_mode, train_val_test_split)

    splits = ["train_data", "val_data", "test_data"]
    counts = defaultdict(lambda: np.zeros(10))

    for split in splits:
        labels = train_val_test_data[split][1]
        for label in labels:
            counts[split][label] += 1

    for i in range(10):
        assert (
            abs(
                counts["train_data"][i] / counts["train_data"].sum()
                - counts["test_data"][i] / counts["test_data"].sum()
            )
            <= 0.01
            and abs(
                counts["train_data"][i] / counts["train_data"].sum()
                - counts["val_data"][i] / counts["val_data"].sum()
            )
            <= 0.01
        )


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that the necessary attributes were created (e.g., the
    dataloader objects), and that dtypes and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "local/data/raw"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir).exists()
    assert len(list(Path(data_dir).glob("*.jpg"))) > 0

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 5_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64
