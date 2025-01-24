from src.data.components.galaxy_dataset import PredictDataset
from src.data.components.galaxy_dataset_factory import GalaxyDatasetFactory
import torch
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms


class GalaxyDataModule(LightningDataModule):
    """
    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str,
        target_maskdir: str,
        dataset: str,
        batch_size: int = 16,
        train_val_test_split: Tuple[int, int, int] = (0.85, 0.05, 0.10),
        resize_shape: Tuple[int, int] = (128, 128),
        random_seed: int = 0,
        num_workers: int = 0,
        pin_memory: int = False,
        mask_interlopers: bool = True
    ) -> None:
        """Initialize a `GalaxyDataModule`.

        :param data_dir: The datasets directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        resize_shape = list(
            resize_shape
        )  # comming from gydra is a OmegaConf.listconfig.ListConfig item
        # train data transformations
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomRotation(degrees=90),
                #transforms.RandomResizedCrop(size=resize_shape, scale=(0.9, 1.0)),
                transforms.Resize(size=resize_shape),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # evaluate data transformations
        self.eval_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=resize_shape),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        print(f'image_dir: {data_dir}')
        print(f'labels_dir: {target_maskdir}')
        full_dataset = GalaxyDatasetFactory.get_dataset(dataset,
                                                        image_dir=data_dir,
                                                        labels_dir=target_maskdir,
                                                        transform=None,
                                                        mask_interlopers=mask_interlopers)

        train_size = int(train_val_test_split[0] * len(full_dataset))
        val_size = int(train_val_test_split[1] * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        print(f'train/val/test sizes: {(train_size, val_size, test_size)}')
        torch.manual_seed(random_seed)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
        predict_ds = PredictDataset(test_ds)

        self.data_train: Optional[Dataset] = train_ds
        self.data_val: Optional[Dataset] = val_ds
        self.data_test: Optional[Dataset] = test_ds
        self.data_predict: Optional[Dataset] = predict_ds

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already

        if not self.data_train and not self.data_val and not self.data_test:
            pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        """Create and return the predict dataloader (same as test but without labels).

        :return: The predict dataloader.
        """
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = GalaxyDataModule()
