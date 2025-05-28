import numpy as np

from src.data.components.galaxy_dataset import PredictDataset
from src.data.components.galaxy_dataset_factory import GalaxyDatasetFactory
import torch
from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

import numpy as np
from scipy import ndimage
import cv2
import torchvision.transforms.functional as TF
import random
#from skimage.transform import resize


class NumpyTransforms:
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape if isinstance(resize_shape, tuple) else (resize_shape, resize_shape)

    def __call__(self, img):
        """Apply all transforms in sequence"""

        # Apply transforms
        #img = self.random_horizontal_flip(img)
        #img = self.random_vertical_flip(img)
        #img = self.random_rotation(img)
        #img = self.random_resized_crop(img)
        img = self.resize(img)

        # Convert to tensor at the end (assuming PyTorch is needed)
        return self.to_tensor(img)

    # Individual transform methods
    def random_horizontal_flip(self, img, p=0.5):
        if random.random() < p:
            return np.fliplr(img).copy()
        return img

    def random_vertical_flip(self, img, p=0.5):
        if random.random() < p:
            return np.flipud(img).copy()
        return img

    def random_rotation(self, img, degrees=45):
        angle = random.uniform(-degrees, degrees)
        return ndimage.rotate(img, angle, reshape=False, mode='nearest')

    def random_resized_crop(self, img, scale=(0.9, 1.0)):
        height, width = img.shape[:2]

        # Random scale factor
        scale_factor = random.uniform(scale[0], scale[1])

        # Compute new dimensions
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        # Random crop position
        y_start = random.randint(0, max(0, height - new_height))
        x_start = random.randint(0, max(0, width - new_width))

        # Crop image
        cropped = img[y_start:y_start + new_height, x_start:x_start + new_width]

        return self.resize(cropped)

    def resize(self, img):
        if isinstance(self.resize_shape, tuple):
            if isinstance(self.resize_shape[0], list):
                if len(self.resize_shape[0]) == 2:
                    resize_shape = (self.resize_shape[0][0], self.resize_shape[0][1])
                elif len(self.resize_shape[0]) == 1:
                    resize_shape = (self.resize_shape[0], self.resize_shape[1])
                else:
                    resize_shape = (128, 128)
        else:
            resize_shape = (self.resize_shape, self.resize_shape)

        if len(img.shape) == 3:
            img = img[0, :, :]

        resized = cv2.resize(img, resize_shape)
        return resized


    def to_tensor(self, img):
        # Convert NumPy array to PyTorch tensor
        # Make sure shape is [C, H, W] for PyTorch
        if len(img.shape) == 2:  # Grayscale
            img = img[np.newaxis, :, :]
        else:  # RGB/RGBA
            img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img).astype(np.float64)

        #return torch.from_numpy(img).float()
        return torch.tensor(img).float()


class PairedTransforms:
    """
    Custom transform class that applies identical transformations to both image and label
    for image-to-image reconstruction tasks.
    """

    def __init__(self, resize_shape):
        """
        Args:
            resize_shape: Target size for resize and crop operations
        """
        self.resize_shape = resize_shape

    def __call__(self, image, label):
        """
        Apply same transformations to both image and label.

        Args:
            image: PIL Image
            label: PIL Image

        Returns:
            tuple: (transformed_image, transformed_label) as torch tensors
        """

        # Random horizontal flip with same probability
        if random.random() < 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random vertical flip with same probability
        if random.random() < 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Random rotation with same angle
        angle = random.uniform(-45, 45)
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle)

        # Random resized crop with same parameters
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.9, 1.0), ratio=(0.75, 1.33)
        )
        image = TF.resized_crop(image, i, j, h, w, self.resize_shape)
        label = TF.resized_crop(label, i, j, h, w, self.resize_shape)

        # Resize (this might be redundant after RandomResizedCrop, but keeping it as in original)
        image = TF.resize(image, self.resize_shape)
        label = TF.resize(label, self.resize_shape)

        # Convert to tensor
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        return image, label


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
        train_val_test_split: Tuple[int, int, int] = (None, None, None),
        resize_shape: Tuple[int, int] = (128, 128),
        random_seed: int = 0,
        num_workers: int = 0,
        pin_memory: int = False,
        mask_interlopers: bool = True,
        fraction_data: float = 1.0,
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
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.RandomResizedCrop(size=resize_shape, scale=(0.9, 1.0)),
                transforms.Resize(size=resize_shape),
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # evaluate data transformations
        self.eval_transforms = transforms.Compose(
            [
                transforms.Resize(size=resize_shape),
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.transforms = PairedTransforms(resize_shape=resize_shape)

        #self.transforms = NumpyTransforms(resize_shape=resize_shape)

        full_dataset = GalaxyDatasetFactory.get_dataset(dataset,
                                                        image_dir=data_dir,
                                                        labels_dir=target_maskdir,
                                                        transform=self.transforms,
                                                        mask_interlopers=mask_interlopers,
                                                        fraction_data=fraction_data)
        torch.manual_seed(random_seed)

        if train_val_test_split == (None, None, None):
            self.data_train: Optional[Dataset] = None
            self.data_val: Optional[Dataset] = None
            self.data_test = full_dataset  # Use the whole dataset for testing/evaluation
            self.data_predict = PredictDataset(full_dataset)  # Use the whole dataset for prediction
        else:
            halo_ids = full_dataset.get_halo_ids()
            num_halos = len(halo_ids)
            train_size = int(train_val_test_split[0] * num_halos)
            val_size = int(train_val_test_split[1] * num_halos)
            test_size = num_halos - train_size - val_size

            # Ensure validation set is not empty
            if val_size == 0 and train_size > 1:
                val_size = max(1, int(0.05 * num_halos))  # Take at least 5% or 1 sample from training
                train_size -= val_size

            # Ensure at least one training sample
            if train_size == 0 and val_size > 1:
                train_size = 1
                val_size -= 1

            train_halos = set(halo_ids[:train_size])
            val_halos = set(halo_ids[train_size:train_size + val_size])
            test_halos = set(halo_ids[train_size + val_size:])

            print(f"TRAIN HALOS")
            print(train_halos)

            print("VAL HALOS")
            print(val_halos)

            print("TEST HALOS")
            print(test_halos)

            # Apply filtering
            train_ds = full_dataset.filter_by_halos(list(train_halos))
            val_ds = full_dataset.filter_by_halos(list(val_halos))
            test_ds = full_dataset.filter_by_halos(list(test_halos))

            print('_' * 25)
            print(f'train/val/test sizes -- # halos: {(train_size, val_size, test_size)}')
            print(f'train/val/test sizes -- # maps: {(len(train_ds), len(val_ds), len(test_ds))}')
            print('_' * 25)
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
        dataset_length = len(self.data_predict)

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
