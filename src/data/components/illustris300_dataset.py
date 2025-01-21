import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.components.galaxy_dataset import GalaxyDataset


class Illustris300Dataset(GalaxyDataset):
    name = 'illustris-300'

    def __init__(self, image_dir, labels_dir, transform=None):
        super().__init__(image_dir, labels_dir, transform)

    def __getitem__(self, index):
        """
        Get a single data point from the dataset.

        Args:
            index (int): Index of the data point.

        Returns:
            tuple: (image, label), both as tensors with shape (C, H, W).
        """
        img_path = os.path.join(self.image_dir, self.images[index])
        label_path = os.path.join(self.labels_dir, self.labels[index])

        # Process image and label
        image = self._load_and_process_npy(img_path)
        label = self._load_and_process_npy(label_path)

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def _load_and_process_npy(file_path):
        """
        Load and process a .npy file.

        Args:
            file_path (str): Path to the .npy file.

        Returns:
            torch.Tensor: Processed tensor with shape (H, W, C).
        """
        try:
            data = np.load(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")

        # Extract main channel and interlopers mask
        main_channel = data['arr_0'][:, :, 0][1:129, 1:129][..., np.newaxis]
        interlopers_mask = data['arr_0'][:, :, 1][1:129, 1:129][..., np.newaxis]

        # Combine channels and convert to tensor
        combined_data = np.concatenate((main_channel, interlopers_mask), axis=-1)
        tensor_data = torch.from_numpy(combined_data).permute(0, 1, 2)  # H x W x C

        return tensor_data
