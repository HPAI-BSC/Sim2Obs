import os

import numpy as np
import torch

from src.data.components.galaxy_dataset import GalaxyDataset


class Illustris300Dataset(GalaxyDataset):
    name = 'illustris-300'

    def __init__(self, image_dir, labels_dir, transform=None, mask_interlopers=True, **kwargs):
        super().__init__(image_dir, labels_dir, transform, **kwargs)
        self.mask_interlopers = mask_interlopers

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

        print(f"Image path: {img_path}, Label path: {label_path}")

        # Process image and label
        image = self._load_and_process_npy(img_path)
        label = self._load_and_process_npy(label_path)

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        if image is None or label is None:
            raise ValueError(f'Error, either the image or the label with index: {index} are None.')

        return image, label

    def _load_and_process_npy(self, file_path):
        """
        Load and process a .npy file.

        Args:
            file_path (str): Path to the .npy file.

        Returns:
            torch.Tensor: Processed tensor with shape (H, W, C).
        """
        try:
            data = np.load(file_path)
            print(f"Image data shape: {data.shape}, dtype: {data.dtype}")

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {e}")

        if self.mask_interlopers:
            main_channel = data['arr_0'][:, :, 0][1:129, 1:129]
            interlopers_mask = data['arr_0'][:, :, 1][1:129, 1:129]

            # Log the results of the loading process
            print(f"Loaded image: {main_channel}, Loaded label: {interlopers_mask}")

            if main_channel is None:
                raise ValueError(f"Failed to load image from: {file_path}")
            if interlopers_mask is None:
                raise ValueError(f"Failed to load label from: {file_path}")

            if np.any(interlopers_mask != 0):
                main_channel = np.ma.masked_array(main_channel, mask=interlopers_mask).filled(fill_value=0)
                main_channel = torch.as_tensor(main_channel.astype('float'))
                main_channel = main_channel.float().unsqueeze(0)
                return main_channel
        else:
            main_channel = data['arr_0'][:, :, 0][1:129, 1:129][..., np.newaxis]
            interlopers_mask = data['arr_0'][:, :, 1][1:129, 1:129][..., np.newaxis]
            # Combine channels and convert to tensor
            combined_data = np.concatenate((main_channel, interlopers_mask), axis=-1)
            tensor_data = torch.as_tensor(combined_data.astype('float'))
            tensor_data = tensor_data.permute(2, 0, 1)  # C x H x W

            return tensor_data
