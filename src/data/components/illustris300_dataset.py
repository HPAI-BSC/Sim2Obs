import os
import re

import numpy as np
import torch

from src.data.components.galaxy_dataset import GalaxyDataset
from PIL import Image


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

        if not os.path.exists(img_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Image or label file not found at index {index}")

        images = np.load(img_path)
        image = images[f'arr_0'].astype('float64')

        labels = np.load(label_path)
        label = labels[f'arr_0'].astype('float64')

        if self.mask_interlopers:
            image = image[..., 0] * (1 - image)[..., 1]
            label = label[..., 0] * (1 - label)[..., 1]

        if self.transform:
            image, label = self.transform(image, label)

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
            if self.mask_interlopers:
                main_channel = data['arr_0'][:, :, 0][1:129, 1:129]
                interlopers_mask = data['arr_0'][:, :, 1][1:129, 1:129]

                if main_channel is None:
                    raise ValueError(f"Failed to load image from: {file_path}")
                if interlopers_mask is None:
                    raise ValueError(f"Failed to load label from: {file_path}")

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
        except Exception as e:
            raise Exception(f'Something went wrong reading the data: {e}')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")

    def get_halo_ids(self):
        halo_ids = set()
        for file in self.images:
            match = re.match(r'halo_(\d+)', file)
            if match:
                halo_id = int(match.group(1))
                halo_ids.add(halo_id)
        return list(halo_ids)

    @staticmethod
    def _filter_image_files_by_halos(image_files, halo_ids_to_include):
        filtered_files = []

        for file in image_files:
            # Extract the halo ID from the filename using regex (assuming it's in the form 'halo_X')
            match = re.match(r'halo_(\d+)', file)
            if match:
                halo_id = int(match.group(1))
                if halo_id in halo_ids_to_include:
                    filtered_files.append(file)
        return filtered_files

    def filter_by_halos(self, halos):
        """
        Filters the dataset by the list of halo IDs.

        :param halos: List of halo IDs to include in the dataset.
        :return: A new `Illustris300Dataset` instance with only the filtered halos.
        """
        filtered_images = self._filter_image_files_by_halos(self.images, halos)
        filtered_labels = self._filter_image_files_by_halos(self.labels, halos)

        # Create a new dataset instance with the filtered data
        filtered_dataset = Illustris300Dataset(
            image_dir=self.image_dir,
            labels_dir=self.labels_dir,
            transform=self.transform,
            mask_interlopers=self.mask_interlopers
        )
        filtered_dataset.images = filtered_images
        filtered_dataset.labels = filtered_labels
        return filtered_dataset
