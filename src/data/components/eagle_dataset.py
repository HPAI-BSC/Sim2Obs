import os
import re
from PIL import Image

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from src.data.components.galaxy_dataset import GalaxyDataset


class EagleDataset(GalaxyDataset):
    name = 'eagle'

    def __init__(self, image_dir, labels_dir, transform=None, mask_interlopers=True, **kwargs):
        self.mask_interlopers = mask_interlopers
        # OmegaConf passes YAML lists as ListConfig, which is not a list subclass,
        # so detect multiple directories by excluding path-like single values.
        if not isinstance(image_dir, (str, bytes, os.PathLike)):
            # Multiple EAGLE simulation directories: build one sub-dataset per
            # directory pair and concatenate them (same layout as CombinedDataset).
            self.sub_datasets = [
                EagleDataset(img_dir, lbl_dir, transform=transform, mask_interlopers=mask_interlopers)
                for img_dir, lbl_dir in zip(image_dir, labels_dir)
            ]
            self.concat_dataset = ConcatDataset(self.sub_datasets)
        else:
            super().__init__(image_dir, labels_dir, transform, mask_interlopers, **kwargs)

    def __len__(self):
        if hasattr(self, 'concat_dataset'):
            return len(self.concat_dataset)
        return super().__len__()

    def __getitem__(self, index):
        if hasattr(self, 'concat_dataset'):
            return self.concat_dataset[index]

        img_path = os.path.join(self.image_dir, self.images[index])
        labels_path = os.path.join(self.labels_dir, self.labels[index])

        images = np.load(img_path)
        image = images[f'arr_0'].astype('float64')

        labels = np.load(labels_path)
        label = labels[f'arr_0'].astype('float64')

        if self.mask_interlopers:
            image = image[..., 0] * (1-image)[..., 1]
            label = label[..., 0] * (1-label)[..., 1]

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

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

    def get_halo_ids(self):
        if hasattr(self, 'concat_dataset'):
            halo_ids = []
            for dataset in self.sub_datasets:
                halo_ids.extend(dataset.get_halo_ids())
            return halo_ids

        halo_ids = set()
        for file in self.images:
            match = re.match(r'halo_(\d+)', file)
            if match:
                halo_id = int(match.group(1))
                halo_ids.add(halo_id)
        return list(halo_ids)

    def filter_by_halos(self, halos):
        """
        Filters the dataset by the list of halo IDs.

        :param halos: List of halo IDs to include in the dataset.
        :return: A new dataset instance with only the filtered halos.
        """
        if hasattr(self, 'concat_dataset'):
            filtered = [dataset.filter_by_halos(halos) for dataset in self.sub_datasets]
            return ConcatDataset(filtered)

        filtered_images = self._filter_image_files_by_halos(self.images, halos)
        filtered_labels = self._filter_image_files_by_halos(self.labels, halos)

        # Create a new dataset instance with the filtered data
        filtered_dataset = EagleDataset(
            image_dir=self.image_dir,
            labels_dir=self.labels_dir,
            transform=self.transform
        )
        filtered_dataset.images = filtered_images
        filtered_dataset.labels = filtered_labels
        return filtered_dataset