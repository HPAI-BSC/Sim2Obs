import os
import re
import matplotlib.pyplot as plt

import numpy as np
import torch

from src.data.components.galaxy_dataset import GalaxyDataset


class EagleDataset(GalaxyDataset):
    name = 'eagle'

    def __init__(self, image_dir, labels_dir, transform=None, mask_interlopers=True, **kwargs):
        super().__init__(image_dir, labels_dir, transform, mask_interlopers, **kwargs)
        self.mask_interlopers = mask_interlopers

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        labels_path = os.path.join(self.labels_dir, self.labels[index])
        images = np.load(img_path)
        image = images[f'arr_0']
        # resize image to 128x128
        image = torch.tensor(image)
        print(image.shape)
        if self.mask_interlopers:
            print('a')
            print(image[...,-1])
            image = image[..., 0] * image[..., 1]
            plt.imshow(image.detach().cpu().numpy())
            plt.savefig("/gpfs/scratch/bsc70/hpai/storage/projects/space/eagle_interlop.png")

        labels = np.load(labels_path)
        label = labels[f'arr_0']
        label = torch.tensor(label)
        # resize image to 128x128
        image = image.float().unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        label = label.float().unsqueeze(0)
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
        :return: A new `DianogaDataset` instance with only the filtered halos.
        """
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