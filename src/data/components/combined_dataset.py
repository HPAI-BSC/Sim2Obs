from src.data.components.dianoga_dataset import DianogaDataset
from src.data.components.galaxy_dataset import GalaxyDataset
from torch.utils.data import ConcatDataset, Subset
import numpy as np
import torch

from src.data.components.illustris300_dataset import Illustris300Dataset
from src.data.components.eagle_dataset import EagleDataset


class CombinedDataset(GalaxyDataset):
    name = 'combined'

    def __init__(self, image_dir, labels_dir, transform=None, **kwargs):
        # image_dirs and label_dirs should be lists of the datasets that we are combining.
        self.datasets = []
        for img_dir, label_dir in zip(image_dir, labels_dir):
            if 'dianoga' in img_dir or 'ilarias' in img_dir:
                dataset = DianogaDataset(img_dir, label_dir, transform=transform)
            elif 'illustris' in img_dir:
                dataset = Illustris300Dataset(img_dir, label_dir, transform=transform)
            elif 'eagle' in img_dir:
                dataset = EagleDataset(img_dir, label_dir, transform=transform)
            else:
                dataset = None
            self.datasets.append(dataset)
        self.datasets = [dataset for dataset in self.datasets if dataset is not None]
        print('_' * 25)
        for dataset in self.datasets:
            print(len(dataset))
            print(dataset[0][0].shape, dataset[0][1].shape)
        print('_' * 25)
        combined_dataset = ConcatDataset(self.datasets)
        indices = list(range(len(combined_dataset)))
        np.random.shuffle(indices)
        self.combined_dataset = Subset(combined_dataset, indices)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, index):
        item = self.combined_dataset[index]

        # Ensure all tensors are contiguous and consistent
        if isinstance(item, dict):
            return {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            return tuple(v.contiguous() if isinstance(v, torch.Tensor) else v for v in item)
        elif isinstance(item, torch.Tensor):
            return item.contiguous()
        else:
            return item


    def get_halo_ids(self):
        halo_ids = []
        for dataset in self.datasets:
            halo_ids.extend(dataset.get_halo_ids())
        return halo_ids

    def filter_by_halos(self, halos):
        filtered_datasets = []
        for dataset in self.datasets:
            filtered_dataset = dataset.filter_by_halos(halos)
            filtered_datasets.append(filtered_dataset)

        filtered_combined_dataset = ConcatDataset(filtered_datasets)
        indices = list(range(len(filtered_combined_dataset)))
        np.random.shuffle(indices)
        filtered_combined_dataset = Subset(filtered_combined_dataset, indices)
        return filtered_combined_dataset