from src.data.components.dianoga_dataset import DianogaDataset
from src.data.components.galaxy_dataset import GalaxyDataset
from torch.utils.data import ConcatDataset, Subset
import numpy as np

from src.data.components.illustris300_dataset import Illustris300Dataset


class CombinedDataset(GalaxyDataset):
    name = 'combined'

    def __init__(self, image_dir, labels_dir, transform=None, **kwargs):
        super().__init__()
        # image_dirs and label_dirs should be lists of the datasets that we are combining.
        self.datasets = []
        for img_dir, label_dir in zip(image_dir, labels_dir):
            if 'dianoga' in img_dir or 'ilarias' in img_dir:
                dataset = DianogaDataset(img_dir, label_dir, transform=transform)
            elif 'illustris' in img_dir:
                dataset = Illustris300Dataset(img_dir, label_dir, transform=transform)
            else:
                dataset = None
            self.datasets.append(dataset)
        self.datasets = [dataset for dataset in self.datasets if dataset is not None]
        print('_' * 8)
        for dataset in self.datasets:
            print(len(dataset))
        print('_' * 8)
        combined_dataset = ConcatDataset(self.datasets)
        indices = list(range(len(combined_dataset)))
        np.random.shuffle(indices)
        self.combined_dataset = Subset(combined_dataset, indices)

    def __len__(self):
        return len(self.combined_dataset)

    def __getitem__(self, index):
        return self.combined_dataset[index]

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