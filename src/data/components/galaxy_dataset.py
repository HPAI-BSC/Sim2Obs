import os
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class GalaxyDataset(Dataset, ABC):
    def __init__(self, image_dir, labels_dir, transform=None, mask_interlopers=False, **kwargs):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(labels_dir))

        if len(self.images) != len(self.labels):
            raise ValueError("Mismatch between number of images and labels.")

    def __len__(self):
        return len(self.images)

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def get_halo_ids(self):
        pass

    @abstractmethod
    def filter_by_halos(self, halos):
        pass


class PredictDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, _ = self.original_dataset[idx]
        return x


