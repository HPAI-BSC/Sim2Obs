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

        # Pair samples by name: keep only files present in both folders,
        # sorted so image i and label i refer to the same sample.
        paired = sorted(set(self.images) & set(self.labels))

        if len(paired) != len(self.images) or len(paired) != len(self.labels):
            print(f"Mismatch between images ({len(self.images)}) and labels ({len(self.labels)}) "
                  f"in {image_dir} / {labels_dir}. Keeping {len(paired)} paired samples.")

        if not paired:
            raise ValueError(f"No matching image/label pairs found between {image_dir} and {labels_dir}.")

        self.images = paired
        self.labels = list(paired)

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


