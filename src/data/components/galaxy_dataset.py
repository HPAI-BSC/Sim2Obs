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

        # Pair samples by name: both folders are sorted, so sample i matches
        # label i. Walk both listings in tandem and stop at the first name
        # mismatch — everything after that point cannot be paired reliably,
        # so the dataset length is that of the longest matching prefix.
        paired_images = []
        paired_labels = []
        for img_name, label_name in zip(self.images, self.labels):
            if img_name != label_name:
                break
            paired_images.append(img_name)
            paired_labels.append(label_name)

        if len(paired_images) != len(self.images) or len(paired_labels) != len(self.labels):
            print(f"Mismatch between images ({len(self.images)}) and labels ({len(self.labels)}) "
                  f"in {image_dir} / {labels_dir}. Keeping the first {len(paired_images)} paired samples.")

        if not paired_images:
            raise ValueError(f"No matching image/label pairs found between {image_dir} and {labels_dir}.")

        self.images = paired_images
        self.labels = paired_labels

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


