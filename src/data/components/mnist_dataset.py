import os
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class MNISTDataset(TorchDataset):
    def __init__(self, data_dir, files, labels, transform):
        super().__init__()
        self._data_dir = data_dir
        self._transform = transform
        self._samples = list(zip(files, labels))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        path, target = self._samples[index]
        sample = self._load_img_data(os.path.join(self._data_dir, path))
        if self._transform is not None:
            sample = self._transform(sample)
        return sample, target

    def _load_img_data(self, path: str):
        img = Image.open(path)
        return np.array(img)
