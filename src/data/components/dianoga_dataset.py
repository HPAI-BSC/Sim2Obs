import os
import numpy as np
import torch

from src.data.components.galaxy_dataset import GalaxyDataset


class DianogaDataset(GalaxyDataset):
    name = 'dianoga'

    def __init__(self, image_dir, labels_dir, transform=None, **kwargs):
        super().__init__(image_dir, labels_dir, transform, **kwargs)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        labels_path = os.path.join(self.labels_dir, self.labels[index])
        images = np.load(img_path)
        image = images[f'arr_0']
        # resize image to 128x128
        image = image[1:129, 1:129]
        image = torch.tensor(image)

        labels = np.load(labels_path)
        label = labels[f'arr_0']
        label = torch.tensor(label)
        # resize image to 128x128
        label = label[1:129, 1:129]
        image = image.float().unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        label = label.float().unsqueeze(0)
        return image, label


