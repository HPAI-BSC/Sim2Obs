from torch.utils.data import Dataset


class Illustris300Dataset(Dataset):
    def __init__(self, image_dir, labels_dir, transform=None, mask_interlopers=False):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.mask_interlopers = mask_interlopers

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        labels_path = os.path.join(self.labels_dir, self.labels[index])
        images = np.load(img_path)
        if self.mask_interlopers:
            image = images[f'arr_0'][:, :, 0]
            interlopers_vel_mask = images['arr_0'][:, :, 1]  # Shape: (128, 128), boolean mask
            if np.any(interlopers_vel_mask != 0):
                image = np.ma.masked_array(image, mask=interlopers_vel_mask).filled(fill_value=0)
        else:
            image = images[f'arr_0']
        # resize image to 128x128
        image = image[1:129, 1:129]
        image = torch.tensor(image)

        labels = np.load(labels_path)
        if self.mask_interlopers:
            label = labels[f'arr_0'][:, :, 0]
            interlopers_label_mask = labels['arr_0'][:, :, 1]
            label = np.ma.masked_array(label, mask=interlopers_label_mask).filled(fill_value=0)
        else:
            label = labels[f'arr_0']

        label = torch.tensor(label)
        # resize image to 128x128
        label = label[1:129, 1:129]

        image = image.float().unsqueeze(0)
        if self.transform:
            image = self.transform(image)
        label = label.float().unsqueeze(0)
        return image, label


class PredictDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, _ = self.original_dataset[idx]
        return x
