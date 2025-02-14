import glob
import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

folder_path = '../../../../illustris_snapshot/rotation_120/VelocityDispersion'
npz_files = glob.glob(os.path.join(folder_path, "*.npz"))

for file_path in npz_files:
    data = np.load(file_path)
    filename = file_path.split('/')[-1].split('.')[0]
    mask_interlopers = True

    if mask_interlopers:
        main_channel = data['arr_0'][:, :, 0][1:129, 1:129]
        interlopers_mask = data['arr_0'][:, :, 1][1:129, 1:129]

        if main_channel is None:
            raise ValueError(f"Failed to load image from: {file_path}")
        if interlopers_mask is None:
            raise ValueError(f"Failed to load label from: {file_path}")

        main_channel = np.ma.masked_array(main_channel, mask=interlopers_mask).filled(fill_value=0)
        main_channel = torch.as_tensor(main_channel.astype('float'))
        main_channel = main_channel.float().unsqueeze(0)
    else:
        main_channel = data['arr_0'][:, :, 0][1:129, 1:129][..., np.newaxis]
        interlopers_mask = data['arr_0'][:, :, 1][1:129, 1:129][..., np.newaxis]
        # Combine channels and convert to tensor
        combined_data = np.concatenate((main_channel, interlopers_mask), axis=-1)
        tensor_data = torch.as_tensor(combined_data.astype('float'))
        tensor_data = tensor_data.permute(2, 0, 1)  # C x H x W

    # Convert to NumPy for visualization
    image_array = main_channel.squeeze().numpy()

    # Normalize image to 0-255 range
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
    image_array = image_array.astype(np.uint8)

    # Apply the default colormap (viridis) to the image
    colored_image = plt.cm.viridis(image_array)  # This returns a RGBA image

    # Remove the alpha channel (if you don't want transparency)
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit RGB

    # Save using PIL
    image = Image.fromarray(colored_image)
    image.save(f"{folder_path}/{filename}.png")

    # Show using matplotlib
    # plt.imshow(image_array)
    # plt.axis('off')
    # plt.show()