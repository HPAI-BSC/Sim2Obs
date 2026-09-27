import glob
import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

folder_path_fi = '../data/millenium/2nd_iteration/VelocityDispersion'
folder_path_fo = '../data/millenium/2nd_iteration/IntraClusterLight'

#folder_path = '../../../../local/data/datasets/millenium/MTNG-L31.25-270-A/no_rotation_s1.5/VelocityDispersion'
#folder_path = '../../../../local/data/datasets/millenium/MTNG-L31.25-270-A/no_rotation_s1.5/IntraClusterLight'

#folder_path = '../../../../local/data/datasets/illustris300/no_rotation_s1.5/VelocityDispersion'
#folder_path = '../../../../local/data/datasets/illustris300/no_rotation_s1.5/IntraClusterLight'

#folder_path = '../../../../local/data/datasets/eagle/RefL0050N0752/no_rotation_s1.5/VelocityDispersion'
#folder_path = '../../../../local/data/datasets/eagle/RefL0050N0752/no_rotation_s1.5/IntraClusterLight'

#folder_path = '../../../../local/data/datasets/eagle/AGNdT9L0050N0752/no_rotation_s1.5/VelocityDispersion'
#folder_path = '../../../../local/data/datasets/eagle/AGNdT9L0050N0752/no_rotation_s1.5/IntraClusterLight'

folder_path = folder_path_fi
npz_files = glob.glob(os.path.join(folder_path, "*.npz"))

npz_files.sort()


def get_halo_size(data):
    halo_array = data['arr_0']  # Assuming 'arr_0' contains the 2D halo data
    halo_size = halo_array.shape[0]  # This will give (height, width)
    return halo_size

errors = []
for file_path in npz_files:
    try:
        print(file_path)
        # Load the npz file
        data = np.load(file_path)
        filename = file_path.split('/')[-1].split('.')[0]

        # Calculate the halo size dynamically based on the halo data
        halo_array = data['arr_0']
        halo_size = get_halo_size(data)  # Replace with actual calculation logic

        # Masking interlopers flag
        mask_interlopers = True

        if mask_interlopers:
            # Get the main channel and interlopers mask
            main_channel = data['arr_0'][:, :, 0][:halo_size, :halo_size]
            interlopers_mask = data['arr_0'][:, :, 1][:halo_size, :halo_size]

            # Check if the data is loaded correctly
            if main_channel is None:
                raise ValueError(f"Failed to load image from: {file_path}")
            if interlopers_mask is None:
                raise ValueError(f"Failed to load label from: {file_path}")

            # Mask the main channel based on the interlopers mask
            main_channel = np.ma.masked_array(main_channel, mask=interlopers_mask).filled(fill_value=0)
            main_channel = torch.as_tensor(main_channel.astype('float'))
            main_channel = main_channel.float().unsqueeze(0)  # Add channel dimension
        else:
            main_channel = data['arr_0'][:, :, 0][:halo_size, :halo_size][..., np.newaxis]
            interlopers_mask = data['arr_0'][:, :, 1][:halo_size, :halo_size][..., np.newaxis]

            # Combine the channels and convert to tensor
            combined_data = np.concatenate((main_channel, interlopers_mask), axis=-1)
            tensor_data = torch.as_tensor(combined_data.astype('float'))
            tensor_data = tensor_data.permute(2, 0, 1)  # C x H x W (channels x height x width)

        # Convert to NumPy for visualization
        image_array = main_channel.squeeze().numpy()

        # Normalize the image to 0-255 range
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255
        image_array = image_array.astype(np.uint8)

        # Apply the default colormap (viridis) to the image
        colored_image = plt.cm.viridis(image_array)  # This returns an RGBA image

        # Remove the alpha channel if you don't want transparency
        colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)  # Convert to 8-bit RGB

        # Save the image using PIL
        image = Image.fromarray(colored_image)
        image.save(f"{folder_path}/{filename}.png")

        # Show the image using matplotlib
        #plt.imshow(image_array)
        #plt.axis('off')
        #plt.show()
    except Exception as e:
        print(e)
        errors.append(file_path)
        continue

print()
print('Files with errors:')
print(errors)