import numpy as np
import matplotlib.pyplot as plt
import os

# Directory containing the .npz files
input_dir = '../data/npz/velocity_dispersion'
output_dir = '../data/images/velocity_dispersion'

# input_dir = '../data/npz/labels'
# output_dir = '../data/images/labels'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate through all .npz files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.npz'):
        file_path = os.path.join(input_dir, filename)

        # Load the npz file
        with np.load(file_path) as data:
            for array_name in data.files:
                array = data[array_name]

                # Plot and save as PNG
                plt.imshow(array)
                plt.axis('off')  # Hide axis
                png_filename = os.path.splitext(filename)[0] + f'_{array_name}.png'
                plt.savefig(os.path.join(output_dir, png_filename), bbox_inches='tight', pad_inches=0)
                plt.close()  # Close the plot to free memory

print("Conversion complete!")