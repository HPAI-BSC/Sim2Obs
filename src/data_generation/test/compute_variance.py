from src.data_generation.src.utils import compute_intrahalo_variance, compute_intrahalo_variance_ilaria

input_dir = "../../../local/data/dataset_rotation_45/npz/labels"
filename = f"variance_{input_dir.split('/')[2]}.csv"
df = compute_intrahalo_variance(input_dir)
df.to_csv(filename, index=False)

input_dir = "../../../local/data/unet_dataset/Labels"
filename = "variance_unet_dataset.csv"
df_0 = compute_intrahalo_variance_ilaria(input_dir)
df_0.to_csv(filename, index=False)