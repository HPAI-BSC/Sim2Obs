from src.data_generation.src.utils import compute_intrahalo_variance

input_dir = "../data/dataset_rotations_90/npz/labels"
filename = f"variance_{input_dir.split('/')[2]}.csv"
df = compute_intrahalo_variance(input_dir)
df.to_csv(filename, index=False)


