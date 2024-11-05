import pandas as pd
import seaborn as sns

df_no_rotation = pd.read_csv("variance_dataset_no_rotations.csv")
df_no_rotation = df_no_rotation.set_index("halo_id")
df_rotation_45 = pd.read_csv("variance_dataset_rotation_45.csv")
df_rotation_45 = df_rotation_45.set_index("halo_id")
df_rotation_45 = df_rotation_45.rename(columns={"icl_variance": "icl_variance_rot_45",
                                                "vel_variance": "vel_variance_rot_45"})
df_rotation_90 = pd.read_csv("variance_dataset_rotations_90.csv")
df_rotation_90 = df_rotation_90.rename(columns={"icl_variance": "icl_variance_rot_90",
                                                "vel_variance": "vel_variance_rot_90"})
df_rotation_90 = df_rotation_90.set_index("halo_id")
df_rotation_120 = pd.read_csv("variance_dataset_rotation_120.csv")
df_rotation_120 = df_rotation_120.rename(columns={"icl_variance": "icl_variance_rot_120",
                                                "vel_variance": "vel_variance_rot_120"})
df_rotation_120 = df_rotation_120.set_index("halo_id")

combined_df = pd.concat([df_no_rotation, df_rotation_45, df_rotation_90, df_rotation_120], ignore_index=False, axis=1)
combined_df.to_csv("variances_across_datasets.csv", index=True)