import pandas as pd

cluster_df = pd.read_csv('../../../../local/data/datasets/magneticum/cluster.txt', delim_whitespace=True, header=0)
print(cluster_df.head())

galaxy_df = pd.read_csv('../../../../local/data/datasets/magneticum/galaxies.txt', delim_whitespace=True, header=0)
print(galaxy_df.head())
