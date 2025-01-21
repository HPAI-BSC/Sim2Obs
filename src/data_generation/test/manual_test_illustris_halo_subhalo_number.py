import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose

from src.data_generation.src.data_fetcher import DataFetcherFactory


load_dotenv('../.env')


def count_subhalos(group):
    halo_id = group['id_parents'].head(n=1).iloc[0]
    subhalos = len(group)
    return pd.Series([halo_id, subhalos])


# Read config
with initialize(config_path="../../../configs"):
    cfg_illustris = compose(config_name="data_generation.yaml")
    local_config = compose(config_name="paths/local.yaml")
    cfg_illustris.paths = local_config.paths
    cfg_illustris.catalog = compose("catalog/illustris300")['catalog']

os.environ['PROJECT_ROOT'] = '../../..'

# Instantiate the fetcher
required_keys = ['subhalo_min_mass', 'halo_min_mass', 'solar_mass_convention', 'G', 'H']
data_loader_dict = next((step for step in cfg_illustris.pipeline if step.name == 'data_loader'), None)
attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
fetcher_illustris = DataFetcherFactory.get_data_fetcher(cfg_illustris.catalog, **attr_dict)

# Get parameters and constants of the data loader step
data_loader_dict = next((step for step in cfg_illustris.pipeline if step.name == 'data_loader'), None)

# Retrieve raw halos
halo_context = fetcher_illustris.get_halos()
n_halos = halo_context.data['halo_id'].nunique()
total_halo_ids = set(halo_context.data['halo_id'])
print(f'Illustris (raw) halos: {n_halos}')

# Filter halos and subhalos by min mass conditions
halo_conditions = [('m200', '>', data_loader_dict['halo_min_mass'] * fetcher_illustris.H)]
halo_context_filtered = fetcher_illustris.filter_halos(halo_context, halo_conditions)
filtered_halo_ids = set(halo_context_filtered.data['halo_id'])
print(f'Illustris filtered halos by mass: {halo_context_filtered.data["halo_id"].nunique()}')

# Retrieve raw subhalos
subhalo_context = fetcher_illustris.get_subhalos()
n_subhalos = subhalo_context.data['subhalo_id'].nunique()
total_subhalo_ids = set(subhalo_context.data['subhalo_id'])
print(f'Illustris (raw) subhalos: {n_subhalos}')

# Filter halos and subhalos by min mass conditions
subhalo_conditions = [('mstr', '>', data_loader_dict['subhalo_min_mass'])]  # solar_masses
subhalo_context_filtered = fetcher_illustris.filter_subhalos(subhalo_context, subhalo_conditions)
filtered_subhalo_ids = set(subhalo_context_filtered.data['subhalo_id'])
print(f'Illustris filtered subhalos by mass: {subhalo_context_filtered.data["subhalo_id"].nunique()}')

# Apply both conditions of mass
halo_subhalos_df = pd.DataFrame(data=[], columns=['halo_id', 'subhalo_counts'])
util_subhalo_ids = []
for id_halo in halo_context_filtered.halo_id:
    # logger.debug(f"Loading data for halo_id: {id_halo}")
    halo_df = halo_context_filtered.get_item(id_halo)
    subhalo_df = subhalo_context_filtered.get_items_by_halo(id_halo)
    util_subhalo_ids.extend(subhalo_df['subhalo_id'])
    tmp = pd.DataFrame(data=[[id_halo, len(subhalo_df)]], columns=['halo_id', 'subhalo_counts'])
    halo_subhalos_df = pd.concat([halo_subhalos_df, tmp])
print(f'From filtered halos, we obtain {np.sum(halo_subhalos_df["subhalo_counts"])} util subhalos')

# Analyze particle data
for i, halo in halo_context_filtered.data.iterrows():
    try:
        subhalo_id = halo.fsub
        halo_id = halo.halo_id
        # Those idx comes directly from halo api
        subhalo_particles = fetcher_illustris.get_p_subhalo(subhalo_id=subhalo_id, group_number=halo_id)
    except Exception as e:
        print(e)
        continue

# Confirm non-accepted halos
non_accepted_halo_ids = total_halo_ids - filtered_halo_ids
halo_context = fetcher_illustris.get_halos()
min_halo_mass = fetcher_illustris.halo_min_mass

non_accepted_halos = halo_context.data[halo_context.data['halo_id'].isin(non_accepted_halo_ids)]
tmp = non_accepted_halos[non_accepted_halos['m200'] >= min_halo_mass]
if len(tmp) > 0:
    raise Exception(f'There is something wrong')
print(f'There are {len(non_accepted_halo_ids)} halos that do not fulfill minimum mass condition')

# Confirm non-accepted subhalos
non_accepted_subhalo_ids = total_subhalo_ids - filtered_subhalo_ids
subhalo_context = fetcher_illustris.get_subhalos()
min_subhalo_mass = fetcher_illustris.subhalo_min_mass
non_accepted_subhalos = subhalo_context.data[subhalo_context.data['subhalo_id'].isin(non_accepted_subhalo_ids)]
tmp = non_accepted_subhalos[non_accepted_subhalos['mstr'] >= min_subhalo_mass / fetcher_illustris.H]
if len(tmp) > 0:
    raise Exception(f'There is something wrong')
print(f'There are {len(non_accepted_subhalo_ids)} subhalos that do not fulfill minimum mass condition')

