import os.path

import pandas as pd
import illustris_python as il
from hydra import initialize, compose

print(os.path.abspath('.'))

with initialize(config_path="../../../../configs"):
    cfg_tngcluster = compose(config_name="data_generation.yaml")

path = cfg_tngcluster.catalog.path
print(path)
halo_fields = ['Group_M_Crit200', 'Group_R_Crit200', 'GroupFirstSub']
halos = il.groupcat.loadHalos(path, 84, fields=halo_fields)
df = pd.DataFrame(halos)
print(f'Illustris (raw) halos: {len(df)}')

h = 0.6774
mass_filter_final_units = 1e13 * h   # Msun/h

df['Group_M_Crit200'] = df['Group_M_Crit200'] * 1e10 # from 10^10 Msun/h to Msun/h
filtered_df = df[df['Group_M_Crit200'] > mass_filter_final_units]
print(len(filtered_df))

subhalo_fields = ['SubhaloMassType', 'SubhaloGrNr']
subhalos = il.groupcat.loadSubhalos(path, 84, fields=subhalo_fields)
subhalos['SubhaloMassType'] = subhalos['SubhaloMassType'][:, 4]
subhalos_df = pd.DataFrame(subhalos)
subhalos_df['SubhaloMassType'] = subhalos_df['SubhaloMassType'] * 1e10 # from 10^10 Msun/h to Msun/h

mass_filter = 1e10 # Msun
mass_filter_units = 1e10 * h # Msun / h

filtered_subhalos = subhalos_df[subhalos_df['SubhaloMassType'] > mass_filter_units]
print(len(filtered_subhalos))

