import os
import pickle
import eagleSqlTools as sql
import illustris_python as il
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.subhalo import SubhaloContext, SubHalo
from src.data_generation.src.utils import get_connection

load_dotenv('../.env')

with initialize(config_path="../../../configs"):
    cfg_illustris = compose(config_name="data_generation.yaml")
    cfg_eagle = compose(config_name="data_generation.yaml")
    local_config = compose(config_name="paths/local.yaml")
    cfg_eagle.paths = local_config.paths
    cfg_illustris.paths = local_config.paths
    cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
    cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
    cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'
    cfg_eagle.catalog.simulation_code = 'RefL0050N0752'

os.environ['PROJECT_ROOT'] = '../../..'

required_keys = ['subhalo_min_mass', 'halo_min_mass', 'solar_mass_convention', 'G', 'H']
data_loader_dict = next((step for step in cfg_illustris.pipeline if step.name == 'data_loader'), None)
attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
fetcher_illustris = DataFetcherFactory.get_data_fetcher(cfg_illustris.catalog, **attr_dict)
fetcher_eagle = DataFetcherFactory.get_data_fetcher(cfg_eagle.catalog, **attr_dict)


# Illustris objects
halos = il.groupcat.loadHalos(cfg_illustris.catalog.path + '/output', cfg_illustris.catalog.snapnum,
                              fields=["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"])

halo_context = fetcher_illustris.get_halos()
conditions = [('m200', '>', fetcher_illustris.halo_min_mass * fetcher_illustris.H)]
halo_context = fetcher_illustris.filter_halos(halo_context, conditions)

halo = halo_context.get_item(halo_id=0)

with open('data/illustris/halo_0.pkl', 'wb') as f:
    pickle.dump(halo, f)

with open('data/illustris/halo_context_0.pkl', 'wb') as f:
    pickle.dump(halo_context, f)

# Subhalos
subhalo_context = fetcher_illustris.get_subhalos()
conditions = [('mstr', '>', fetcher_illustris.subhalo_min_mass)]
subhalo_context = fetcher_illustris.filter_subhalos(subhalo_context, conditions)
subhalo = subhalo_context.get_item(subhalo_id=0)

with open('data/illustris/subhalo_any_0.pkl', 'wb') as f:
    pickle.dump(subhalo, f)

with open('data/illustris/subhalo_context_0.pkl', 'wb') as f:
    pickle.dump(subhalo_context, f)

subhalo_particles_1 = fetcher_illustris.get_p_subhalo(subhalo_id=0)
with open('data/illustris/subhalo_particles_0.pkl', 'wb') as f:
    pickle.dump(subhalo_particles_1, f)

# Eagle objects
halo_context_2 = fetcher_eagle.get_halos()
halo_conditions = [('m200', '>', data_loader_dict['halo_min_mass'] / data_loader_dict['H'])]  # M⊙
halo_context_2 = fetcher_eagle.filter_halos(halo_context_2, halo_conditions)
halo_data_2 = halo_context_2.get_item(halo_id=halo_context_2.halo_id[0])

with open('data/eagle/halo_0.pkl', 'wb') as f:
    pickle.dump(halo_data_2, f)

with open('data/eagle/halo_context_0.pkl', 'wb') as f:
    pickle.dump(halo_context_2, f)

subhalo_context_2 = fetcher_eagle.get_subhalos()
subhalo_conditions = [('mstr', '>', data_loader_dict['subhalo_min_mass'])]  # M⊙
subhalo_context_2 = fetcher_eagle.filter_subhalos(subhalo_context_2, subhalo_conditions)
subhalo_data_2 = subhalo_context_2.get_item(subhalo_id=subhalo_context_2.subhalo_id[0])

with open('data/eagle/subhalo_any_0.pkl', 'wb') as f:
    pickle.dump(subhalo_data_2, f)

with open('data/eagle/subhalo_context_0.pkl', 'wb') as f:
    pickle.dump(subhalo_context_2, f)

subhalo_particles_2 = fetcher_eagle.get_p_subhalo(subhalo_id=subhalo_context_2.subhalo_id[0], group_number=2)
with open('data/eagle/subhalo_particles_0.pkl', 'wb') as f:
    pickle.dump(subhalo_particles_2, f)