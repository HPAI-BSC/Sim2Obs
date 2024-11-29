import os
import pickle
import eagleSqlTools as sql
import illustris_python as il
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import initialize, compose

from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.subhalo import SubhaloContext, SubHalo, SubhaloParticles
from src.data_generation.src.utils import get_connection, read_dataset

load_dotenv('../.env')

# EAGLE TESTS WITH GROUPNUMBER=2 AND SUBGROUPNUMBER=0 (SUBHALO_ID)
# Taking the first n particles for both simulations

n = 100

# Initialize Hydra and load configuration
with initialize(config_path="../../../configs"):
    cfg_illustris = compose(config_name="data_generation.yaml")
    cfg_eagle = compose(config_name="data_generation.yaml")

with initialize(config_path="../../../configs"):
    cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
    cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
    cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'

os.environ['PROJECT_ROOT'] = '../../..'

with open('data/illustris/subhalo_context_0.pkl', 'rb') as f:
    subhalo_context = pickle.load(f)
    halo_id, subhalo_id = tuple(subhalo_context.data[['id_parents', 'subhalo_id']].iloc[0, :])
    print()

fields = ["Coordinates", "Velocities", "Masses"]
path = cfg_illustris.catalog.path + '/output'
star_p = il.snapshot.loadSubhalo(path, cfg_illustris.catalog.snapnum, subhalo_id, "star", fields=fields)

if len(star_p) > 0:
    p = star_p["Coordinates"][:n]
    v = (star_p["Velocities"] * np.sqrt(cfg_illustris.catalog.scalf))[:n] # 𝑘𝑚𝑎‾√/𝑠 -> kms-1
    m = (star_p["Masses"] * cfg_illustris.solar_mass_convention)[:n]
    subhalo_particles = SubhaloParticles(m, p, v, cfg_illustris.catalog.scalf)
    with open('data/illustris/subhalo_particles_0.pkl', 'wb') as f:
        pickle.dump(subhalo_particles, f)


with open('data/eagle/subhalo_context_0.pkl', 'rb') as f:
    subhalo_context = pickle.load(f)
    group_number, subhalo_id = tuple(subhalo_context.data[['group_number', 'subhalo_id']].iloc[0, :])
    print()

# Eagle
itype = 4  # PartType
path = cfg_eagle.catalog.path
if not os.path.exists(path):
    raise Exception(f'The path {path} does not exist.')

sgns = read_dataset(itype, 'SubGroupNumber', path)

data = {
    'mass': read_dataset(itype, 'Mass', path),
    'coords': read_dataset(itype, 'Coordinates', path),
    'velocity': read_dataset(itype, 'Velocity', path),
    'GroupNumber': read_dataset(itype, 'GroupNumber', path),
    'SubGroupNumber': read_dataset(itype,'SubGroupNumber', path)
}

data['coords'] = [data['coords'][i] for i in range(data['coords'].shape[0])]
data['velocity'] = [data['velocity'][i] for i in range(data['velocity'].shape[0])]

particles_no_group = 2**30

tmp = pd.DataFrame(data)
tmp = tmp.sort_values(by=['GroupNumber','SubGroupNumber'])
unique_halo_ids = tmp['SubGroupNumber'].unique()
tmp_filtered = tmp[(tmp['SubGroupNumber'] == subhalo_id) & (tmp['GroupNumber'] == group_number)]
tmp_filtered = tmp_filtered.head(n)
subhalo_particles = SubhaloParticles(tmp_filtered['mass'], tmp_filtered['coords'], tmp_filtered['velocity'],
                                     cfg_eagle.catalog.scalf)
with open('data/eagle/subhalo_particles_0.pkl', 'wb') as f:
    pickle.dump(subhalo_particles, f)