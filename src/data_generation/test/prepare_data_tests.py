import os
import pickle

from dotenv import load_dotenv
from hydra import initialize, compose

from src.data_generation.src.data_fetcher import DataFetcherFactory


load_dotenv('../.env')

with initialize(config_path="configs"):
    # Initialize the cfg_illustris and cfg_eagle
    cfg_illustris = compose(config_name="data_generation.yaml")
    cfg_eagle = compose(config_name="data_generation.yaml")

    # Load the local config for paths and assign them
    local_config = compose(config_name="paths/local.yaml")
    cfg_illustris.paths = local_config.paths
    cfg_eagle.paths = local_config.paths

    # Dynamic catalog loading for both Illustris and Eagle
    cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
    cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

os.environ['PROJECT_ROOT'] = '../../..'

required_keys = ['G']
data_loader_dict = next((step for step in cfg_illustris.pipeline if step.name == 'data_loader'), None)
attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
fetcher_illustris = DataFetcherFactory.get_data_fetcher(cfg_illustris.catalog, **attr_dict)
fetcher_eagle = DataFetcherFactory.get_data_fetcher(cfg_eagle.catalog, **attr_dict)


def save_to_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# Illustris objects
halo_context = fetcher_illustris.get_halos()
halo_context = halo_context[:5]

save_to_pickle('data/illustris/halo_context_0.pkl', halo_context)
halo = halo_context.get_halo_by_id(halo_id=0)
save_to_pickle('data/illustris/halo_0.pkl', halo)

subhalo_context = fetcher_illustris.get_subhalos()
subhalo_context = subhalo_context[:5]
save_to_pickle('data/illustris/subhalo_context_0.pkl', subhalo_context)
subhalo = subhalo_context.get_subhalo_by_id(subhalo_id=0)
save_to_pickle('data/illustris/subhalo_0.pkl', subhalo)

subhalo_particles_1 = fetcher_illustris.get_p_subhalo(subhalo_id=0)
save_to_pickle('data/illustris/subhalo_particles_0.pkl', subhalo_particles_1)

# Eagle objects
halo_context_2 = fetcher_eagle.get_halos()
halo_context_2 = halo_context_2[:5]
save_to_pickle('data/eagle/halo_context_0.pkl', halo_context_2)
halo_data_2 = halo_context_2.get_halo_by_id(halo_id=halo_context_2.halo_id[0])
save_to_pickle('data/eagle/halo_0.pkl', halo_data_2)

subhalo_context_2 = fetcher_eagle.get_subhalos()
if 'id_parents' in subhalo_context_2.data.index.names:
    subhalo_context_2.data = subhalo_context_2.data.reset_index(drop=True)
subhalo_context_2.data = subhalo_context_2.data.sort_values(by=["id_parents"])
subhalo_context_2.data.set_index(["id_parents", "subhalo_id"], inplace=True, drop=False)

subhalo_context_2 = subhalo_context_2[:5]
save_to_pickle('data/eagle/subhalo_context_0.pkl', subhalo_context_2)
subhalo_data_2 = subhalo_context_2.get_subhalo_by_id(subhalo_id=subhalo_context_2.subhalo_id[0])
save_to_pickle('data/eagle/subhalo_0.pkl', subhalo_data_2)

subhalo_particles_2 = fetcher_eagle.get_p_subhalo(subhalo_id=subhalo_context_2.subhalo_id[0], group_number=2)
save_to_pickle('data/eagle/subhalo_particles_0.pkl', subhalo_particles_2)