import os
from dotenv import load_dotenv
from hydra import initialize, compose
import pandas as pd

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.utils import read_dataset

load_dotenv('../.env')


if __name__ == '__main__':
    cfg_eagle = None

    with initialize(config_path="../../../configs"):
        cfg_eagle = compose(config_name="data_generation.yaml")
        local_config = compose(config_name="paths/local.yaml")
        cfg_eagle.paths = local_config.paths
        cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
        cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
        cfg_eagle.catalog.path = '../../../local/data/eagle/RefL0050N0752'

    os.environ['PROJECT_ROOT'] = '../../..'

    required_keys = ['subhalo_min_mass', 'halo_min_mass', 'solar_mass_convention', 'G', 'H']
    data_loader_dict = next((step for step in cfg_eagle.pipeline if step.name == 'data_loader'), None)
    attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
    fetcher_eagle = DataFetcherFactory.get_data_fetcher(cfg_eagle.catalog, **attr_dict)

    halo_context = fetcher_eagle.get_halos()
    print(f'total halos: {halo_context.data["halo_id"].nunique()}')
    subhalo_context = fetcher_eagle.get_subhalos()
    print(f'total subhalos: {subhalo_context.data["subhalo_id"].nunique()}')

    subhalos_with_particles = 0
    for index, halo in halo_context.data.iterrows():
        try:
            halo_id = halo.halo_id
            group_num = halo.group_number
            subhalo_id = halo.fsub
            linked_subhalo = subhalo_context.data[(subhalo_context.data['group_number'] == group_num) &
                                                  (subhalo_context.data['subhalo_id'] == subhalo_id) &
                                                  (subhalo_context.data['id_parents'] == halo_id)]
            subhalo_particles = fetcher_eagle.get_p_subhalo(subhalo_id=subhalo_id, group_number=group_num)
            if len(subhalo_particles.m) > 0:
                subhalos_with_particles += 1
            print(f'Halo: {halo_id}, Group_number: {group_num} with {len(linked_subhalo)} subhalos linked and'
                  f' {len(subhalo_particles.m)} particles')
        except Exception as e:
            continue
    print(f'Subhalos with particles: {subhalos_with_particles}')

    # Retrieve all the particle data
    path = cfg_eagle.catalog.path
    if not os.path.exists(path):
        print(f'The path {path} does not exist.')

    itype = 4
    data = {
        'mass': read_dataset(itype, 'Mass', path),
        'coords': read_dataset(itype, 'Coordinates', path),
        'velocity': read_dataset(itype, 'Velocity', path),
        'GroupNumber': read_dataset(itype, 'GroupNumber', path),
        'SubGroupNumber': read_dataset(itype, 'SubGroupNumber', path)
    }
    data['coords'] = [data['coords'][i] for i in range(data['coords'].shape[0])]
    data['velocity'] = [data['velocity'][i] for i in range(data['velocity'].shape[0])]
    particle_df = pd.DataFrame(data)

    # Analyze subgroupnum and groupnum
    tmp = particle_df[['GroupNumber', 'SubGroupNumber']]
    tmp = tmp.drop_duplicates()
    tmp = tmp.sort_values(by=['GroupNumber', 'SubGroupNumber'])
    list_group_num = tmp['GroupNumber'].unique()
    print(list_group_num)
    list_subgroup_num = tmp['SubGroupNumber'].unique()
    print(list_subgroup_num)

    # Retrieve and marge halo - subhalo data
    halo_context.data = halo_context.data.rename(columns={'halo_id': 'id_parents', 'fsub':'subhalo_id'})
    halo_context.data = subhalo_context.data.set_index(['id_parents', 'subhalo_id'])
    subhalo_context.data = subhalo_context.data.set_index(['id_parents', 'subhalo_id'])
    eagle_dataset = halo_context.data.join(subhalo_context.data, on=['id_parents', 'subhalo_id'], how='outer',
                                           lsuffix='_halo', rsuffix='_subhalo')

    eagle_dataset = eagle_dataset.reset_index()
    eagle_dataset = eagle_dataset.sort_values(by=['id_parents', 'subhalo_id'])
    eagle_dataset.to_csv(f'eagle_halo_subhalo_data_{cfg_eagle.catalog.simulation_code}.csv')