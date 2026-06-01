import tracemalloc
import gc

import pandas as pd
#from rootutils import rootutils
import pyarrow.compute as pc
from src.data_generation.src.data_fetcher import DataFetcherFactory

#rootutils.setup_root(__file__, pythonpath=True)

from hydra import initialize, compose

from src.data_generation.src.halo import Halo


def magneticum_particle_test():
    with initialize(config_path="'../../../../../../configs"):
        cfg_magneticum = compose(config_name="data_generation.yaml")

        # Load the local config for paths and assign them
        local_config = compose(config_name="paths/gpfs.yaml")
        cfg_magneticum.paths = local_config.paths
        cfg_magneticum.catalog = compose("catalog/magneticum.yaml")['catalog']

    required_keys = ['G']
    data_loader_dict = next((step for step in cfg_magneticum.pipeline if step.name == 'data_loader'), None)
    attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
    fetcher_magneticum = DataFetcherFactory.get_data_fetcher(cfg_magneticum.catalog, **attr_dict)

    tracemalloc.start()
    '''
    halo_context = fetcher_magneticum.get_halos()
    subhalo_context = fetcher_magneticum.get_subhalos()

    total_rows = subhalo_context.dataset.count_rows()
    print("Total rows:", total_rows)

    table = subhalo_context.dataset.to_table(columns=["parent_bucket_data"])
    buckets = pc.unique(table["parent_bucket_data"])
    print("Num buckets:", len(buckets))
    print("Buckets:", buckets.to_pylist())

    table = subhalo_context.dataset.to_table(columns=["id_parents"])
    unique_ids = pc.unique(table["id_parents"])
    print("Unique halos stored:", len(unique_ids))

    print('-------- Subhalo Particles --------')
    mock_halo = {'halo_id': 0, 'm200': 26000000, 'r200': 744.56, 'fsub': 0, 'gpos': [200, 200, 200]}
    halo = Halo(**mock_halo)
    subhalo_particles = fetcher_magneticum.get_p_halo(halo=halo)
    total_rows = subhalo_particles.dataset.count_rows()
    print("Total rows:", total_rows)
    table = subhalo_particles.dataset.to_table(columns=["parent_bucket_data"])
    buckets = pc.unique(table["parent_bucket_data"])
    print("Num buckets:", len(buckets))
    print("Buckets:", buckets.to_pylist())
    '''
    subhalo_particles_0 = fetcher_magneticum.get_p_halo(halo=halo)
    print(type(subhalo_particles_0))
    print(subhalo_particles_0)
    tracemalloc.stop()


if __name__ == "__main__":
    # TODO DEBUG
    magneticum_particle_test()
