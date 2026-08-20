import tracemalloc
import gc

import pandas as pd
#from rootutils import rootutils
import pyarrow.compute as pc
from src.data_generation.src.data_fetcher import DataFetcherFactory

#rootutils.setup_root(__file__, pythonpath=True)

from hydra import initialize, compose
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from src.data_generation.src.feature_computer import FeatureComputer
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

    halo_context = fetcher_magneticum.get_halos()
    total_rows = halo_context.dataset.count_rows()
    print("Total rows:", total_rows)

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
    # Pick a valid one
    valid_subhalo_id = -1 #subhalo_ids['subhalo_id'].iloc[0]
    print(f"\nTesting with subhalo_id={valid_subhalo_id}")

    subhalo_particles_ = fetcher_magneticum.get_p_subhalo(subhalo_id=valid_subhalo_id)
    print(type(subhalo_particles_))
    print(subhalo_particles_.star_p.size)
    print(type(subhalo_particles_.star_p))
    print(subhalo_particles_.star_p)


if __name__ == "__main__":
    # TODO DEBUG
    magneticum_particle_test()
