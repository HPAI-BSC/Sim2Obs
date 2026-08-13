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


    '''
    total_rows = subhalo_particles.dataset.count_rows()
    print("Total rows:", total_rows)
    print(vars(subhalo_particles).keys())
    table = subhalo_particles.dataset.to_table(columns=["parent_bucket_data"])
    buckets = pc.unique(table["parent_bucket_data"])
    print("Num buckets:", len(buckets))
    print("Buckets:", buckets.to_pylist())
    '''
    '''
    subhalos_parquet = f"{fetcher_magneticum.catalog.path}/subhalos_parquet"
    # Read just the subhalo_id column from bucket 0, fast, no full load
    dataset = ds.dataset(subhalos_parquet, format='parquet', partitioning='hive')
    subhalo_ids = (
        dataset
        .to_table(columns=['subhalo_id', 'parent_id', 'parent_bucket'])
        .to_pandas()
        .drop_duplicates('subhalo_id')
        .sort_values('subhalo_id')
    )

    print(f"Total unique subhalo_ids: {len(subhalo_ids)}")
    '''
    # Pick a valid one
    valid_subhalo_id = -1 #subhalo_ids['subhalo_id'].iloc[0]
    print(f"\nTesting with subhalo_id={valid_subhalo_id}")

    particles_parquet = f"{fetcher_magneticum.catalog.path}/particles_parquet"
    subhalo_particles_ = fetcher_magneticum.get_p_subhalo(subhalo_id=valid_subhalo_id)
    print(type(subhalo_particles_))
    print(subhalo_particles_.star_p.size)
    print(type(subhalo_particles_.star_p))
    print(subhalo_particles_.star_p)


    result = FeatureComputer.compute('stellar_half_mass_radius', particles=subhalo_particles_)
    print(f'shmr: {result}')
    tracemalloc.stop()


if __name__ == "__main__":
    # TODO DEBUG
    magneticum_particle_test()
