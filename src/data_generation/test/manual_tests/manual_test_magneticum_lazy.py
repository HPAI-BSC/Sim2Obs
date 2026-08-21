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
    print(f'type: {type(halo_context)}')

    total_rows = len(halo_context.data)
    print("Total rows:", total_rows)

    subhalo_context = fetcher_magneticum.get_subhalos()
    print(f'type: {type(subhalo_context)}')

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

    print('-------- First Halo & Its Primary Subhalo --------')

    # "first" = first row as loaded (row order from the halo pipeline, not
    # necessarily lowest halo_id). Use sorted(halo_context.halo_id)[0] instead
    # if you specifically want the smallest halo_id.
    first_halo_id = halo_context.halo_id[0]
    first_halo = halo_context.get_halo_by_id(first_halo_id)

    print(f"First halo_id: {first_halo_id}")
    print("Halo attributes:")
    for key, value in vars(first_halo).items():
        print(f"  {key}: {value}")

    # Halo.fsub IS the primary subhalo's subhalo_id (see Halo docstring:
    # "the index of the primary subhalo" -- matches how _get_subhalos()
    # assigns subhalo_id via a running global offset).
    primary_subhalo_id = first_halo.fsub
    print(f"\nPrimary subhalo_id (halo.fsub): {primary_subhalo_id}")

    # Filter directly on the dataset -- avoid LazySubhaloContext.get_subhalo_by_id(),
    # which filters on 'parent_id', a field that doesn't exist in the subhalo
    # catalog schema (it's 'id_parents' there). subhalo_id alone is enough
    # since it's globally unique for this fetcher.
    bucket = first_halo_id % 128
    primary_subhalo_row = subhalo_context.dataset.to_table(
        filter=(ds.field("parent_bucket_data") == bucket) & (ds.field("subhalo_id") == primary_subhalo_id)
    ).to_pandas()

    if primary_subhalo_row.empty:
        print(f"WARNING: no subhalo row found for subhalo_id={primary_subhalo_id} -- "
              f"this halo may have zero subhalos, or fsub doesn't correspond 1:1 "
              f"to subhalo_id for this catalog (worth double-checking).")
    else:
        print("Primary subhalo attributes:")
        row = primary_subhalo_row.iloc[0]
        for col in primary_subhalo_row.columns:
            print(f"  {col}: {row[col]}")


if __name__ == "__main__":
    # TODO DEBUG
    magneticum_particle_test()
