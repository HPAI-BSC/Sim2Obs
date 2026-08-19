import os
import tracemalloc
import gc

import pandas as pd
import pyarrow.compute as pc
from src.data_generation.src.data_fetcher import DataFetcherFactory

from hydra import initialize, compose
import pyarrow.parquet as pq
import pyarrow.dataset as ds

from src.data_generation.src.compute_features_stage import compute_features_stage


def magneticum_compute_features_test():
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

    # Same sanity-check prints as before -- still useful to confirm the
    # catalog looks as expected before spending time on feature computation.
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

    print('-------- Compute Features Stage (cluster smoke test) --------')
    # First real run against cluster data -- deliberately small and
    # isolated:
    #   - halo_ids restricted to the first 10 halos, so this doesn't wait
    #     on (or read particles for) the full catalog.
    #   - write_to_new_bucket=True so this writes to a completely separate
    #     output tree (features_root + '_new'). The real, already-
    #     materialized features_root is never opened, read, or touched --
    #     if anything about the feature computation is still wrong, this
    #     can't corrupt production output.
    #   - force_recompute=True since this is a first cluster run and there
    #     is no prior "_DONE" marker to respect yet; safe to leave True on
    #     reruns too since this debug output is disposable and small.
    #
    # IMPORTANT: debug ids are drawn from HaloContext.halo_id, not from
    # subhalo_context's 'id_parents' column. The halo and subhalo catalogs
    # apply INDEPENDENT filters (halo_conditions vs subhalo_conditions),
    # so an id_parents value from the subhalo table is not guaranteed to
    # exist in HaloContext -- and "first N unique id_parents" from a
    # bucket-partitioned parquet scan isn't a meaningful "first 10 halos"
    # selection anyway (it reflects ingestion/partition order, not any
    # sorted id). Also intersect with subhalo_context's id_parents so the
    # selected halos are guaranteed to actually have subhalos to process
    # -- otherwise a valid-but-subhalo-less halo could again produce zero
    # output rows.
    halo_context = fetcher_magneticum.get_halos()
    valid_halo_ids = set(halo_context.halo_id)
    subhalo_ids_present = set(unique_ids.to_pylist())
    candidate_ids = sorted(valid_halo_ids & subhalo_ids_present)

    if not candidate_ids:
        raise RuntimeError(
            "No halo_id is present in BOTH HaloContext and the subhalo catalog's "
            "id_parents -- halo_conditions and subhalo_conditions filters may be "
            "fully disjoint for this catalog. Check the catalog config."
        )

    debug_halo_ids = candidate_ids[:10]
    print(f"Restricting to halo_ids={debug_halo_ids} "
          f"(drawn from {len(candidate_ids)} ids valid in both halo and subhalo catalogs)")

    compute_features_stage(
        fetcher=fetcher_magneticum,
        feature_names=['stellar_half_mass_radius'],
        cfg=cfg_magneticum,
        force_recompute=True,
        halo_ids=debug_halo_ids,
        write_to_new_bucket=True,
    )

    # Spot-check: read back whatever the stage just wrote to the SEPARATE
    # debug output tree (features_root + '_new'), not the production one.
    dataset_dir = cfg_magneticum.paths.datasets_dir
    catalog_name = cfg_magneticum.catalog.path.replace(dataset_dir, '').replace('/', '_')
    features_root = f'{dataset_dir}/../features/{catalog_name}_subhalo_features_new'

    if not os.path.exists(features_root):
        raise RuntimeError(
            f"{features_root} was never created -- compute_features_stage produced "
            f"zero output rows for halo_ids={debug_halo_ids}. Check the WARNING/ERROR "
            f"lines above (e.g. 'Halo ID ... not found', fetch failures, or every "
            f"subhalo coming back with zero star particles)."
        )

    features_ds = ds.dataset(features_root, format="parquet")
    features_table = features_ds.to_table()
    print(f"Total feature rows written: {features_table.num_rows}")
    print(f"(expected: one row per subhalo across the {len(debug_halo_ids)} requested halos)")
    result_df = features_table.to_pandas()
    print(result_df.sort_values(['id_parents', 'subhalo_id']))

    # Every requested halo should actually show up -- flags anything that
    # silently failed to resolve (e.g. a halo_id not found in the catalog).
    missing = set(debug_halo_ids) - set(result_df['id_parents'].unique())
    if missing:
        print(f"WARNING: no output rows for halo_ids={missing} -- check logs for errors on these halos.")

    print('-------- Verify: read back new bucket, inspect first 5 subhalos --------')
    # Re-open the new bucket tree fresh (not reusing features_table above)
    # so this check reflects exactly what's actually on disk, independent
    # of anything still held in memory from the run itself.
    verify_ds = ds.dataset(features_root, format="parquet")
    verify_df = verify_ds.to_table().to_pandas().sort_values(['id_parents', 'subhalo_id']).reset_index(drop=True)

    n_check = 5
    sample = verify_df.head(n_check)
    print(f"Columns present: {list(verify_df.columns)}")

    if 'stellar_half_mass_radius' not in verify_df.columns:
        print("WARNING: 'stellar_half_mass_radius' column is missing entirely from the written bucket.")
    else:
        for _, row in sample.iterrows():
            print(f"id_parents={row['id_parents']}, subhalo_id={row['subhalo_id']}, "
                  f"stellar_half_mass_radius={row['stellar_half_mass_radius']}")

        n_nan = sample['stellar_half_mass_radius'].isna().sum()
        if n_nan == n_check:
            print(f"WARNING: all {n_check} sampled subhalos have NaN stellar_half_mass_radius -- "
                  f"feature computation may be silently failing (check logs above for errors).")
        elif n_nan > 0:
            print(f"Note: {n_nan}/{n_check} sampled subhalos have NaN stellar_half_mass_radius "
                  f"(expected for subhalos with no/zero-mass star particles).")

    tracemalloc.stop()


if __name__ == "__main__":
    magneticum_compute_features_test()