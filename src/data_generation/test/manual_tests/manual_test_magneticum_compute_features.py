import os
import tracemalloc

import numpy as np
import pandas as pd
import pyarrow.compute as pc
from src.data_generation.src.data_fetcher import DataFetcherFactory

from hydra import initialize, compose
import pyarrow.dataset as ds


from src.data_generation.src.features_dataset.compute_features import _load_bucket_particle_groups, \
    compute_features_stage
from src.data_generation.src.subhalo import SubhaloContext


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

    print('-------- Verify preprocessing preconditions before computing SHMR --------')
    # These are asserted directly against the REAL production code path
    # (SubhaloContext.preprocess / SubhaloParticles.preprocess -- the same
    # functions compute_features_stage itself calls), not a
    # reimplementation, so this actually validates what the pipeline does
    # rather than what we assume it does. Checked assumptions:
    #   1. Subhalo position (spos) has been centered relative to its halo.
    #   2. The subhalo's own star particles have been centered relative to
    #      the SAME halo.
    #   3. Minimum-image PBC correction was applied to both -- the direct
    #      consequence is that every corrected position must satisfy
    #      |pos - halo.gpos| <= lbox/2. If that bound is violated, PBC
    #      correction did not actually run.
    # Note: this preprocessing is never persisted (matches
    # compute_features_stage's in-memory-only preprocessing) -- this block
    # only verifies it happens, it does not write anything new.
    lbox = cfg_magneticum.catalog.get('Lbox', None)
    if lbox is None:
        print("WARNING: cfg.catalog.get('Lbox') is None -- PBC bound checks below will be skipped "
              "(no box size to check against). This itself is worth investigating.")

    # Flagged (not raised) so one problem subhalo doesn't abort the whole
    # verification pass -- collected for inspection afterward instead.
    pbc_violations = []

    n_check = 5
    checked = 0
    for hid in candidate_ids:
        if checked >= n_check:
            break

        halo = halo_context.get_halo_by_id(hid)
        bucket = hid % 128

        subhalo_rows = subhalo_context.dataset.to_table(
            columns=['id_parents', 'subhalo_id', 'spos'],
            filter=ds.field('id_parents') == hid
        ).to_pandas()
        if subhalo_rows.empty:
            continue

        particle_groups = _load_bucket_particle_groups(fetcher_magneticum, bucket)

        # Find a subhalo of this halo that actually has star particles --
        # otherwise there's nothing meaningful to centre.
        target_row, target_particles = None, None
        for _, r in subhalo_rows.iterrows():
            sp = particle_groups.get(int(r['subhalo_id']))
            if sp is not None and hasattr(sp, 'star_m') and len(sp.star_m) > 0:
                target_row, target_particles = r, sp
                break
        if target_row is None:
            continue

        checked += 1
        sid = target_row['subhalo_id']
        print(f"\n=== Halo {hid} (gpos={halo.gpos}), subhalo {sid} "
              f"({len(target_particles.star_m)} star particles) ===")

        # --- Assumption 1 + PBC: subhalo position centered on its halo ---
        raw_spos = np.asarray(target_row['spos'], dtype=float)
        single_ctx = SubhaloContext.from_dataframe(pd.DataFrame([target_row]))
        single_ctx.preprocess(halo, lbox)
        corrected_spos = np.asarray(single_ctx.data.iloc[0]['spos'], dtype=float)

        spos_changed = not np.allclose(raw_spos, corrected_spos)
        print(f"  spos: raw(absolute)={raw_spos} -> corrected(HALO-RELATIVE)={corrected_spos} "
              f"(changed: {spos_changed})")

        if lbox is not None:
            # IMPORTANT: preprocess() already returns a halo-RELATIVE,
            # PBC-corrected delta (origin = halo center), not a re-wrapped
            # absolute position. Do NOT subtract halo.gpos again here --
            # doing so double-subtracts the halo center and produces a
            # bogus giant "violation" even when preprocessing is correct.
            dist = np.linalg.norm(corrected_spos)
            print(f"  |corrected_spos| (relative to halo center) = {dist:.4f}  "
                  f"(bound: lbox/2 = {lbox / 2:.4f})")
            if dist > lbox / 2 + 1e-6:
                pbc_violations.append({
                    'halo_id': hid, 'subhalo_id': sid, 'kind': 'spos', 'distance': float(dist),
                })
                print(f"  \u26a0 FLAGGED: spos PBC bound violated for halo {hid}, subhalo {sid} "
                      f"(dist={dist:.4f} > lbox/2={lbox / 2:.4f})")

        # --- Assumption 2 + PBC: particle positions centered on the SAME halo ---
        raw_star_p_sample = target_particles.star_p[0].copy()
        target_particles.preprocess(halo, lbox)
        corrected_star_p = target_particles.star_p

        p_changed = not np.allclose(raw_star_p_sample, corrected_star_p[0])
        print(f"  star_p[0]: raw(absolute)={raw_star_p_sample} -> corrected(HALO-RELATIVE)={corrected_star_p[0]} "
              f"(changed: {p_changed})")

        if lbox is not None:
            # Already centered on halo.gpos by preprocess(), so distance
            # from the origin IS distance from the halo center.
            dists = np.linalg.norm(corrected_star_p, axis=1)
            max_dist = float(dists.max())
            print(f"  particle distances from halo center: min={dists.min():.4f}, "
                  f"max={max_dist:.4f}  (bound: lbox/2 = {lbox / 2:.4f})")
            if max_dist > lbox / 2 + 1e-6:
                n_bad = int(np.sum(dists > lbox / 2 + 1e-6))
                pbc_violations.append({
                    'halo_id': hid, 'subhalo_id': sid, 'kind': 'star_p',
                    'max_distance': max_dist, 'n_particles_over_bound': n_bad,
                    'n_particles_total': int(len(dists)),
                })
                print(f"  \u26a0 FLAGGED: {n_bad}/{len(dists)} particles violate PBC bound for "
                      f"halo {hid}, subhalo {sid} (max dist={max_dist:.4f} > lbox/2={lbox / 2:.4f})")

    if checked == 0:
        print("WARNING: could not verify preprocessing -- no subhalo with star particles "
              "found among the first candidate halos. Preconditions UNVERIFIED.")
    elif pbc_violations:
        print(f"\n\u26a0 {len(pbc_violations)} PBC violation(s) flagged out of {checked} subhalo(s) "
              f"checked -- NOT treated as fatal, pipeline continues. Logged for later investigation:")
        for v in pbc_violations:
            print(f"   {v}")

        # Persist alongside the debug output so this survives past the
        # console/job log, since "manage later" implies a durable record.
        violations_path = os.path.join(
            f'{cfg_magneticum.paths.datasets_dir}/../features',
            f"{cfg_magneticum.catalog.path.replace(cfg_magneticum.paths.datasets_dir, '').replace('/', '_')}"
            f"_subhalo_features_new",
            '_pbc_violations.json',
        )
        os.makedirs(os.path.dirname(violations_path), exist_ok=True)
        import json
        with open(violations_path, 'w') as f:
            json.dump(pbc_violations, f, indent=2)
        print(f"   Written to: {violations_path}")
    else:
        print(f"\nVerified centering + PBC bound on {checked} subhalo(s) across distinct halos -- "
              f"no violations.")

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