import os
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.data_generation.src.subhalo import SubhaloParticles, SubhaloContext
from src.data_generation.src.feature_computer import FeatureComputer
from src.data_generation.src.utils import setup_logger

logger = setup_logger("Compute-Features-Stage")


def compute_features_stage(fetcher, feature_names, cfg, force_recompute=False, halo_ids=None,
                            write_to_new_bucket=False):
    """
    Standalone stage: compute one or more per-subhalo features and
    materialize them into a dedicated features parquet dataset, keyed by
    (id_parents, subhalo_id), partitioned by parent_bucket for consistency
    with the existing subhalo catalog layout.

    EFFICIENCY: particles are read ONCE PER BUCKET (128 filtered parquet
    reads total), then split by subhalo_id in memory via pandas groupby.
    This deliberately bypasses fetcher.get_p_subhalo(), which -- per
    subhalo -- does an O(n_halos) linear scan in
    MagneticumFetcherV2._get_halo_id_for_subhalo AND issues its own
    pd.read_parquet(..., filters=...) call. Calling that once per subhalo
    would cost O(n_subhalos) reads and O(n_subhalos * n_halos) scan work;
    the per-particle 'subhalo_id' column already written by
    MagneticumFetcherV2._store_particles makes both unnecessary here.

    Idempotency across repeated executions:

    1. NEVER writes back into the source subhalo/particle parquet files.
       preprocess() is always applied in-memory only -- this stage only
       ever writes to a brand-new, separate output directory. Source data
       stays immutable across any number of runs.

    2. Work is chunked per bucket (parent_bucket = id_parents % 128, same
       scheme used for subhalos_parquet/particles_parquet), and each bucket
       writes a "_DONE" marker file only after its output has been fully
       and successfully written. On the next run, any bucket with a marker
       present is skipped entirely; any bucket without one (e.g. a
       previous run crashed mid-write) is fully redone from scratch --
       never partially appended to.

    3. Each bucket's output is written to a temp path first, then swapped
       into place, then the marker is written last -- so a crash mid-write
       never leaves a marker with incomplete or duplicated data behind it.

    NOTE on halo_ids + idempotency: when halo_ids is set, a bucket's
    "_DONE" marker still means "everything requested from this bucket in
    THIS run is written" -- it does not distinguish a full-catalog pass
    from a halo_ids-restricted one. Running once with halo_ids=[...] and
    then again with halo_ids=None will skip buckets already marked done,
    even though the unrestricted run wants more subhalos out of them. Use
    force_recompute=True (or a separate features_root) when mixing
    restricted and full runs against the same output directory.

    Parameters
    ----------
    fetcher : MagneticumFetcherV2 (or compatible DataFetcher)
        Must expose get_subhalos(), get_halos(), and .catalog.path
        (used to locate particles_parquet directly).
    feature_names : list[str]
        Registered FeatureComputer feature names to compute, e.g. ['stellar_half_mass_radius'].
    cfg : DictConfig
        Needs cfg.catalog.path / cfg.paths.datasets_dir / cfg.catalog.get('Lbox').
    force_recompute : bool
        If True, ignore existing "_DONE" markers and redo every bucket.
    halo_ids : list[int] | None
        If provided, only subhalos whose id_parents is in this list are
        processed -- useful for quick debug runs against a handful of
        halos instead of the full catalog.
    write_to_new_bucket : bool
        If False (default): results are merged into the SAME bucket
        location used by every other run of this stage (upsert by
        (id_parents, subhalo_id) -- see _write_bucket_atomic). The
        original bucket's other rows are preserved, but the touched rows
        are replaced.
        If True: results are written to a SEPARATE output tree
        (features_root + '_new'), entirely independent of the main
        materialized output. The original bucket is never opened, read,
        or touched in any way. Use this for a debug/scratch run (e.g.
        "just the first 5 halos") that you don't want mixed into the
        real feature catalog at all.
    """
    lbox = cfg.catalog.get('Lbox', None)
    if lbox is None:
        logger.warning("cfg.catalog.get('Lbox') returned None -- PBC/minimum-image "
                        "correction will be a no-op for both particle and subhalo "
                        "positions. If the catalog has a periodic box, this is a bug.")

    dataset_dir = cfg.paths.datasets_dir
    catalog_name = cfg.catalog.path.replace(dataset_dir, '').replace('/', '_')
    features_root = os.path.join(dataset_dir, '..', 'features', f'{catalog_name}_subhalo_features')
    if write_to_new_bucket:
        # Entirely separate output tree -- the real features_root is never
        # opened, read, or written to in this mode.
        features_root = features_root + '_new'

    # get_subhalos()/get_halos() are already cheap after the first run --
    # both hit their own on-disk caches internally.
    subhalo_context = fetcher.get_subhalos()
    halo_context = fetcher.get_halos()  # already indexed by halo_id -- O(1) get_halo_by_id lookups

    halo_ids_set = set(int(h) for h in halo_ids) if halo_ids is not None else None

    buckets = _get_buckets(subhalo_context)

    if halo_ids_set is not None:
        # Only visit buckets that could contain one of the requested halos --
        # no need to scan all 128 when e.g. only 5 halos were asked for.
        relevant_buckets = {h % 128 for h in halo_ids_set}
        buckets = [b for b in buckets if b in relevant_buckets]
        logger.info(f"Compute-features stage: restricted to {len(halo_ids_set)} halo(s), "
                    f"{len(buckets)} bucket(s) to process.")
    else:
        logger.info(f"Compute-features stage: {len(buckets)} buckets to process.")

    total_rows_written = 0
    total_halos_not_found = 0

    for bucket in buckets:
        # NOTE: deliberately NOT named "parent_bucket={bucket}" (Hive-style
        # key=value). PyArrow's dataset reader auto-discovers Hive
        # partitioning from that exact folder-naming pattern -- even for a
        # single, non-partitioned file read via pq.read_table(). That
        # silently injects a synthetic 'parent_bucket' column into any
        # existing-file read inside _write_bucket_atomic, which then gets
        # written back as REAL data on the merge, and the next read finds
        # both a literal int64 column and a Hive-inferred dictionary
        # column with the same name -> ArrowTypeError merging schemas.
        # These folders are addressed directly by path, never scanned as
        # a partitioned dataset, so there's no reason to use that naming.
        bucket_dir = os.path.join(features_root, f'bucket_{bucket}')
        marker_path = os.path.join(bucket_dir, '_DONE')

        if os.path.exists(marker_path) and not force_recompute:
            logger.debug(f"Bucket {bucket}: already materialized, skipping.")
            continue

        logger.info(f"Bucket {bucket}: computing features.")
        bucket_subhalos = _get_bucket_subhalos(subhalo_context, bucket)

        if halo_ids_set is not None:
            bucket_subhalos = bucket_subhalos[bucket_subhalos['id_parents'].isin(halo_ids_set)]

        if bucket_subhalos.empty:
            logger.debug(f"Bucket {bucket}: no matching subhalos, skipping.")
            continue

        # Build a lightweight SubhaloContext scoped to just this bucket's
        # rows, and run the REAL SubhaloContext.preprocess() once per
        # unique halo in the bucket -- this reuses the tested minimum-image
        # PBC correction for subhalo positions (spos) rather than
        # re-deriving it here. Only affects in-memory data -- never
        # written anywhere, matching particle preprocessing's
        # in-memory-only behavior.
        bucket_subhalo_ctx = SubhaloContext.from_dataframe(bucket_subhalos.reset_index(drop=True))

        has_spos = 'spos' in bucket_subhalo_ctx.data.columns
        has_catalog_mstr = 'mstr' in bucket_subhalo_ctx.data.columns
        if bucket == buckets[0]:
            # Logged once, not per-bucket -- this is a schema property of
            # the whole catalog, not something that varies bucket to
            # bucket. If this logs False, subhalo-position preprocessing
            # is silently no-op'ing for the ENTIRE run: preprocess() checks
            # `if 'spos' in self.data` and does nothing otherwise, with no
            # further warning. Check the catalog's column_map if so.
            logger.info(f"Subhalo catalog has 'spos' column: {has_spos} "
                        f"(subhalo-position preprocessing is a no-op if False)")
            logger.info(f"Subhalo catalog has 'mstr' column: {has_catalog_mstr} "
                        f"(mstr_catalog will be NaN for every row if False -- "
                        f"only mstr_particles, summed from fetched star particles, "
                        f"will be available)")

        # Halo lookups cached here and reused in the feature loop below --
        # avoids resolving the same halo_id twice per subhalo.
        halos_by_id = {}
        for id_parents in bucket_subhalos['id_parents'].unique():
            try:
                halo = halo_context.get_halo_by_id(id_parents)
            except Exception as e:
                logger.warning(f"Bucket {bucket}, halo {id_parents}: {e}, "
                                f"skipping subhalo-position preprocessing for this halo.")
                total_halos_not_found += 1
                continue
            halos_by_id[id_parents] = halo
            bucket_subhalo_ctx.preprocess(halo, lbox)

        # ONE read for the whole bucket, not one per subhalo.
        particle_groups = _load_bucket_particle_groups(fetcher, bucket)

        rows = []
        for _, row in bucket_subhalo_ctx.data.iterrows():
            id_parents = row['id_parents']
            subhalo_id = row['subhalo_id']

            halo = halos_by_id.get(id_parents)
            if halo is None:
                # Already logged above when the lookup first failed.
                continue

            result_row = {'id_parents': id_parents, 'subhalo_id': subhalo_id}
            sub_particles = particle_groups.get(int(subhalo_id))

            # Two independently-derived mass values, kept separate so they
            # can be cross-checked against each other:
            #   mstr_catalog   -- the halo finder's own precomputed value,
            #                      pulled straight from the subhalo catalog
            #                      (same category as spos), NaN if that
            #                      column isn't present in this catalog.
            #   mstr_particles -- summed directly from the star particles
            #                      we actually fetched for this subhalo.
            # A meaningful disagreement between the two points at a
            # particle-to-subhalo matching bug, not just a display nicety.
            result_row['mstr_catalog'] = float(row['mstr']) if has_catalog_mstr else float('nan')

            has_particles = sub_particles is not None and hasattr(sub_particles, 'star_m') \
                and len(sub_particles.star_m) > 0
            result_row['n_star_particles'] = int(len(sub_particles.star_m)) if has_particles else 0
            result_row['mstr_particles'] = float(sub_particles.star_m.sum()) if has_particles else 0.0

            if not has_particles or sub_particles.star_m.sum() <= 0:
                for feature_name in feature_names:
                    result_row[feature_name] = float('nan')
                rows.append(result_row)
                continue

            if not hasattr(sub_particles, 'star_p') or sub_particles.star_p is None:
                # preprocess() would silently no-op here too -- surface it,
                # since every distance-based feature depends on star_p
                # having been PBC-corrected.
                logger.warning(f"Bucket {bucket}, subhalo {subhalo_id}: no star_p attribute -- "
                                f"particle preprocessing will be a no-op for this subhalo.")

            # In-memory only -- never written back to source particle files.
            sub_particles.preprocess(halo, lbox)

            for feature_name in feature_names:
                try:
                    result_row[feature_name] = FeatureComputer.compute(feature_name, particles=sub_particles)
                except Exception as e:
                    logger.error(f"Bucket {bucket}, subhalo {subhalo_id}, feature {feature_name}: {e}")
                    result_row[feature_name] = float('nan')

            rows.append(result_row)

        if not rows:
            logger.debug(f"Bucket {bucket}: no rows produced, skipping write.")
            continue

        _write_bucket_atomic(bucket_dir, rows)
        total_rows_written += len(rows)
        logger.info(f"Bucket {bucket}: wrote {len(rows)} rows.")

    if total_rows_written == 0:
        logger.warning(
            f"Compute-features stage produced ZERO output rows across "
            f"{len(buckets)} bucket(s) processed"
            f"{f' (halo_ids={sorted(halo_ids_set)})' if halo_ids_set is not None else ''}. "
            f"{total_halos_not_found} halo lookup(s) failed with 'not found' -- "
            f"if that number is high, the requested halo_ids likely don't exist in "
            f"HaloContext (e.g. filtered out by halo_conditions, or drawn from an id "
            f"range that doesn't overlap between the halo and subhalo catalogs). "
            f"No output directory will exist for this run."
        )

    logger.info("Compute-features stage complete.")


def _load_bucket_particle_groups(fetcher, bucket):
    """
    Read ALL star particles for a bucket in a single filtered parquet read,
    then split into one SubhaloParticles per subhalo_id in memory.

    Mirrors MagneticumFetcherV2._load_subhalo_particles' column handling
    (strip whitespace-padded g3read field names, apply the same
    column_map) but applied once per bucket instead of once per subhalo.
    """
    particles_parquet = os.path.join(fetcher.catalog.path, "particles_parquet")

    if not os.path.exists(particles_parquet):
        logger.warning(f"Bucket {bucket}: {particles_parquet} does not exist yet.")
        return {}

    try:
        df = pd.read_parquet(particles_parquet, filters=[('parent_bucket', '==', bucket)])
    except Exception as e:
        logger.error(f"Bucket {bucket}: failed to read particles: {e}")
        return {}

    if len(df) == 0:
        return {}

    groups = {}
    for subhalo_id, group in df.groupby('subhalo_id'):
        data = {}
        for col in group.columns:
            if col in ('parent_id', 'subhalo_id', 'parent_bucket', 'parent_bucket_data'):
                continue
            data[col] = group[col].to_numpy()
        data = {key.rstrip(): value for key, value in data.items()}

        # Stack vector-valued columns (POS/VEL) into real 2D float arrays.
        # After a parquet list-column round trip through pandas, these come
        # back as object-dtype arrays of arrays/lists, not proper 2D
        # ndarrays -- SubhaloParticles.__init__'s column_map branch only
        # stacks when the source is a pd.Series, so this shape survives
        # into star_p/star_v untouched and breaks preprocess()'s PBC
        # arithmetic (elementwise comparisons on object arrays raise
        # "truth value of an array... is ambiguous"). Stack explicitly here.
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray) and value.dtype == object and len(value) > 0 \
                    and isinstance(value[0], (list, np.ndarray)):
                data[key] = np.stack(value)
        # TODO avoid hardcoding -- same column_map used in
        # MagneticumFetcherV2._load_subhalo_particles, centralize if that
        # ever changes.
        groups[int(subhalo_id)] = SubhaloParticles(data=data, column_map={'POS': 'p', 'MASS': 'm', 'VEL': 'v'})

    return groups


def _write_bucket_atomic(bucket_dir, rows):
    """
    Write a bucket's rows into a COPY of any existing bucket data (upsert
    by (id_parents, subhalo_id)), then swap the copy into place and write
    the _DONE marker last.

    This must NOT wholesale-replace the bucket: if a prior full run
    already materialized every subhalo in this bucket, and the current
    run is halo_ids-restricted (or force_recompute on a subset), `rows`
    only contains the subset actually touched this time. Replacing the
    bucket outright would silently drop every other subhalo's previously
    computed feature values. So existing on-disk rows are read first, new
    rows are merged in as an upsert (matching keys are overwritten,
    everything else is preserved), and only the MERGED result is written
    -- never the new rows alone.

    The merge happens entirely in an in-memory copy / temp directory; the
    real bucket_dir is only touched by the final atomic swap.
    """
    tmp_dir = bucket_dir + '.tmp'
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    key_cols = ['id_parents', 'subhalo_id']
    new_df = pd.DataFrame(rows)

    existing_path = os.path.join(bucket_dir, 'data.parquet')
    if os.path.exists(existing_path):
        existing_df = pq.read_table(existing_path).to_pandas()
        # Upsert on a COPY: concat then drop_duplicates(keep='last') keeps
        # the new row wherever a key overlaps, and keeps every untouched
        # existing row as-is.
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True)
    else:
        merged_df = new_df

    table = pa.Table.from_pandas(merged_df, preserve_index=False)
    pq.write_table(table, os.path.join(tmp_dir, 'data.parquet'))

    if os.path.exists(bucket_dir):
        shutil.rmtree(bucket_dir)  # swap target only, after the merged copy is ready
    os.rename(tmp_dir, bucket_dir)

    # Marker written LAST, only once the swap succeeded.
    with open(os.path.join(bucket_dir, '_DONE'), 'w') as f:
        f.write('')


def _get_buckets(subhalo_context):
    """
    subhalos_parquet columns are 'id_parents' / 'parent_bucket' /
    'parent_bucket_data' (per MagneticumFetcherV2._get_subhalos) -- NOT
    'parent_id', which is only the particles_parquet's naming.
    """
    if hasattr(subhalo_context, 'dataset'):
        table = subhalo_context.dataset.to_table(columns=['parent_bucket_data'])
        return sorted(set(table['parent_bucket_data'].to_pylist()))
    return sorted((subhalo_context.data['id_parents'] % 128).unique().tolist())


def _get_bucket_subhalos(subhalo_context, bucket):
    """
    Pulls id_parents/subhalo_id plus 'spos', 'mstr' (and 'group_number'
    if present) when available in the catalog schema. 'mstr' is a
    catalog-provided property (like spos), not something computed here --
    it's carried through so it can be cross-checked against the
    independently particle-summed mass in compute_features_stage.
    Columns are selected conditionally so this still works against
    catalogs/fixtures that don't have these fields at all.
    """
    optional_cols = ['spos', 'mstr', 'group_number']

    if hasattr(subhalo_context, 'dataset'):
        schema_names = set(subhalo_context.dataset.schema.names)
        cols = ['id_parents', 'subhalo_id'] + [c for c in optional_cols if c in schema_names]
        table = subhalo_context.dataset.to_table(
            columns=cols,
            filter=ds.field('parent_bucket_data') == bucket
        )
        return table.to_pandas()

    cols = ['id_parents', 'subhalo_id'] + [c for c in optional_cols if c in subhalo_context.data.columns]
    return subhalo_context.data[subhalo_context.data['id_parents'] % 128 == bucket][cols]