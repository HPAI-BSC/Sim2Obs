"""
Independent, standalone smoke test for ObeliskFetcher against the real
Obelisk catalogue on the cluster (not a pytest fixture — run it directly).

Loads the hydra config the same way the pipeline does (data_generation.yaml
with the "obelisk" catalog swapped in, paths defaulting to the "gpfs"
cluster mount — see configs/data_generation.yaml / configs/paths/gpfs.yaml),
instantiates ObeliskFetcher via DataFetcherFactory, and exercises
get_halos()/get_subhalos()/get_p_halo()/get_p_subhalo().

Edit the constants below to point at real data, then run directly:
    python src/data_generation/test/manual_tests/manual_test_obelisk.py
"""
import os

import numpy as np
from hydra import compose, initialize_config_dir
from src.data_generation.src.data_fetcher import DataFetcherFactory


CATALOG_PATH = '/gpfs/scratch/bsc70/hpai/storage/projects/space/datasets/obelisk'
SNAPNUM = 201
LBOX = 100000.0
# Number of halos to try get_p_halo() on in TEST 3 (star-particle files are
# read from disk one galaxy at a time, so keep this small for a quick smoke test)
N_HALOS_FOR_PARTICLE_TEST = 3


def load_obelisk_config():
    """
    Build the same DictConfig the real pipeline would use for a data_generation
    run, with the "obelisk" catalog selected instead of the "magneticum"
    default (see configs/data_generation.yaml's `defaults` list), then apply
    the CATALOG_PATH/SNAPNUM/LBOX overrides above.
    """
    # initialize_config_dir() requires an ABSOLUTE path (unlike initialize(),
    # it does no relative-path resolution itself) — resolve relative to this
    # file's own location rather than the process's cwd, so it works no
    # matter where this script is invoked from.
    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "configs"))
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name="data_generation.yaml")
        cfg.catalog = compose(config_name="catalog/obelisk.yaml")["catalog"]

    if CATALOG_PATH is not None:
        cfg.catalog.path = CATALOG_PATH
    if SNAPNUM is not None:
        cfg.catalog.snapnum = SNAPNUM
    if LBOX is not None:
        cfg.catalog.Lbox = LBOX

    return cfg


def main():
    cfg = load_obelisk_config()

    print("=" * 60)
    print("Config")
    print("=" * 60)
    print(f"catalog.path:    {cfg.catalog.path}")
    print(f"catalog.snapnum: {cfg.catalog.snapnum}")
    print(f"catalog.Lbox:    {cfg.catalog.Lbox}")

    print()
    print("=" * 60)
    print("Instantiating ObeliskFetcher")
    print("=" * 60)
    fetcher = DataFetcherFactory.get_data_fetcher(cfg.catalog)
    print(f"Fetcher: {type(fetcher).__name__}")
    print(f"Raw catalogue columns: {sorted(fetcher._raw.keys())}")
    print(f"Raw catalogue rows: {len(fetcher._raw['ID'])}")

    print()
    print("=" * 60)
    print("DIAGNOSTIC: raw level / mvir distributions (before any filtering)")
    print("=" * 60)
    level = np.asarray(fetcher._raw["level"])
    mvir = np.asarray(fetcher._raw["mvir"])
    levels, counts = np.unique(level, return_counts=True)
    print(f"level value counts: {dict(zip(levels.tolist(), counts.tolist()))}")
    n_level1 = int(np.sum(level == 1))
    print(f"rows with level == 1: {n_level1}")
    print(f"mvir [Msun] over level==1 rows: min={mvir[level == 1].min():.3e}  "
          f"max={mvir[level == 1].max():.3e}  median={np.median(mvir[level == 1]):.3e}"
          if n_level1 > 0 else "mvir over level==1 rows: N/A (no level==1 rows)")
    # Evaluate the config's current mvir filter directly rather than hardcoding
    # a threshold here, so this stays correct as configs/catalog/obelisk.yaml's
    # filter is tuned.
    mvir_filter_expr = cfg.catalog.fields.halo.mvir.filter[0]
    threshold_msun = float(eval(mvir_filter_expr.split(' ', 1)[1], {}, {"h": cfg.catalog.H}))
    n_above = int(np.sum((level == 1) & (mvir > threshold_msun)))
    print(f"level==1 rows with mvir > {threshold_msun:.3e} Msun (config filter '{mvir_filter_expr}'): {n_above}")

    print()
    print("=" * 60)
    print("TEST 1: get_halos()")
    print("=" * 60)
    halo_context = None
    try:
        halo_context = fetcher.get_halos()
        print(f"Total halos: {len(halo_context.data)}")
        print(f"Columns: {list(halo_context.data.columns)}")
        print(halo_context.data.head())
    except Exception as e:
        print(f"get_halos() FAILED: {type(e).__name__}: {e}")
        print("(continuing to TEST 2 anyway — get_subhalos() has its own independent mstar filter, "
              "so it can still return results even if get_halos() legitimately found zero halos "
              "above the mvir cut)")

    print()
    print("=" * 60)
    print("TEST 2: get_subhalos()")
    print("=" * 60)
    subhalo_context = fetcher.get_subhalos()
    print(f"Total subhalos: {len(subhalo_context.data)}")
    print(f"Columns: {list(subhalo_context.data.columns)}")
    print(subhalo_context.data.head())

    if halo_context is None:
        print()
        print("=" * 60)
        print("SANITY CHECK skipped (get_halos() failed above)")
        print("=" * 60)
        return

    print()
    print("=" * 60)
    print("SANITY CHECK: every halo_id from get_halos() is a central (self-parented) row in get_subhalos()")
    print("=" * 60)
    halo_ids = set(halo_context.halo_id)
    centrals = set(
        subhalo_context.data.loc[subhalo_context.data["id_parents"] == subhalo_context.data["subhalo_id"], "subhalo_id"]
    )
    missing = halo_ids - centrals
    if missing:
        print(f"FAILED: {len(missing)} halo_id(s) have no matching central subhalo row: {sorted(missing)[:10]}...")
    else:
        print(f"PASSED: all {len(halo_ids)} halos have a matching central subhalo row")

    print()
    print("=" * 60)
    print(f"TEST 3: get_p_subhalo() / get_p_halo() on up to {N_HALOS_FOR_PARTICLE_TEST} halos")
    print("=" * 60)
    sample_halo_ids = halo_context.halo_id[:N_HALOS_FOR_PARTICLE_TEST]
    if not sample_halo_ids:
        print("No halos to test particle access on (get_halos() returned 0 rows) — skipping.")
        return

    for hid in sample_halo_ids:
        try:
            halo_row = halo_context.get_halo_by_id(hid)
            particles = fetcher.get_p_subhalo(subhalo_id=hid)
            n_star = len(particles.star_m)
            print(f"halo {hid}: central galaxy has {n_star} star particles "
                  f"(star_m dtype={particles.star_m.dtype}, star_p shape={particles.star_p.shape})")

            # UNITS CROSS-CHECK: configs/catalog/obelisk.yaml flags
            # fields.particles.stars.position's units as unconfirmed. Compare
            # the mean of this galaxy's raw star positions (before any
            # scaling, since scaling is currently null/pass-through) against
            # the main catalogue's own xstar/ystar/zstar for the same ID —
            # both should be the same order of magnitude if the assumed
            # units (matching the catalogue's own centre fields) are right.
            raw_star_position_mean = np.mean(
                fetcher._read_gal_stars_file(hid)['position'], axis=0
            )
            row_mask = fetcher._raw['ID'] == hid
            catalogue_star_centre = np.array([
                fetcher._raw['xstar'][row_mask][0],
                fetcher._raw['ystar'][row_mask][0],
                fetcher._raw['zstar'][row_mask][0],
            ])
            print(f"  mean(gal_stars position) = {raw_star_position_mean}")
            print(f"  catalogue xstar/ystar/zstar = {catalogue_star_centre}")
        except Exception as e:
            print(f"halo {hid}: get_p_subhalo() FAILED: {type(e).__name__}: {e}")

    print()
    print("get_p_halo() on the first sampled halo (aggregates all its subhalos' star particles):")
    try:
        first_halo = halo_context.get_halo_by_id(sample_halo_ids[0])
        print(f"  subhalo_ids: {first_halo.subhalo_ids}")
        halo_particles = fetcher.get_p_halo(halo=first_halo)
        print(f"  total star particles across halo {sample_halo_ids[0]}: {len(halo_particles.star_m)}")
    except Exception as e:
        print(f"get_p_halo() FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
