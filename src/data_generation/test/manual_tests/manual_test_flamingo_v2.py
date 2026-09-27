"""
Independent, standalone smoke test for FlamingoFetcherV2 against the real
Flamingo SOAP catalogue on the cluster (not a pytest fixture — run it directly).

Loads the hydra config the same way the pipeline does (data_generation.yaml
with the "flamingo" catalog swapped in, paths defaulting to the "gpfs"
cluster mount — see configs/data_generation.yaml / configs/paths/gpfs.yaml),
forces catalog.name to "flamingo_v2" so DataFetcherFactory instantiates
FlamingoFetcherV2 (the checked-in configs/catalog/flamingo.yaml still says
"flamingo", i.e. V1 — this script does not depend on that being switched),
and exercises get_halos()/get_subhalos()/get_p_halo()/get_p_subhalo().

Also runs a diagnostic specific to V2's fix: Flamingo's SOAP catalogue has
two non-interchangeable ID spaces for the same rows (see
FlamingoFetcherV2's own class docstring in data_fetcher.py) — row-position
(what V1 uses as halo_id/subhalo_id) and true catalogue index (what the
particle data is actually keyed by). This script finds halos where the two
diverge on THIS run's data and shows V1 and V2 disagreeing on particle
counts for them directly, rather than only asserting the fix from unit
tests.

Edit the constants below to point at real data, then run directly:
    python src/data_generation/test/manual_tests/manual_test_flamingo_v2.py
"""
import os

import numpy as np
from hydra import compose, initialize_config_dir
from src.data_generation.src.data_fetcher import DataFetcherFactory, FlamingoFetcher, FlamingoFetcherV2


CATALOG_PATH = '/gpfs/scratch/bsc70/hpai/storage/projects/space/datasets/flamingo/L1000N0900'
SNAPNUM = 73       # None => use configs/catalog/flamingo.yaml's default (73)
# Number of halos to try get_p_halo()/get_p_subhalo() on in TEST 3/4 (particle
# files are read one galaxy at a time, so keep this small for a quick smoke test)
N_HALOS_FOR_PARTICLE_TEST = 3
# How many halos to scan (in get_halos() order) looking for ones where
# row-position != true catalogue index, for TEST 4's V1-vs-V2 comparison.
N_HALOS_TO_SCAN_FOR_DIVERGENCE = 2000


def load_flamingo_config(fetcher_name: str = "flamingo_v2"):
    """
    Build the same DictConfig the real pipeline would use for a data_generation
    run, with the "flamingo" catalog selected, then apply the
    CATALOG_PATH/SNAPNUM overrides above and force catalog.name to
    `fetcher_name` so DataFetcherFactory picks the fetcher we actually want
    (V1 or V2), independent of what's checked in to
    configs/catalog/flamingo.yaml.
    """
    # initialize_config_dir() requires an ABSOLUTE path (unlike initialize(),
    # it does no relative-path resolution itself) — resolve relative to this
    # file's own location rather than the process's cwd, so it works no
    # matter where this script is invoked from.
    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "configs"))
    with initialize_config_dir(config_dir=configs_dir, version_base=None):
        cfg = compose(config_name="data_generation.yaml")
        cfg.catalog = compose(config_name="catalog/flamingo.yaml")["catalog"]

    if CATALOG_PATH is not None:
        cfg.catalog.path = CATALOG_PATH
    if SNAPNUM is not None:
        cfg.catalog.snapnum = SNAPNUM
    cfg.catalog.name = fetcher_name

    return cfg


def main():
    cfg = load_flamingo_config("flamingo_v2")

    print("=" * 60)
    print("Config")
    print("=" * 60)
    print(f"catalog.name:    {cfg.catalog.name}")
    print(f"catalog.path:    {cfg.catalog.path}")
    print(f"catalog.snapnum: {cfg.catalog.snapnum}")
    print(f"catalog.Lbox:    {cfg.catalog.Lbox}")

    print()
    print("=" * 60)
    print("Instantiating FlamingoFetcherV2")
    print("=" * 60)
    fetcher = DataFetcherFactory.get_data_fetcher(cfg.catalog)
    print(f"Fetcher: {type(fetcher).__name__}")
    assert isinstance(fetcher, FlamingoFetcherV2), \
        f"Expected FlamingoFetcherV2, got {type(fetcher).__name__} — check catalog.name routing in DataFetcherFactory"

    print()
    print("=" * 60)
    print("TEST 1: get_halos()")
    print("=" * 60)
    halo_context = fetcher.get_halos()
    print(f"Total halos: {len(halo_context.data)}")
    print(f"Columns: {list(halo_context.data.columns)}")
    print(halo_context.data.head())

    print()
    print("=" * 60)
    print("TEST 2: get_subhalos()")
    print("=" * 60)
    subhalo_context = fetcher.get_subhalos()
    print(f"Total subhalos: {len(subhalo_context.data)}")
    print(f"Columns: {list(subhalo_context.data.columns)}")
    print(subhalo_context.data.head())

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
    print(f"TEST 3: get_p_subhalo() / get_p_halo() on up to {N_HALOS_FOR_PARTICLE_TEST} halos (V2)")
    print("=" * 60)
    sample_halo_ids = halo_context.halo_id[:N_HALOS_FOR_PARTICLE_TEST]
    if not sample_halo_ids:
        print("No halos to test particle access on (get_halos() returned 0 rows) — skipping remaining tests.")
        return

    for hid in sample_halo_ids:
        try:
            halo_row = halo_context.get_halo_by_id(hid)
            particles = fetcher.get_p_subhalo(subhalo_id=hid)
            n_star = len(particles.star_m)
            print(f"halo {hid}: central galaxy has {n_star} star particles "
                  f"(star_m dtype={particles.star_m.dtype}, star_p shape={particles.star_p.shape})")
        except Exception as e:
            print(f"halo {hid}: get_p_subhalo() FAILED: {type(e).__name__}: {e}")

    print()
    print("get_p_halo() on the first sampled halo (aggregates the halo's own + satellites' star particles):")
    try:
        first_halo = halo_context.get_halo_by_id(sample_halo_ids[0])
        halo_particles = fetcher.get_p_halo(halo=first_halo)
        print(f"  total star particles across halo {sample_halo_ids[0]}: {len(halo_particles.star_m)}")
    except Exception as e:
        print(f"get_p_halo() FAILED: {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print("TEST 4: V1 vs V2 — row-position vs true-catalogue-index divergence (the bug V2 fixes)")
    print("=" * 60)
    print("See FlamingoFetcherV2's class docstring in data_fetcher.py for the full explanation:")
    print("SOAP exposes row-position IDs (what V1 uses as halo_id) and true catalogue indices")
    print("(what particle data is actually keyed by) as two separate, non-interchangeable spaces.")
    print()

    catalogue_index_field = np.asarray(fetcher.fof_subhalos.input_halos.halo_catalogue_index)
    scan_ids = halo_context.halo_id[:N_HALOS_TO_SCAN_FOR_DIVERGENCE]
    diverging = [hid for hid in scan_ids if 0 <= hid < len(catalogue_index_field) and catalogue_index_field[hid] != hid]
    print(f"Scanned {len(scan_ids)} halos: {len(diverging)} have row-position != true catalogue index.")

    if not diverging:
        print("No diverging halo found in this scan window — try increasing N_HALOS_TO_SCAN_FOR_DIVERGENCE.")
        return

    v1_fetcher = DataFetcherFactory.get_data_fetcher(load_flamingo_config("flamingo").catalog)
    # Share the already-loaded SWIFT datasets instead of re-reading files from disk.
    v1_fetcher.fof_subhalos = fetcher.fof_subhalos
    v1_fetcher.particles = fetcher.particles

    for hid in diverging[:N_HALOS_FOR_PARTICLE_TEST]:
        true_id = int(catalogue_index_field[hid])
        halo_row = halo_context.get_halo_by_id(hid)
        try:
            v1_particles = v1_fetcher.get_p_halo(halo=halo_row)
            v2_particles = fetcher.get_p_halo(halo=halo_row)
            print(f"halo {hid} (true catalogue index {true_id}): "
                  f"V1 star count = {len(v1_particles.star_m)}, V2 star count = {len(v2_particles.star_m)}"
                  + ("  <-- disagree, as expected" if len(v1_particles.star_m) != len(v2_particles.star_m) else ""))
        except Exception as e:
            print(f"halo {hid}: comparison FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
