import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data_generation.src.feature_computer import FeatureComputer
from src.data_generation.src.features_dataset.compute_features import compute_features_stage


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeCatalog:
    def __init__(self, path, lbox=None):
        self.path = path
        self._lbox = lbox

    def get(self, key, default=None):
        return self._lbox if key == 'Lbox' else default


class FakeCfg:
    def __init__(self, catalog_path, datasets_dir, lbox=None):
        self.catalog = FakeCatalog(catalog_path, lbox)
        self.paths = SimpleNamespace(datasets_dir=datasets_dir)


class FakeHalo:
    def __init__(self, halo_id, gpos=(0.0, 0.0, 0.0)):
        self.halo_id = halo_id
        self.gpos = np.array(gpos, dtype=float)


class FakeHaloContext:
    def __init__(self, halos_by_id):
        self._halos = halos_by_id

    def get_halo_by_id(self, halo_id):
        if halo_id not in self._halos:
            raise Exception(f"Halo ID {halo_id} not found.")
        return self._halos[halo_id]


class FakeSubhaloContext:
    """No `.dataset` attribute -> compute_features_stage uses the eager
    (pandas-backed) branch of _get_buckets / _get_bucket_subhalos."""
    def __init__(self, df: pd.DataFrame):
        self.data = df


class FakeFetcher:
    """Only needs .catalog.path -- particles are read directly from
    particles_parquet on disk by _load_bucket_particle_groups, matching
    how MagneticumFetcherV2 actually stores them."""
    def __init__(self, catalog_path, subhalo_context, halo_context):
        self.catalog = SimpleNamespace(path=catalog_path)
        self._subhalo_context = subhalo_context
        self._halo_context = halo_context

    def get_subhalos(self):
        return self._subhalo_context

    def get_halos(self):
        return self._halo_context


@pytest.fixture(autouse=True)
def stub_feature_computer(monkeypatch):
    """Avoid depending on the real SHMR math -- return particle count so
    tests can assert on a predictable value."""
    def fake_compute(name, particles):
        return float(len(particles.star_m))

    monkeypatch.setattr(FeatureComputer, "compute", staticmethod(fake_compute))


@pytest.fixture()
def catalog_dir(tmp_path):
    d = tmp_path / "catalog"
    d.mkdir()
    return str(d)


@pytest.fixture()
def cfg(tmp_path, catalog_dir):
    datasets_dir = str(tmp_path / "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    return FakeCfg(catalog_path=catalog_dir, datasets_dir=datasets_dir, lbox=100.0)


@pytest.fixture()
def features_root(cfg):
    dataset_dir = cfg.paths.datasets_dir
    catalog_name = cfg.catalog.path.replace(dataset_dir, '').replace('/', '_')
    return os.path.join(dataset_dir, '..', 'features', f'{catalog_name}_subhalo_features')


def write_particles_parquet(catalog_dir, particle_rows):
    """particle_rows: list of dicts with POS/MASS/VEL/parent_id/subhalo_id/parent_bucket."""
    particles_parquet = os.path.join(catalog_dir, "particles_parquet")
    df = pd.DataFrame(particle_rows)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(table, root_path=particles_parquet, partition_cols=["parent_bucket"])


def build_catalog(catalog_dir, halo_subhalo_particle_counts):
    """
    halo_subhalo_particle_counts: dict {halo_id: {subhalo_id: n_particles}}.
    Builds a subhalo catalog DataFrame, a halo lookup, and writes matching
    particles_parquet rows -- one distinct star particle mass=1.0 per
    particle, so stub SHMR (particle count) is predictable.
    """
    subhalo_rows = []
    particle_rows = []
    halos_by_id = {}

    for halo_id, subhalos in halo_subhalo_particle_counts.items():
        bucket = halo_id % 128
        halos_by_id[halo_id] = FakeHalo(halo_id=halo_id)
        for subhalo_id, n_particles in subhalos.items():
            subhalo_rows.append({"id_parents": halo_id, "subhalo_id": subhalo_id})
            for i in range(n_particles):
                particle_rows.append({
                    "POS ": [float(i), 0.0, 0.0],
                    "MASS": 1.0,
                    "VEL ": [0.0, 0.0, 0.0],
                    "parent_id": halo_id,
                    "subhalo_id": subhalo_id,
                    "parent_bucket": bucket,
                    "parent_bucket_data": bucket,
                })

    if particle_rows:
        write_particles_parquet(catalog_dir, particle_rows)

    subhalo_context = FakeSubhaloContext(pd.DataFrame(subhalo_rows))
    halo_context = FakeHaloContext(halos_by_id)
    fetcher = FakeFetcher(catalog_dir, subhalo_context, halo_context)
    return fetcher


def read_bucket(features_root, bucket):
    path = os.path.join(features_root, f"bucket_{bucket}", "data.parquet")
    return pq.read_table(path).to_pandas()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestComputeFeaturesStage:

    def test_writes_bucket_output_and_done_marker(self, catalog_dir, cfg, features_root):
        fetcher = build_catalog(catalog_dir, {0: {100: 1, 101: 2}})

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        bucket_dir = os.path.join(features_root, "bucket_0")
        assert os.path.exists(os.path.join(bucket_dir, "_DONE"))

        table = read_bucket(features_root, 0)
        assert set(table["subhalo_id"]) == {100, 101}

    def test_computed_feature_values_are_correct(self, catalog_dir, cfg, features_root):
        fetcher = build_catalog(catalog_dir, {0: {100: 1, 101: 2}})

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        table = read_bucket(features_root, 0).set_index("subhalo_id")
        assert table.loc[100, "stellar_half_mass_radius"] == 1.0
        assert table.loc[101, "stellar_half_mass_radius"] == 2.0

    def test_mstar_and_n_star_particles_are_recorded_alongside_feature(self, catalog_dir, cfg, features_root):
        """mstar/n_star_particles should let results be cross-checked
        against the underlying data (e.g. a tiny SHMR next to a single
        particle vs. hundreds of particles)."""
        # build_catalog's make_particles fixture gives every star mass=1.0.
        fetcher = build_catalog(catalog_dir, {0: {100: 1, 101: 3}})

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        table = read_bucket(features_root, 0).set_index("subhalo_id")
        assert table.loc[100, "n_star_particles"] == 1
        assert table.loc[100, "mstar_particles"] == 1.0
        assert table.loc[101, "n_star_particles"] == 3
        assert table.loc[101, "mstar_particles"] == 3.0

    def test_mstar_and_n_star_particles_are_zero_for_empty_subhalo(self, catalog_dir, cfg, features_root):
        fetcher = build_catalog(catalog_dir, {0: {100: 1, 101: 0}})

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        table = read_bucket(features_root, 0).set_index("subhalo_id")
        assert table.loc[101, "n_star_particles"] == 0
        assert table.loc[101, "mstar_particles"] == 0.0
        assert pd.isna(table.loc[101, "stellar_half_mass_radius"])

    def test_mstar_catalog_is_pulled_through_when_present(self, catalog_dir, cfg, features_root):
        """
        mstar_catalog is a halo-finder-provided property (like spos), NOT
        computed here -- it should be carried straight through from the
        subhalo catalog schema, independent of and comparable against
        mstar_particles (summed from the fetched star particles).
        """
        halo_id, subhalo_id = 0, 100
        catalog_mstar_value = 42.0  # deliberately different from the
                                     # particle-summed mass (3 particles * 1.0 = 3.0)
                                     # to prove these are genuinely two
                                     # separate values, not aliases.

        subhalo_df = pd.DataFrame([{
            "id_parents": halo_id, "subhalo_id": subhalo_id, "mstar": catalog_mstar_value,
        }])
        subhalo_context = FakeSubhaloContext(subhalo_df)
        halo_context = FakeHaloContext({halo_id: FakeHalo(halo_id=halo_id)})

        write_particles_parquet(catalog_dir, [
            {"POS ": [float(i), 0.0, 0.0], "MASS": 1.0, "VEL ": [0.0, 0.0, 0.0],
             "parent_id": halo_id, "subhalo_id": subhalo_id,
             "parent_bucket": 0, "parent_bucket_data": 0}
            for i in range(3)
        ])

        fetcher = FakeFetcher(catalog_dir, subhalo_context, halo_context)

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        table = read_bucket(features_root, 0).set_index("subhalo_id")
        assert table.loc[subhalo_id, "mstar_catalog"] == catalog_mstar_value
        assert table.loc[subhalo_id, "mstar_particles"] == 3.0
        assert table.loc[subhalo_id, "mstar_catalog"] != table.loc[subhalo_id, "mstar_particles"]

    def test_mstar_catalog_is_nan_when_column_absent(self, catalog_dir, cfg, features_root):
        """No 'mstar' column in the catalog schema at all -> mstar_catalog
        is NaN, not silently 0 or missing -- distinguishes "we don't have
        this value" from "the catalog says it's zero"."""
        fetcher = build_catalog(catalog_dir, {0: {100: 1}})  # no 'mstar' column

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        table = read_bucket(features_root, 0).set_index("subhalo_id")
        assert pd.isna(table.loc[100, "mstar_catalog"])
        assert table.loc[100, "mstar_particles"] == 1.0  # this one's still computed fine

    def test_second_run_skips_already_materialized_bucket(self, catalog_dir, cfg, features_root):
        fetcher = build_catalog(catalog_dir, {0: {100: 1}})

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)
        table_first = read_bucket(features_root, 0)

        # Corrupt the particles on disk -- if the second run actually
        # recomputed, this would change the result. It shouldn't, because
        # the bucket's _DONE marker should cause it to be skipped.
        shutil.rmtree(os.path.join(catalog_dir, "particles_parquet"))

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)
        table_second = read_bucket(features_root, 0)

        pd.testing.assert_frame_equal(table_first, table_second)

    def test_halo_ids_restricts_to_requested_halos_only(self, catalog_dir, cfg, features_root):
        fetcher = build_catalog(catalog_dir, {0: {100: 1}, 1: {101: 1}, 2: {102: 1}})

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg, halo_ids=[0, 1])

        # Bucket 2 was never touched -- no output for it at all.
        assert not os.path.exists(os.path.join(features_root, "bucket_2"))

        table0 = read_bucket(features_root, 0)
        table1 = read_bucket(features_root, 1)
        assert set(table0["id_parents"]) == {0}
        assert set(table1["id_parents"]) == {1}

    def test_partial_recompute_does_not_delete_other_subhalos_in_bucket(self, catalog_dir, cfg, features_root):
        """
        Regression test: two halos landing in the SAME bucket (0 and 128,
        both % 128 == 0). A full run materializes both. A later
        halo_ids-restricted + force_recompute run touching only one of
        them must NOT wipe out the other's already-computed row -- the
        bucket write has to merge into a copy of existing data, not
        replace the bucket wholesale.
        """
        fetcher = build_catalog(catalog_dir, {0: {100: 1}, 128: {228: 2}})

        # Full run: both halo 0's and halo 128's subhalos land in bucket 0.
        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)
        table_before = read_bucket(features_root, 0)
        assert set(table_before["subhalo_id"]) == {100, 228}

        # Restricted + forced recompute, touching ONLY halo 0's subhalo.
        compute_features_stage(
            fetcher, ["stellar_half_mass_radius"], cfg,
            force_recompute=True, halo_ids=[0],
        )

        table_after = read_bucket(features_root, 0)
        # Both subhalos must still be present -- halo 128's subhalo 228
        # must survive even though this run never touched it.
        assert set(table_after["subhalo_id"]) == {100, 228}

        # And halo 128's previously-computed value must be unchanged --
        # not dropped, not reset to nan.
        assert table_after.set_index("subhalo_id").loc[228, "stellar_half_mass_radius"] == 2.0

    def test_force_recompute_updates_the_touched_row_in_place(self, catalog_dir, cfg, features_root):
        """The upsert half of the merge: a genuinely recomputed row's
        value should reflect the NEW computation, not the stale one."""
        fetcher = build_catalog(catalog_dir, {0: {100: 1}})

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)
        table_before = read_bucket(features_root, 0).set_index("subhalo_id")
        assert table_before.loc[100, "stellar_half_mass_radius"] == 1.0

        # Add more particles to subhalo 100 and force a recompute.
        shutil.rmtree(os.path.join(catalog_dir, "particles_parquet"))
        write_particles_parquet(catalog_dir, [
            {"POS ": [0.0, 0.0, 0.0], "MASS": 1.0, "VEL ": [0.0, 0.0, 0.0],
             "parent_id": 0, "subhalo_id": 100, "parent_bucket": 0, "parent_bucket_data": 0}
            for _ in range(5)
        ])

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg, force_recompute=True)

        table_after = read_bucket(features_root, 0).set_index("subhalo_id")
        assert table_after.loc[100, "stellar_half_mass_radius"] == 5.0

    def test_all_halos_missing_produces_no_output_but_no_crash(self, catalog_dir, cfg, features_root, caplog):
        """
        Regression test for the exact cluster failure: halo_ids requested
        that don't exist in HaloContext (e.g. drawn from an id range that
        doesn't overlap between the halo and subhalo catalogs). Every
        subhalo row should be skipped gracefully, zero rows written, no
        bucket directory created -- and the stage should log a clear
        WARNING explaining why, rather than silently finishing with no
        output and no explanation (which is what caused the downstream
        FileNotFoundError when the manual test tried to read it back).
        """
        # Subhalo catalog references halo 999, but HaloContext knows
        # nothing about halo 999 at all.
        fetcher = build_catalog(catalog_dir, {999: {100: 1}})
        fetcher._halo_context = FakeHaloContext({})  # no halos known

        import logging
        # caplog.at_level(level, logger=name) only lowers that logger's
        # level threshold -- it does NOT move where pytest's capture
        # handler is attached (that stays on the root logger). If this
        # logger has propagate=False, records never reach root's handlers
        # no matter what level is set, so at_level() alone can't capture
        # them. Attach caplog's handler directly to the named logger
        # instead -- Python logging always calls a logger's own attached
        # handlers, independent of its propagate setting.
        target_logger = logging.getLogger("Compute-Features-Stage")
        target_logger.addHandler(caplog.handler)
        original_level = target_logger.level
        target_logger.setLevel(logging.WARNING)
        try:
            compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)
        finally:
            target_logger.removeHandler(caplog.handler)
            target_logger.setLevel(original_level)

        assert not os.path.exists(features_root)
        assert any("ZERO output rows" in r.message for r in caplog.records)

    def test_no_subhalos_produces_no_output(self, catalog_dir, cfg, features_root):
        subhalo_context = FakeSubhaloContext(pd.DataFrame(columns=["id_parents", "subhalo_id"]))
        halo_context = FakeHaloContext({})
        fetcher = FakeFetcher(catalog_dir, subhalo_context, halo_context)

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        assert not os.path.exists(features_root)

    def test_halo_lookup_is_cached_not_repeated_per_subhalo(self, catalog_dir, cfg, features_root):
        """
        get_halo_by_id() must be called once per unique halo per bucket,
        not once per subhalo -- a halo with many subhalos should not
        trigger a lookup for each one.
        """
        n_subhalos = 20
        fetcher = build_catalog(catalog_dir, {0: {sid: 1 for sid in range(n_subhalos)}})

        original_get_halo_by_id = fetcher._halo_context.get_halo_by_id
        call_count = {"n": 0}

        def counting_get_halo_by_id(halo_id):
            call_count["n"] += 1
            return original_get_halo_by_id(halo_id)

        fetcher._halo_context.get_halo_by_id = counting_get_halo_by_id

        compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)

        assert call_count["n"] == 1, (
            f"Expected exactly 1 call to get_halo_by_id for 1 halo with "
            f"{n_subhalos} subhalos, got {call_count['n']}"
        )

    def test_subhalo_spos_gets_pbc_corrected_relative_to_halo(self, catalog_dir, cfg):
        """
        When the subhalo catalog has an 'spos' column, compute_features_stage
        should PBC-correct it relative to the halo's gpos using the real
        SubhaloContext.preprocess() -- same minimum-image logic used
        elsewhere, applied here purely in-memory (never written to any
        output; this test spies on it via a monkeypatched feature function
        since SHMR itself never reads spos).
        """
        halo_id = 0
        subhalo_id = 100
        lbox = cfg.catalog.get('Lbox')

        # Halo center near one edge of the box.
        halo_gpos = np.array([2.0, 2.0, 2.0])

        # Subhalo's OWN catalog position sits near the opposite edge --
        # physically close to halo_gpos once wrapped, far apart if not.
        raw_spos = np.array([lbox - 2.0, 2.0, 2.0])

        subhalo_df = pd.DataFrame([{
            "id_parents": halo_id,
            "subhalo_id": subhalo_id,
            "spos": raw_spos,
        }])
        subhalo_context = FakeSubhaloContext(subhalo_df)
        halo_context = FakeHaloContext({halo_id: FakeHalo(halo_id=halo_id, gpos=halo_gpos)})

        # One particle so the subhalo isn't skipped as empty.
        write_particles_parquet(catalog_dir, [{
            "POS ": [0.0, 0.0, 0.0], "MASS": 1.0, "VEL ": [0.0, 0.0, 0.0],
            "parent_id": halo_id, "subhalo_id": subhalo_id,
            "parent_bucket": halo_id % 128, "parent_bucket_data": halo_id % 128,
        }])

        fetcher = FakeFetcher(catalog_dir, subhalo_context, halo_context)

        captured = {}

        def spy_compute(name, particles):
            return 1.0  # value doesn't matter for this test

        # Patch compute_features_stage's SubhaloContext.preprocess to also
        # record what it did, without changing its behavior.
        import src.data_generation.src.features_dataset.compute_features as cfs_module
        real_preprocess = cfs_module.SubhaloContext.preprocess

        def spying_preprocess(self, halo, lbox=None):
            real_preprocess(self, halo, lbox)
            captured['spos_after'] = self.data.loc[
                (self.data['id_parents'] == halo_id) & (self.data['subhalo_id'] == subhalo_id),
                'spos'
            ].iloc[0]

        import pytest as _pytest
        mp = _pytest.MonkeyPatch()
        mp.setattr(cfs_module.SubhaloContext, "preprocess", spying_preprocess)
        mp.setattr(FeatureComputer, "compute", staticmethod(spy_compute))
        try:
            compute_features_stage(fetcher, ["stellar_half_mass_radius"], cfg)
        finally:
            mp.undo()

        assert 'spos_after' in captured, "preprocess() was never called for this halo"
        corrected = np.asarray(captured['spos_after'])
        # Minimum-image correction should pull the wrapped-around position
        # to WITHIN half a box length of halo_gpos on every axis.
        delta = corrected - halo_gpos
        assert np.all(np.abs(delta) <= lbox / 2 + 1e-9)
        # And it should specifically be the wrapped value, not the raw one.
        assert not np.allclose(corrected, raw_spos)

    def test_first_n5_halos_write_to_new_bucket_when_flagged(self, catalog_dir, cfg, features_root):
        """
        write_to_new_bucket=True: computing SHMR for just the first 5 halos
        must land in a SEPARATE output tree, and must not touch (create,
        read, or modify) the original features_root at all.
        """
        # 10 halos, one subhalo each, all landing in bucket 0 (multiples of 128).
        halo_ids = [i * 128 for i in range(10)]
        catalog = {hid: {hid + 1: 1} for hid in halo_ids}
        fetcher = build_catalog(catalog_dir, catalog)

        first_5 = halo_ids[:5]

        compute_features_stage(
            fetcher, ["stellar_half_mass_radius"], cfg,
            halo_ids=first_5, write_to_new_bucket=True,
        )

        new_root = features_root + '_new'
        assert os.path.exists(new_root)
        table = read_bucket(new_root, 0)
        assert set(table["id_parents"]) == set(first_5)

        # Original location was never created at all.
        assert not os.path.exists(features_root)

    def test_first_n5_halos_replace_original_bucket_when_not_flagged(self, catalog_dir, cfg, features_root):
        """
        write_to_new_bucket=False (default): computing SHMR for the first 5
        halos merges into the SAME bucket the full run would use -- i.e.
        it replaces/upserts the touched rows in the original location,
        per the merge-not-wholesale-replace behavior already covered by
        test_partial_recompute_does_not_delete_other_subhalos_in_bucket.
        """
        halo_ids = [i * 128 for i in range(10)]
        catalog = {hid: {hid + 1: 1} for hid in halo_ids}
        fetcher = build_catalog(catalog_dir, catalog)

        first_5 = halo_ids[:5]

        compute_features_stage(
            fetcher, ["stellar_half_mass_radius"], cfg,
            halo_ids=first_5, write_to_new_bucket=False,
        )

        # Written directly to the original features_root -- no separate tree.
        assert not os.path.exists(features_root + '_new')
        table = read_bucket(features_root, 0)
        assert set(table["id_parents"]) == set(first_5)

        # Now run the remaining 5 halos, also NOT using write_to_new_bucket
        # -- the first 5 rows must survive the second write (merge, not
        # wholesale replace).
        remaining_5 = halo_ids[5:]
        compute_features_stage(
            fetcher, ["stellar_half_mass_radius"], cfg,
            halo_ids=remaining_5, force_recompute=True, write_to_new_bucket=False,
        )

        table = read_bucket(features_root, 0)
        assert set(table["id_parents"]) == set(halo_ids)