import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from rootutils import rootutils
rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.data_fetcher import MagneticumFetcherV2
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.lazy_subhalo import LazySubhaloContext, LazySubhaloParticles
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles


# ── helpers ────────────────────────────────────────────────────────────────────

def make_fake_halos(n: int = 5) -> dict:
    """Minimal raw halo dict as g3read would return."""
    return {
        "GOFF": np.arange(n) * 10,
        "GLEN": np.full(n, 10),
        "FSUB": np.arange(n) * 2,
        "NSUB": np.full(n, 2),
        "MCRI": np.full(n, 100_000_000, dtype=float),
        "RCRI": np.full(n, 100_000,     dtype=float),
    }


def make_fake_subhalos(n: int = 10) -> dict:
    return {
        "GRNR": np.arange(n) % 5,
        "MCRI": np.full(n, 1_000_000, dtype=float),
        "RCRI": np.full(n, 10_000,    dtype=float),
    }


def make_fake_snap_data(n_particles: int = 20) -> dict:
    """Minimal particle dict as g3read would return for a snapshot file."""
    return {
        "ID  ": np.arange(n_particles, dtype=np.int64),
        "POS ": np.random.uniform(-1, 1, (n_particles, 3)).astype(np.float32),
        "MASS": np.random.uniform(1e6, 1e9, n_particles).astype(np.float32),
    }


def write_fake_parquet(path: str, n_halos: int = 3, rows_per_halo: int = 5) -> None:
    """
    Write a minimal partitioned parquet dataset to path.

    Column names (POS/MASS/VEL) match what _store_particles() actually
    writes in production: g3read field names ("POS ", "MASS", "VEL ",
    padded to 4 chars) with trailing whitespace, which _load_halo_particles
    / _load_subhalo_particles strip back off (`key.rstrip()`) before
    applying their column_map={'POS': 'p', 'MASS': 'm', 'VEL': 'v'} —
    that column_map only matches these names, not arbitrary ones.
    """
    records = []
    for halo_id in range(n_halos):
        bucket = halo_id % 128
        for i in range(rows_per_halo):
            records.append({
                "parent_id":          halo_id,
                "subhalo_id":         halo_id * rows_per_halo + i,
                "parent_bucket_data": bucket,
                "parent_bucket":      bucket,
                "POS": np.random.uniform(-1, 1, 3).tolist(),
                "MASS": float(np.random.uniform(1e8, 1e12)),
                "VEL": np.random.uniform(-500, 500, 3).tolist(),
            })
    table = pa.Table.from_pandas(pd.DataFrame(records), preserve_index=False)
    pq.write_to_dataset(table, root_path=path, partition_cols=["parent_bucket"])


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_catalog():
    return OmegaConf.create({
        "path":    "/fake/path",
        "snapnum": "120",
        "fields": {
            "halo": {
                "GOFF": {"input": None, "output": None, "scaling": None, "name": "goff"},
                "GLEN": {"input": None, "output": None, "scaling": None, "name": "glen"},
                "FSUB": {"input": None, "output": None, "scaling": None, "name": "fsub"},
                "MCRI": {"input": None, "output": None, "scaling": None, "name": "m200"},
                "RCRI": {"input": None, "output": None, "scaling": None, "name": "r200"},
            },
            "subhalo":   {
                "SPOS": {"input": None, "output": None, "scaling": None, "name": "spos"},
                "SMST": {"input": None, "output": None, "scaling": None, "name": "mstr"},
                "stellar_half_mass_radius": {"input": None, "output": None, "scaling": None, "name": "shmr", "needs": {
                    "star": ["POS", "MASS"]
                }},
            },
            "particles": {},
        },
    })


@pytest.fixture
def fetcher(mock_catalog):
    with patch("g3read.read_new"), patch("g3read.yield_all_files", return_value=[]):
        f = MagneticumFetcherV2(catalog=mock_catalog)

    f._cache_loaded        = True
    f._group_base          = "/fake/group_base"
    f._snap_base           = "/fake/snap_base"
    f._goff                = np.array([0,  10, 20, 30, 40])
    f._glen                = np.array([10, 10, 10, 10, 10])
    f._fsub                = np.array([0,   2,  4,  6,  8])
    f._nsub                = np.array([2,   2,  2,  2,  2])
    f._soff                = np.array([0,   5, 10, 15, 20])
    f._slen                = np.array([5,   5,  5,  5,  5])
    f._subhalo_pid_boundaries = []
    f._pid_file_boundaries    = []
    f.halo_fields          = ["GOFF", "GLEN", "FSUB", "MCRI", "RCRI"]
    f.subhalo_fields       = ["GRNR", "MCRI", "RCRI"]
    f.particle_fields      = ["stars"]
    f.function_needs       = {}
    f.column_map           = {
        "GOFF": "goff", "GLEN": "glen", "FSUB": "fsub",
        "MCRI": "m200", "RCRI": "r200", "GRNR": "id_parents",
    }
    return f


# ── tests ──────────────────────────────────────────────────────────────────────


def _assign_all_to_zero(ids):
    """Stand-in for _assign_halo_ids/_assign_subhalo_ids: every particle -> id 0.
    These tests exercise _store_particles' output shape, not the PID matching
    (which has its own tests in test_fetcher_magneticum_v2.py)."""
    return np.zeros(len(ids), dtype=np.int32)


class TestMagneticumFetcherV2:

    # ── _load_catalog_cache ────────────────────────────────────────────────────

    def test_load_catalog_cache_sets_cache_loaded(self, mock_catalog):
        halo_data = {
            "GOFF": np.array([0]),
            "GLEN": np.array([1]),
            "FSUB": np.array([0]),
            "NSUB": np.array([1]),
        }

        def fake_read_new(base, fields, block, **kwargs):
            # All group-file reads now use ptype=-1 (no ptype flattening).
            # Dispatch on field name since block is always -1 for is_snap=False.
            if "GOFF" in fields:
                return halo_data
            if "SOFF" in fields:
                return {"SOFF": np.array([0])}
            if "SLEN" in fields:
                return {"SLEN": np.array([1])}
            if "PID " in fields:
                return {"PID ": np.array([0], dtype=np.int64)}
            return MagicMock()

        with patch("g3read.read_new", side_effect=fake_read_new), \
             patch("g3read.yield_all_files", return_value=[]), \
             patch.object(MagneticumFetcherV2, "_build_pid_file_boundaries", return_value=[]):
            f = MagneticumFetcherV2(catalog=mock_catalog)
            f._load_catalog_cache()
        assert f._cache_loaded is True

    def test_load_catalog_cache_empty_returns_early(self, mock_catalog):
        with patch("g3read.read_new", return_value={}), \
             patch("g3read.yield_all_files", return_value=[]):
            f = MagneticumFetcherV2(catalog=mock_catalog)
            result = f._load_catalog_cache()
        assert result == {}

    def test_load_catalog_cache_sets_metadata_arrays(self, mock_catalog):
        fake_catalog = {
            "GOFF": np.array([0, 10]),
            "GLEN": np.array([10, 10]),
            "FSUB": np.array([0, 2]),
            "NSUB": np.array([2, 2]),
        }
        soff_data = {"SOFF": np.array([0, 5])}
        slen_data = {"SLEN": np.array([5, 5])}
        pid_data  = {"PID ": np.arange(10, dtype=np.int64)}

        def fake_read_new(base, fields, block, **kwargs):
            # All group-file reads now use ptype=-1 (no ptype flattening).
            # Dispatch on field name since block is always -1 for is_snap=False.
            if "GOFF" in fields:
                return fake_catalog
            if "SOFF" in fields:
                return soff_data
            if "SLEN" in fields:
                return slen_data
            if "PID " in fields:
                return pid_data
            return MagicMock()

        with patch("g3read.read_new", side_effect=fake_read_new), \
             patch("g3read.yield_all_files", return_value=[]), \
             patch.object(MagneticumFetcherV2, "_build_pid_file_boundaries", return_value=[]):
            f = MagneticumFetcherV2(catalog=mock_catalog)
            f._load_catalog_cache()

        assert len(f._goff) == 2
        assert len(f._soff) == 2
        # Lazy boundary index is built instead of loading the full PID array.
        assert hasattr(f, "_subhalo_pid_boundaries")

    # ── _load_particle_cache ───────────────────────────────────────────────────

    def test_load_particle_cache_particle_conditions_is_empty_dict(self, fetcher):
        """Dead-code bug was fixed — _particle_conditions must end up as {}."""
        fetcher._load_particle_cache()
        assert fetcher._particle_conditions == {}

    def test_load_particle_cache_builds_snap_cache_config(self, fetcher, mock_catalog):
        # Patch MAGNETICUM_PART_TYPES so the test is independent of the real
        # mapping and works regardless of whether the key is "star" or "stars".
        fake_part_types = {"star": 4}
        OmegaConf.set_struct(mock_catalog, False)
        mock_catalog.fields.particles = OmegaConf.create({
            "star": {"POS": {}, "MASS": {}}
        })
        fetcher.catalog = mock_catalog
        with patch("src.data_generation.src.data_fetcher.MAGNETICUM_PART_TYPES",
                   fake_part_types):
            fetcher._load_particle_cache()
        assert "star" in fetcher._snap_cache_config
        # ID must be appended and padded to 4 chars
        assert "ID  " in fetcher._snap_cache_config["star"]["fields_padded"]
        # ptype_int must resolve from the patched mapping
        assert fetcher._snap_cache_config["star"]["ptype_int"] == 4

    # ── _get_halos ────────────────────────────────────────────────────────────

    def test_get_halos_returns_halo_context(self, fetcher):
        with patch("g3read.read_new", return_value=make_fake_halos(5)):
            result = fetcher.get_halos()
        assert isinstance(result, HaloContext)

    def test_get_halos_correct_row_count(self, fetcher):
        with patch("g3read.read_new", return_value=make_fake_halos(5)):
            result = fetcher.get_halos()
        assert len(result.data) == 5

    def test_get_halos_halo_ids_assigned_sequentially(self, fetcher):
        with patch("g3read.read_new", return_value=make_fake_halos(5)):
            result = fetcher.get_halos()
        assert sorted(result.data["halo_id"].tolist()) == list(range(5))

    def test_get_halos_subhalo_ids_correct_for_halo_0(self, fetcher):
        with patch("g3read.read_new", return_value=make_fake_halos(5)):
            result = fetcher.get_halos()
        row = result.data.loc[result.data["halo_id"] == 0, "subhalo_ids"].iloc[0]
        # fetcher._fsub[0]=0, fetcher._nsub[0]=2 → [0, 1]
        assert list(row) == [0, 1]

    def test_get_halos_fsub_injected_when_missing(self, fetcher):
        fetcher.halo_fields = ["MCRI", "RCRI"]  # FSUB absent
        with patch("g3read.read_new", return_value=make_fake_halos(5)):
            fetcher.get_halos()
        assert "FSUB" in fetcher.halo_fields

    def test_get_halos_empty_raises(self, fetcher):
        # HaloContext.__init__ forbids empty DataFrames — _get_halos must
        # propagate that exception when the catalog returns no halos.
        with patch("g3read.read_new", return_value={}):
            with pytest.raises(Exception, match="empty"):
                fetcher.get_halos()

    def test_get_halos_triggers_cache_load_if_not_loaded(self, fetcher):
        fetcher._cache_loaded = False
        with patch.object(fetcher, "_load_catalog_cache") as mock_load, \
             patch("g3read.read_new", return_value=make_fake_halos(5)):
            fetcher.get_halos()
        mock_load.assert_called_once()

    # ── _get_subhalos ─────────────────────────────────────────────────────────

    def test_get_subhalos_returns_lazy_context_from_parquet(self, fetcher):
        with tempfile.TemporaryDirectory() as tmpdir:
            write_fake_parquet(os.path.join(tmpdir, "subhalos_parquet"))
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            result = fetcher.get_subhalos()
        assert isinstance(result, LazySubhaloContext)

    def test_get_subhalos_no_files_raises(self, fetcher):
        # No files yielded -> processed_batches empty -> SubhaloContext(empty df)
        # -> SubhaloContext.__init__ raises, same as HaloContext for empty data.
        with tempfile.TemporaryDirectory() as tmpdir:
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            with patch("g3read.yield_all_files", return_value=[]), \
                 patch("g3read.read_new", return_value=make_fake_subhalos()):
                with pytest.raises((KeyError, Exception)):
                    fetcher.get_subhalos()

    def test_get_subhalos_grnr_injected_when_missing(self, fetcher):
        # GRNR injection happens before the file loop, so it fires even when
        # the pipeline subsequently raises on an empty result.
        fetcher.subhalo_fields = ["MCRI"]
        with tempfile.TemporaryDirectory() as tmpdir:
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            with patch("g3read.yield_all_files", return_value=[]):
                with pytest.raises((KeyError, Exception)):
                    fetcher.get_subhalos()
        assert "GRNR" in fetcher.subhalo_fields

    # ── _store_particles ──────────────────────────────────────────────────────

    def test_store_particles_loads_from_existing_parquet(self, fetcher):
        with tempfile.TemporaryDirectory() as tmpdir:
            particles_path = os.path.join(tmpdir, "particles_parquet")
            write_fake_parquet(particles_path)
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            halo = MagicMock()
            halo.halo_id = 0  # concrete int: bucket = halo_id % 128 needs a real value,
                              # not a MagicMock, to be usable as a parquet filter
            result = fetcher._get_p_halo(halo=halo)
        # _load_halo_particles (the path taken when particles_parquet already
        # exists) always returns SubhaloParticles, never LazySubhaloParticles
        # — that type is only produced by _store_particles() on a fresh build.
        assert isinstance(result, SubhaloParticles)

    def test_store_particles_loads_particle_cache_even_if_catalog_cache_already_loaded(
        self, fetcher, mock_catalog
    ):
        """
        Regression: _get_halos()/_get_subhalos() call _load_catalog_cache()
        alone and set _cache_loaded=True without touching the particle
        cache. _store_particles() must not gate _load_particle_cache()
        behind that same flag — doing so previously meant _snap_cache_config
        was never built whenever halos/subhalos were fetched before
        particles (production traceback: "'MagneticumFetcherV2' object has
        no attribute '_snap_cache_config'").
        """
        OmegaConf.set_struct(mock_catalog, False)
        mock_catalog.fields.particles = OmegaConf.create({
            "stars": {"POS": {}, "MASS": {}}
        })
        fetcher.catalog = mock_catalog
        # Simulates a fetcher that already had _get_halos()/_get_subhalos()
        # called on it: catalog cache loaded, particle cache not.
        fetcher._cache_loaded = True
        assert not hasattr(fetcher, "_particle_cache_loaded")

        snap_data = make_fake_snap_data(5)

        def fake_read_new(path, fields, ptype, **kwargs):
            return {4: snap_data}

        with tempfile.TemporaryDirectory() as tmpdir:
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            with patch("src.data_generation.src.data_fetcher.MAGNETICUM_PART_TYPES",
                       {"stars": 4}), \
                 patch("g3read.yield_all_files", return_value=["snap.hdf5"]), \
                 patch("g3read.read_new", side_effect=fake_read_new), \
                 patch("pyarrow.parquet.write_to_dataset"), \
                 patch("src.data_generation.src.data_fetcher.LazySubhaloParticles"), \
                 patch.object(fetcher, "_run_particle_pipeline", side_effect=lambda d: d), \
                 patch.object(fetcher, "_assign_halo_ids", side_effect=_assign_all_to_zero), \
                 patch.object(fetcher, "_assign_subhalo_ids", side_effect=_assign_all_to_zero):
                fetcher._store_particles()

        assert fetcher._particle_cache_loaded is True
        assert "stars" in fetcher._snap_cache_config

    def test_store_particles_includes_subhalo_id_column(self, fetcher):
        """Regression: subhalo_id must be written into the parquet output."""
        snap_data = make_fake_snap_data(20)

        # _subhalo_index is now 2-tuples (start, end) — no third element.
        fetcher._snap_cache_config = {
            "stars": {"fields_padded": ["ID  ", "POS ", "MASS"], "ptype_int": 4}
        }
        # _cache_loaded and _particle_cache_loaded are independent flags —
        # set both so _store_particles() doesn't rebuild the pre-seeded
        # _snap_cache_config above via the real _load_particle_cache().
        fetcher._particle_cache_loaded = True

        written = {}

        def fake_read_new(path, fields, ptype, **kwargs):
            return {4: snap_data}

        def capture_write(table, root_path, **kwargs):
            written["table"] = table

        with tempfile.TemporaryDirectory() as tmpdir:
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            with patch("g3read.yield_all_files", return_value=["snap.hdf5"]), \
                 patch("g3read.read_new", side_effect=fake_read_new), \
                 patch("pyarrow.parquet.write_to_dataset", side_effect=capture_write), \
                 patch("src.data_generation.src.data_fetcher.LazySubhaloParticles"), \
                 patch.object(fetcher, "_run_particle_pipeline", side_effect=lambda d: d), \
                 patch.object(fetcher, "_assign_halo_ids", side_effect=_assign_all_to_zero), \
                 patch.object(fetcher, "_assign_subhalo_ids", side_effect=_assign_all_to_zero):
                fetcher._store_particles()

        assert "table" in written, "write_to_dataset was never called"
        assert "subhalo_id" in written["table"].schema.names

    def test_store_particles_parent_bucket_data_independent_copy(self, fetcher):
        """Regression: parent_bucket_data must be a copy, not an alias."""
        snap_data = make_fake_snap_data(10)

        fetcher._snap_cache_config = {
            "stars": {"fields_padded": ["ID  ", "POS ", "MASS"], "ptype_int": 4}
        }
        # See test_store_particles_includes_subhalo_id_column: both cache
        # flags must be set so _store_particles() doesn't overwrite the
        # pre-seeded _snap_cache_config above via the real _load_particle_cache().
        fetcher._particle_cache_loaded = True

        written = {}

        def fake_read_new(path, fields, ptype, **kwargs):
            return {4: snap_data}

        def capture_write(table, root_path, **kwargs):
            written["table"] = table

        with tempfile.TemporaryDirectory() as tmpdir:
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            with patch("g3read.yield_all_files", return_value=["snap.hdf5"]), \
                 patch("g3read.read_new", side_effect=fake_read_new), \
                 patch("pyarrow.parquet.write_to_dataset", side_effect=capture_write), \
                 patch("src.data_generation.src.data_fetcher.LazySubhaloParticles"), \
                 patch.object(fetcher, "_run_particle_pipeline", side_effect=lambda d: d), \
                 patch.object(fetcher, "_assign_halo_ids", side_effect=_assign_all_to_zero), \
                 patch.object(fetcher, "_assign_subhalo_ids", side_effect=_assign_all_to_zero):
                fetcher._store_particles()

        assert "table" in written, "write_to_dataset was never called"
        tbl = written["table"].to_pandas()
        np.testing.assert_array_equal(
            tbl["parent_bucket"].values,
            tbl["parent_bucket_data"].values,
        )

    # ── _get_p_halo ────────────────────────────────────────────────────────────

    def test_get_p_halo_raises_without_halo_kwarg(self, fetcher):
        with pytest.raises(ValueError, match="halo must be provided"):
            fetcher.get_p_halo()

    def test_get_p_halo_returns_lazy_particles_from_existing_parquet(self, fetcher):
        with tempfile.TemporaryDirectory() as tmpdir:
            particles_path = os.path.join(tmpdir, "particles_parquet")
            write_fake_parquet(particles_path)
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            halo = MagicMock()
            halo.halo_id = 0  # concrete int: bucket = halo_id % 128 needs a real value,
                              # not a MagicMock, to be usable as a parquet filter
            result = fetcher.get_p_halo(halo=halo)
        # See test_store_particles_loads_from_existing_parquet: the
        # existing-parquet path returns SubhaloParticles, not
        # LazySubhaloParticles.
        assert isinstance(result, SubhaloParticles)

    def test_get_p_halo_triggers_store_particles_when_no_parquet(self, fetcher):
        with tempfile.TemporaryDirectory() as tmpdir:
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            with patch.object(fetcher, "_store_particles",
                              return_value=MagicMock(spec=LazySubhaloParticles)) as mock_store:
                fetcher.get_p_halo(halo=MagicMock())
            mock_store.assert_called_once()

    def test_get_p_halo_triggers_cache_load_if_not_loaded(self, fetcher):
        fetcher._cache_loaded = False
        with patch.object(fetcher, "_load_catalog_cache") as mock_load, \
             patch.object(fetcher, "_load_particle_cache"),              \
             patch.object(fetcher, "_store_particles",
                          return_value=MagicMock(spec=LazySubhaloParticles)):
            with tempfile.TemporaryDirectory() as tmpdir:
                OmegaConf.set_struct(fetcher.catalog, False)
                fetcher.catalog.path = tmpdir
                fetcher.get_p_halo(halo=MagicMock())
        mock_load.assert_called_once()