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

from src.data_generation.src.data_fetcher import MagneticumFetcher
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
        f = MagneticumFetcher(catalog=mock_catalog)

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

class TestMagneticumFetcher:

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
             patch.object(MagneticumFetcher, "_build_pid_file_boundaries", return_value=[]):
            f = MagneticumFetcher(catalog=mock_catalog)
            f._load_catalog_cache()
        assert f._cache_loaded is True

    def test_load_catalog_cache_empty_returns_early(self, mock_catalog):
        with patch("g3read.read_new", return_value={}), \
             patch("g3read.yield_all_files", return_value=[]):
            f = MagneticumFetcher(catalog=mock_catalog)
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
             patch.object(MagneticumFetcher, "_build_pid_file_boundaries", return_value=[]):
            f = MagneticumFetcher(catalog=mock_catalog)
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

    # ── _build_pid_file_boundaries ─────────────────────────────────────────────

    def test_build_pid_file_boundaries_empty_files(self, fetcher):
        with patch("g3read.yield_all_files", return_value=[]):
            result = fetcher._build_pid_file_boundaries()
        assert result == []

    def test_build_pid_file_boundaries_accumulates_offsets(self, fetcher):
        fake_file = MagicMock()
        fake_file.read.return_value = np.arange(10, dtype=np.int64)

        with patch("g3read.yield_all_files", return_value=["f1.hdf5", "f2.hdf5"]), \
             patch("g3read.GadgetFile", return_value=fake_file):
            result = fetcher._build_pid_file_boundaries()

        assert len(result) == 2
        assert result[0]["start"] == 0
        assert result[0]["end"]   == 10
        assert result[1]["start"] == 10
        assert result[1]["end"]   == 20

    def test_build_pid_file_boundaries_skips_empty_pid_blocks(self, fetcher):
        fake_file = MagicMock()
        fake_file.read.return_value = np.array([])

        with patch("g3read.yield_all_files", return_value=["f1.hdf5"]), \
             patch("g3read.GadgetFile", return_value=fake_file):
            result = fetcher._build_pid_file_boundaries()

        assert result == []

    # ── _get_pid_for_entity ────────────────────────────────────────────────────

    def test_get_pid_for_entity_zero_length_returns_empty(self, fetcher):
        result = fetcher._get_pid_for_entity(offset=0, length=0)
        assert len(result) == 0
        # FIX: np.array([]) defaults to float64; source must use dtype=np.int64
        assert result.dtype == np.int64

    def test_get_pid_for_entity_no_boundaries_returns_empty(self, fetcher):
        fetcher._pid_file_boundaries = []
        result = fetcher._get_pid_for_entity(offset=0, length=5)
        assert len(result) == 0

    def test_get_pid_for_entity_reads_correct_slice(self, fetcher):
        fake_file = MagicMock()
        fake_file.read.return_value = np.arange(20, dtype=np.int64)

        fetcher._pid_file_boundaries = [{"file": "f1", "start": 0, "end": 20}]

        with patch("g3read.GadgetFile", return_value=fake_file):
            result = fetcher._get_pid_for_entity(offset=5, length=3)

        np.testing.assert_array_equal(result, [5, 6, 7])

    def test_get_pid_for_entity_spans_two_files(self, fetcher):
        fake_file_1 = MagicMock()
        fake_file_1.read.return_value = np.arange(10, dtype=np.int64)
        fake_file_2 = MagicMock()
        fake_file_2.read.return_value = np.arange(10, 20, dtype=np.int64)

        fetcher._pid_file_boundaries = [
            {"file": "f1", "start":  0, "end": 10},
            {"file": "f2", "start": 10, "end": 20},
        ]

        def fake_gadget(path, **kwargs):
            return fake_file_1 if path == "f1" else fake_file_2

        with patch("g3read.GadgetFile", side_effect=fake_gadget):
            result = fetcher._get_pid_for_entity(offset=8, length=4)

        np.testing.assert_array_equal(result, [8, 9, 10, 11])

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

    # ── _build_halo_intervals ─────────────────────────────────────────────────

    def test_build_halo_intervals_skips_zero_length_halos(self, fetcher):
        fetcher._glen = np.array([0, 10, 0])
        fetcher._goff = np.array([0,  0, 0])

        # Only called for halo 1 (length=10); halos 0 and 2 are length=0 and
        # must be skipped by _build_halo_intervals before calling _get_pid_for_entity.
        def fake_get_pid(offset, length):
            assert length > 0, "_get_pid_for_entity must not be called with length=0"
            return np.arange(offset, offset + length, dtype=np.int64)

        with patch.object(fetcher, "_get_pid_for_entity", side_effect=fake_get_pid):
            fetcher._build_halo_intervals()

        assert len(fetcher._halo_intervals) == 1
        assert fetcher._halo_intervals[0][2] == 1   # halo_id=1

    def test_build_halo_intervals_sorted_by_min_pid(self, fetcher):
        fetcher._glen = np.array([5, 5, 5])
        fetcher._goff = np.array([0, 5, 10])

        def fake_get_pid(offset, length):
            return np.arange(offset, offset + length, dtype=np.int64)

        with patch.object(fetcher, "_get_pid_for_entity", side_effect=fake_get_pid):
            fetcher._build_halo_intervals()

        starts = [iv[0] for iv in fetcher._halo_intervals]
        assert starts == sorted(starts)

    def test_build_halo_intervals_stores_min_max_halo_id(self, fetcher):
        fetcher._glen = np.array([3])
        fetcher._goff = np.array([0])

        pids = np.array([7, 3, 9], dtype=np.int64)
        with patch.object(fetcher, "_get_pid_for_entity", return_value=pids):
            fetcher._build_halo_intervals()

        iv = fetcher._halo_intervals[0]
        assert iv[0] == 3   # min pid
        assert iv[1] == 9   # max pid
        assert iv[2] == 0   # halo_id

    # ── _assign_halo_ids ──────────────────────────────────────────────────────

    def test_assign_halo_ids_empty_input_returns_empty(self, fetcher):
        fetcher._halo_intervals = [(0, 10, 0)]
        result = fetcher._assign_halo_ids(np.array([], dtype=np.int64))
        assert len(result) == 0

    def test_assign_halo_ids_assigns_correct_halo(self, fetcher):
        fetcher._halo_intervals = [(0, 9, 0), (10, 19, 1)]
        ids    = np.array([5, 15], dtype=np.int64)
        result = fetcher._assign_halo_ids(ids)
        np.testing.assert_array_equal(result, [0, 1])

    def test_assign_halo_ids_unmatched_returns_minus_one(self, fetcher):
        fetcher._halo_intervals = [(0, 9, 0)]
        ids    = np.array([50], dtype=np.int64)
        result = fetcher._assign_halo_ids(ids)
        assert result[0] == -1

    def test_assign_halo_ids_early_exit_past_max(self, fetcher):
        fetcher._halo_intervals = [(0, 9, 0), (100, 200, 1)]
        ids    = np.array([5], dtype=np.int64)
        result = fetcher._assign_halo_ids(ids)
        # Halo 1 start (100) > ids.max() (5) → early exit; only halo 0 can match
        assert result[0] == 0

    # ── _build_subhalo_index ──────────────────────────────────────────────────

    def test_build_subhalo_index_length_matches_soff(self, fetcher):
        fetcher._build_subhalo_index()
        assert len(fetcher._subhalo_index) == len(fetcher._soff)

    def test_build_subhalo_index_tuples_are_pairs(self, fetcher):
        # Source (doc 3) stores (start, end) — a 2-tuple only.
        # The earlier 3-tuple design was reverted; third element removed.
        fetcher._build_subhalo_index()
        for item in fetcher._subhalo_index:
            assert len(item) == 2

    def test_build_subhalo_index_correct_slice_boundaries(self, fetcher):
        fetcher._soff = np.array([0, 5])
        fetcher._slen = np.array([5, 3])
        fetcher._build_subhalo_index()
        start0, end0 = fetcher._subhalo_index[0]
        start1, end1 = fetcher._subhalo_index[1]
        assert (start0, end0) == (0, 5)
        assert (start1, end1) == (5, 8)

    # ── _build_subhalo_pid_arrays ──────────────────────────────────────────────

    def test_build_subhalo_pid_arrays_correct_mapping(self, fetcher):
        fetcher._soff = np.array([0, 3])
        fetcher._slen = np.array([3, 2])
        fetcher._build_subhalo_index()

        all_pids = np.array([10, 11, 12, 13, 14], dtype=np.int64)

        def fake_get_subhalo_pid(offset, length):
            return all_pids[offset:offset + length]

        with patch.object(fetcher, "_get_subhalo_pid_for_entity",
                          side_effect=fake_get_subhalo_pid):
            fetcher._build_subhalo_pid_arrays()

        def lookup(pid):
            i = np.searchsorted(fetcher._subhalo_sorted_pids, pid)
            if i < len(fetcher._subhalo_sorted_pids) and fetcher._subhalo_sorted_pids[i] == pid:
                return int(fetcher._subhalo_sorted_ids[i])
            return -1

        assert lookup(10) == 0
        assert lookup(12) == 0
        assert lookup(13) == 1
        assert lookup(14) == 1

    def test_build_subhalo_pid_arrays_no_undefined_attributes(self, fetcher):
        """Regression: no reference to self._pid or self._subhalos_pid."""
        fetcher._build_subhalo_index()
        with patch.object(fetcher, "_get_subhalo_pid_for_entity",
                          return_value=np.array([], dtype=np.int64)):
            fetcher._build_subhalo_pid_arrays()
        assert hasattr(fetcher, "_subhalo_sorted_pids")
        assert hasattr(fetcher, "_subhalo_sorted_ids")

    # ── _assign_subhalo_ids ───────────────────────────────────────────────────

    def test_assign_subhalo_ids_empty_input_returns_empty(self, fetcher):
        fetcher._subhalo_sorted_pids = np.array([], dtype=np.int64)
        fetcher._subhalo_sorted_ids  = np.array([], dtype=np.int32)
        result = fetcher._assign_subhalo_ids(np.array([], dtype=np.int64))
        assert len(result) == 0

    def test_assign_subhalo_ids_correct_mapping(self, fetcher):
        fetcher._subhalo_sorted_pids = np.array([10, 11, 12], dtype=np.int64)
        fetcher._subhalo_sorted_ids  = np.array([0, 0, 1],   dtype=np.int32)
        ids    = np.array([10, 12, 99], dtype=np.int64)
        result = fetcher._assign_subhalo_ids(ids)
        assert result[0] == 0
        assert result[1] == 1
        assert result[2] == -1   # unassigned

    def test_assign_subhalo_ids_builds_arrays_if_absent(self, fetcher):
        fetcher._build_subhalo_index()
        if hasattr(fetcher, "_subhalo_sorted_pids"):
            del fetcher._subhalo_sorted_pids
        with patch.object(fetcher, "_get_subhalo_pid_for_entity",
                          return_value=np.array([], dtype=np.int64)):
            fetcher._assign_subhalo_ids(np.array([0, 1, 2], dtype=np.int64))
        assert hasattr(fetcher, "_subhalo_sorted_pids")

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

        fetcher._halo_intervals    = [(0, 4, 0)]
        fetcher._subhalo_index     = [(0, 5)]
        fetcher._subhalo_sorted_pids = np.arange(5, dtype=np.int64)
        fetcher._subhalo_sorted_ids  = np.zeros(5, dtype=np.int32)

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
                 patch.object(fetcher, "_build_halo_intervals"), \
                 patch.object(fetcher, "_build_subhalo_index"), \
                 patch.object(fetcher, "_build_subhalo_pid_arrays"):
                fetcher._store_particles()

        assert fetcher._particle_cache_loaded is True
        assert "stars" in fetcher._snap_cache_config

    def test_store_particles_includes_subhalo_id_column(self, fetcher):
        """Regression: subhalo_id must be written into the parquet output."""
        snap_data = make_fake_snap_data(20)

        # _subhalo_index is now 2-tuples (start, end) — no third element.
        fetcher._halo_intervals    = [(0, 19, 0)]
        fetcher._subhalo_index     = [(0, 20)]
        fetcher._subhalo_sorted_pids = np.arange(20, dtype=np.int64)
        fetcher._subhalo_sorted_ids  = np.zeros(20, dtype=np.int32)
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
                 patch.object(fetcher, "_build_halo_intervals"), \
                 patch.object(fetcher, "_build_subhalo_index"), \
                 patch.object(fetcher, "_build_subhalo_pid_arrays"):
                # _store_particles() unconditionally rebuilds these three via
                # the calls above; patch them out so the pre-seeded
                # _halo_intervals / _subhalo_index / _subhalo_sorted_pids
                # above are what's actually exercised, instead of being
                # silently overwritten by an empty rebuild (no PID boundary
                # files exist in this mock setup).
                fetcher._store_particles()

        assert "table" in written, "write_to_dataset was never called"
        assert "subhalo_id" in written["table"].schema.names

    def test_store_particles_parent_bucket_data_independent_copy(self, fetcher):
        """Regression: parent_bucket_data must be a copy, not an alias."""
        snap_data = make_fake_snap_data(10)

        fetcher._halo_intervals    = [(0, 9, 0)]
        fetcher._subhalo_index     = [(0, 10)]
        fetcher._subhalo_sorted_pids = np.arange(10, dtype=np.int64)
        fetcher._subhalo_sorted_ids  = np.zeros(10, dtype=np.int32)
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
                 patch.object(fetcher, "_build_halo_intervals"), \
                 patch.object(fetcher, "_build_subhalo_index"), \
                 patch.object(fetcher, "_build_subhalo_pid_arrays"):
                # See test_store_particles_includes_subhalo_id_column: these
                # three are unconditionally rebuilt by _store_particles() and
                # would otherwise overwrite the pre-seeded state above with
                # an empty rebuild.
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