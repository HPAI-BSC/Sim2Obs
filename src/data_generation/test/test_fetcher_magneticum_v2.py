import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from rootutils import rootutils
rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.data_fetcher import MagneticumFetcher, MagneticumFetcherV2
from src.data_generation.src.lazy_subhalo import LazySubhaloParticles

# Reuse the exact fixtures/helpers already used to test MagneticumFetcher, so
# V2's tests stay aligned with V1's conventions and fixture shapes.
from src.data_generation.test.test_fetcher_magneticum import (
    mock_catalog,
    make_fake_snap_data,
    write_fake_parquet,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def fetcher(mock_catalog):
    with patch("g3read.read_new"), patch("g3read.yield_all_files", return_value=[]):
        f = MagneticumFetcherV2(catalog=mock_catalog)

    f._cache_loaded = True
    f._group_base = "/fake/group_base"
    f._snap_base = "/fake/snap_base"
    f._goff = np.array([0, 10, 20, 30, 40])
    f._glen = np.array([10, 10, 10, 10, 10])
    f._fsub = np.array([0, 2, 4, 6, 8])
    f._nsub = np.array([2, 2, 2, 2, 2])
    f._soff = np.array([0, 5, 10, 15, 20])
    f._slen = np.array([5, 5, 5, 5, 5])
    f._subhalo_pid_boundaries = []
    f._pid_file_boundaries = []
    f.halo_fields = ["GOFF", "GLEN", "FSUB", "MCRI", "RCRI"]
    f.subhalo_fields = ["GRNR", "MCRI", "RCRI"]
    f.particle_fields = ["stars"]
    f.function_needs = {}
    f.column_map = {
        "GOFF": "goff", "GLEN": "glen", "FSUB": "fsub",
        "MCRI": "m200", "RCRI": "r200", "GRNR": "id_parents",
    }
    return f


class _FakeGadgetFile:
    """Records how many times a "PID " block is actually read from disk,
    so tests can assert V2 reads each file once instead of once per entity."""
    read_calls = []

    def __init__(self, path, is_snap=False):
        self.path = path

    def read(self, block, *args, **kwargs):
        if block == "PID ":
            _FakeGadgetFile.read_calls.append(self.path)
            return _FakeGadgetFile.pid_by_file.get(self.path, np.array([], dtype=np.int64))
        return None


# ── tests ──────────────────────────────────────────────────────────────────────

class TestMagneticumFetcherV2:

    # ── _build_pid_file_boundaries / _build_subhalo_pid_file_boundaries ───────
    # V2 has no per-entity lazy readers left to index (see _build_pid_index),
    # so these are no-ops kept only so the inherited _load_catalog_cache()
    # doesn't perform a wasted boundary-only scan.

    def test_build_pid_file_boundaries_is_noop(self, fetcher):
        assert fetcher._build_pid_file_boundaries() == []

    def test_build_subhalo_pid_file_boundaries_is_noop(self, fetcher):
        assert fetcher._build_subhalo_pid_file_boundaries() == []

    # ── _build_pid_index ────────────────────────────────────────────────────

    def test_build_pid_index_empty_files_produces_empty_arrays(self, fetcher):
        with patch("g3read.yield_all_files", return_value=[]):
            fetcher._build_pid_index()
        assert len(fetcher._halo_sorted_pids) == 0
        assert len(fetcher._subhalo_sorted_pids) == 0
        assert fetcher._pid_index_built is True

    def test_build_pid_index_reads_each_group_file_exactly_once(self, fetcher):
        """The core I/O fix: V1 re-reads a file's PID block once per halo/
        subhalo whose range overlaps it; V2 must read each file exactly once
        regardless of how many halos/subhalos it contains."""
        # 5 halos (glen=10 each) -> 50 positions total, split across 2 files.
        fetcher._goff = np.array([0, 10, 20, 30, 40])
        fetcher._glen = np.array([10, 10, 10, 10, 10])
        fetcher._soff = np.array([0, 10, 20, 30, 40])
        fetcher._slen = np.array([10, 10, 10, 10, 10])

        pid_by_file = {
            "f1": np.arange(0, 25, dtype=np.int64),
            "f2": np.arange(25, 50, dtype=np.int64),
        }
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = pid_by_file

        with patch("g3read.yield_all_files", return_value=["f1", "f2"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            fetcher._build_pid_index()

        assert _FakeGadgetFile.read_calls == ["f1", "f2"], (
            "Expected exactly one PID read per group file "
            f"(got {len(_FakeGadgetFile.read_calls)} reads for 5 halos / 5 subhalos)"
        )
        assert len(fetcher._halo_sorted_pids) == 50

    def test_build_pid_index_is_idempotent(self, fetcher):
        """_store_particles() calls _build_halo_intervals(), _build_subhalo_index()
        and _build_subhalo_pid_arrays() in sequence; under V2 all three delegate
        to _build_pid_index(), which must only do real work once."""
        fetcher._goff = np.array([0, 5])
        fetcher._glen = np.array([5, 5])
        fetcher._soff = np.array([0, 5])
        fetcher._slen = np.array([5, 5])

        pid_by_file = {"f1": np.arange(10, dtype=np.int64)}
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = pid_by_file

        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            fetcher._build_halo_intervals()
            fetcher._build_subhalo_index()
            fetcher._build_subhalo_pid_arrays()

        assert len(_FakeGadgetFile.read_calls) == 1, (
            "Three delegating calls (mirroring _store_particles) must only "
            f"trigger one real PID read, got {len(_FakeGadgetFile.read_calls)}"
        )

    def test_build_pid_index_halo_assignment_exact_partition(self, fetcher):
        """GOFF/GLEN partition the PID block with no gaps -> plain run-length
        expansion must give exact halo ownership per position."""
        fetcher._goff = np.array([0, 3])
        fetcher._glen = np.array([3, 2])
        fetcher._soff = np.array([0])
        fetcher._slen = np.array([0])

        pid_values = np.array([100, 5, 50, 60, 150], dtype=np.int64)
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": pid_values}

        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            fetcher._build_pid_index()

        assigned = fetcher._assign_halo_ids(np.array([100, 5, 50, 60, 150], dtype=np.int64))
        np.testing.assert_array_equal(assigned, [0, 0, 0, 1, 1])

    def test_build_pid_index_subhalo_gap_yields_minus_one(self, fetcher):
        """SOFF/SLEN nest inside their halo's range but may leave a gap
        (diffuse, unbound halo particles not owned by any subhalo)."""
        fetcher._goff = np.array([0])
        fetcher._glen = np.array([5])
        fetcher._soff = np.array([0])   # subhalo 0 only covers positions 0-2
        fetcher._slen = np.array([3])   # positions 3-4 belong to no subhalo

        pid_values = np.array([10, 11, 12, 13, 14], dtype=np.int64)
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": pid_values}

        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            fetcher._build_pid_index()

        # _assign_subhalo_ids is inherited unchanged from MagneticumFetcher —
        # it already does exact searchsorted matching, it just now runs
        # against an index built without the redundant per-entity re-reads.
        assigned = fetcher._assign_subhalo_ids(np.array([10, 13, 14], dtype=np.int64))
        np.testing.assert_array_equal(assigned, [0, -1, -1])

    def test_build_pid_index_raises_on_glen_mismatch(self, fetcher):
        """Guard: if GLEN no longer sums to the PID block length, the
        contiguous-partition assumption this method relies on is broken and
        must fail loudly rather than silently misassign particles."""
        fetcher._goff = np.array([0])
        fetcher._glen = np.array([999])   # deliberately wrong
        fetcher._soff = np.array([0])
        fetcher._slen = np.array([0])

        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": np.arange(5, dtype=np.int64)}

        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            with pytest.raises(ValueError, match="does not match PID"):
                fetcher._build_pid_index()

    # ── _assign_halo_ids: correctness regression vs V1's interval bug ─────────

    def test_assign_halo_ids_empty_input_returns_empty(self, fetcher):
        fetcher._pid_index_built = True
        fetcher._halo_sorted_pids = np.array([1, 2, 3], dtype=np.int64)
        fetcher._halo_sorted_ids = np.array([0, 0, 1], dtype=np.int32)
        result = fetcher._assign_halo_ids(np.array([], dtype=np.int64))
        assert len(result) == 0

    def test_assign_halo_ids_unmatched_returns_minus_one(self, fetcher):
        fetcher._pid_index_built = True
        fetcher._halo_sorted_pids = np.array([5, 50, 100], dtype=np.int64)
        fetcher._halo_sorted_ids = np.array([0, 0, 0], dtype=np.int32)
        result = fetcher._assign_halo_ids(np.array([999], dtype=np.int64))
        assert result[0] == -1

    def test_assign_halo_ids_builds_index_lazily_if_not_built(self, fetcher):
        assert not hasattr(fetcher, "_pid_index_built")
        with patch("g3read.yield_all_files", return_value=[]):
            fetcher._assign_halo_ids(np.array([1], dtype=np.int64))
        assert fetcher._pid_index_built is True

    def test_assign_halo_ids_rejects_false_positive_that_v1_interval_would_accept(self, fetcher):
        """
        The actual bug, reproduced directly: halo 0's true PIDs are
        {5, 50, 100} -> V1's interval for halo 0 is [5, 100]. Particle 77 was
        never a member of ANY halo (not present in the PID block at all),
        but V1's range test still matches it to halo 0 purely because 77
        falls inside [5, 100].

        V1's own _assign_halo_ids (unchanged, exercised here directly)
        confirms the false positive; V2's exact-match _assign_halo_ids on
        the same underlying data must reject it (-1).
        """
        # halo0 true pids = {100, 5, 50} -> V1 interval [5, 100]
        # halo1 true pids = {200, 250}   -> V1 interval [200, 250] (no overlap)
        pid_values = np.array([100, 5, 50, 200, 250], dtype=np.int64)
        probe_id = np.array([77], dtype=np.int64)  # not a real particle anywhere, but 5<=77<=100

        # V1 behaviour (unmodified MagneticumFetcher logic) on the same data.
        v1_intervals = [(5, 100, 0), (200, 250, 1)]  # (min_pid, max_pid, halo_id), sorted by min
        v1_fetcher = MagicMock(spec=MagneticumFetcher)
        v1_fetcher._halo_intervals = v1_intervals
        v1_result = MagneticumFetcher._assign_halo_ids(v1_fetcher, probe_id)
        assert v1_result[0] == 0, "sanity check: V1's interval approximation should reproduce the false positive"

        # V2 behaviour on equivalent data via the real (mocked-I/O) build path.
        fetcher._goff = np.array([0, 3])
        fetcher._glen = np.array([3, 2])
        fetcher._soff = np.array([0, 3])
        fetcher._slen = np.array([3, 2])
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": pid_values}
        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            fetcher._build_pid_index()

        v2_result = fetcher._assign_halo_ids(probe_id)
        assert v2_result[0] == -1, "V2 must not reproduce V1's false-positive interval match"

    # ── end-to-end through _store_particles (inherited, unmodified) ───────────

    def test_store_particles_assigns_correct_halo_where_v1_interval_would_fail(self, fetcher):
        """
        End-to-end regression through the real ingestion entry point.
        Star snapshot contains an ID (77) that isn't a true member of any
        halo but sits inside halo 0's V1 interval [5, 100]; under V2 it must
        be dropped (mask=False), not written into halo 0's output rows.
        """
        fetcher._goff = np.array([0, 3])
        fetcher._glen = np.array([3, 2])
        fetcher._soff = np.array([0, 3])
        fetcher._slen = np.array([3, 2])
        fetcher._snap_cache_config = {
            "stars": {"fields_padded": ["ID  ", "POS ", "MASS"], "ptype_int": 4}
        }

        # halo0 true pids = {100, 5, 50} -> V1 interval [5, 100]
        # halo1 true pids = {200, 250}   -> V1 interval [200, 250] (no overlap)
        group_pid_values = np.array([100, 5, 50, 200, 250], dtype=np.int64)
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"group_f1": group_pid_values}

        snap_data = {
            "ID  ": np.array([100, 77, 200], dtype=np.int64),   # 77 = decoy, not a real member
            "POS ": np.random.uniform(-1, 1, (3, 3)).astype(np.float32),
            "MASS": np.random.uniform(1e6, 1e9, 3).astype(np.float32),
        }

        written = {}

        def fake_read_new(path, fields, ptype, **kwargs):
            return {4: snap_data}

        def capture_write(table, root_path, **kwargs):
            written["table"] = table

        def fake_yield_all_files(base):
            return ["group_f1"] if base == fetcher._group_base else ["snap.hdf5"]

        with tempfile.TemporaryDirectory() as tmpdir:
            OmegaConf.set_struct(fetcher.catalog, False)
            fetcher.catalog.path = tmpdir
            with patch("g3read.yield_all_files", side_effect=fake_yield_all_files), \
                 patch("g3read.GadgetFile", _FakeGadgetFile), \
                 patch("g3read.read_new", side_effect=fake_read_new), \
                 patch("pyarrow.parquet.write_to_dataset", side_effect=capture_write), \
                 patch("src.data_generation.src.data_fetcher.LazySubhaloParticles"), \
                 patch.object(fetcher, "_run_particle_pipeline", side_effect=lambda d: d):
                fetcher._store_particles()

        assert "table" in written, "write_to_dataset was never called"
        df = written["table"].to_pandas()

        # Only the two genuine members (100 -> halo 0, 60 -> halo 1) may be
        # written; the decoy (77) must have been dropped, not assigned to
        # halo 0 as V1's interval approximation would have done.
        assert len(df) == 2
        assert set(df["parent_id"].tolist()) == {0, 1}
