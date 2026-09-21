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
    so tests can assert V2 reads each file once per streaming scan instead
    of once per halo/subhalo (V1's bug) or building one giant global index
    (the memory-unsafe intermediate version this class replaced)."""
    read_calls = []

    def __init__(self, path, is_snap=False):
        self.path = path

    def read(self, block, *args, **kwargs):
        if block == "PID ":
            _FakeGadgetFile.read_calls.append(self.path)
            return _FakeGadgetFile.pid_by_file.get(self.path, np.array([], dtype=np.int64))
        return None


# ── tests ──────────────────────────────────────────────────────────────────────

class TestUnwrapInt32Overflow:
    """
    GOFF/SOFF are stored on disk as signed 4-byte integers — confirmed via
    the group catalogue's own INFO-table type declaration and by directly
    measuring the on-disk block byte length, on real production data. For
    a large box the true cumulative offset for later halos/subhalos
    exceeds the 4-byte signed limit (2**31 - 1) and g3read reads back a
    negative number instead. Since the whole catalogue's true offsets stay
    under 2**32, each value wraps at most once, so it's always recoverable
    independently by adding 2**32 back whenever the value is negative.
    """

    def test_positive_values_unchanged(self):
        raw = np.array([0, 100, 2_147_483_647], dtype=np.int64)
        result = MagneticumFetcherV2._unwrap_int32_overflow(raw)
        np.testing.assert_array_equal(result, raw)

    def test_negative_values_get_2_32_added_back(self):
        wrapped = np.array([-2_147_480_517, -1], dtype=np.int64)
        result = MagneticumFetcherV2._unwrap_int32_overflow(wrapped)
        np.testing.assert_array_equal(result, wrapped + (1 << 32))

    def test_matches_real_production_wraparound(self):
        """
        Exact values observed on the real cluster (Magneticum snapnum 120):
        halo row 139521's true GOFF is 2147486779 (= halo 139520's
        GOFF+GLEN, 2147483622+3157), but reads back as -2147480517 because
        it exceeds the 4-byte signed limit.
        """
        raw = np.array([-2_147_480_517], dtype=np.int64)
        result = MagneticumFetcherV2._unwrap_int32_overflow(raw)
        assert result[0] == 2_147_486_779


class TestMagneticumFetcherV2:

    # ── no-op builders ─────────────────────────────────────────────────────
    # V2 builds nothing upfront (no global index — see class docstring for
    # why: an in-RAM global index scales with total group-catalogue size
    # and OOM'd on a real production run with a hard memory cap). Ownership
    # is streamed and matched per call to _assign_halo_ids/_assign_subhalo_ids
    # instead, so all four of V1's builder methods are no-ops here.

    def test_build_pid_file_boundaries_is_noop(self, fetcher):
        assert fetcher._build_pid_file_boundaries() == []

    def test_build_subhalo_pid_file_boundaries_is_noop(self, fetcher):
        assert fetcher._build_subhalo_pid_file_boundaries() == []

    def test_build_halo_intervals_is_noop(self, fetcher):
        assert fetcher._build_halo_intervals() is None
        assert not hasattr(fetcher, "_halo_intervals")

    def test_build_subhalo_index_is_noop(self, fetcher):
        assert fetcher._build_subhalo_index() is None
        assert not hasattr(fetcher, "_subhalo_index")

    def test_build_subhalo_pid_arrays_is_noop(self, fetcher):
        assert fetcher._build_subhalo_pid_arrays() is None
        assert not hasattr(fetcher, "_subhalo_sorted_pids")

    # ── _load_group_offsets ────────────────────────────────────────────────

    def test_load_group_offsets_builds_expected_arrays(self, fetcher):
        fetcher._load_group_offsets()
        np.testing.assert_array_equal(fetcher._halo_goff, [0, 10, 20, 30, 40])
        np.testing.assert_array_equal(fetcher._halo_gend, [10, 20, 30, 40, 50])
        np.testing.assert_array_equal(fetcher._halo_id_sorted, [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(fetcher._sub_soff_sorted, [0, 5, 10, 15, 20])
        np.testing.assert_array_equal(fetcher._sub_send_sorted, [5, 10, 15, 20, 25])
        np.testing.assert_array_equal(fetcher._sub_id_sorted, [0, 1, 2, 3, 4])

    def test_load_group_offsets_is_idempotent(self, fetcher):
        fetcher._load_group_offsets()
        first = fetcher._halo_goff
        fetcher._goff = np.array([999])  # would change the result if rebuilt
        fetcher._load_group_offsets()
        assert fetcher._halo_goff is first

    def test_load_group_offsets_sorts_out_of_order_goff(self, fetcher):
        """
        Regression: multi-file group catalogues concatenate halos in
        on-disk/file order, which is NOT guaranteed to already be ascending
        GOFF (e.g. a zoom/multi-resolution box where per-file halo
        numbering doesn't line up globally). np.searchsorted requires its
        input sorted; feeding it unsorted GOFF/GLEN silently returns wrong
        (often empty) matches instead of raising — production symptom was
        "halo ownership for chunk [...] covers 0 positions, expected N".
        Halo 1 (GOFF=30) is on-disk BEFORE halo 0 (GOFF=0) here.
        """
        fetcher._goff = np.array([30, 0])
        fetcher._glen = np.array([10, 10])
        fetcher._soff = np.array([])
        fetcher._slen = np.array([])
        fetcher._load_group_offsets()

        np.testing.assert_array_equal(fetcher._halo_goff, [0, 30])
        np.testing.assert_array_equal(fetcher._halo_gend, [10, 40])
        # halo_id 1 (on-disk row 0, GOFF=0) now sorts before halo_id 0
        # (on-disk row 1, GOFF=30) — _halo_id_sorted must track that.
        np.testing.assert_array_equal(fetcher._halo_id_sorted, [1, 0])

        halo_owner, _ = fetcher._owners_for_chunk(0, 10)
        np.testing.assert_array_equal(halo_owner, [1] * 10)

    def test_load_group_offsets_unwraps_int32_overflow_goff(self, fetcher):
        """
        Regression for the real production bug: for a box with more than
        2**31 - 1 particles across all halos, GOFF for halos past that
        point is stored wrapped (negative) on disk. Sorting/comparing the
        raw wrapped value instead of the true one is exactly what caused
        "GOFF/GLEN contiguous-layout assumption no longer holds" to fire —
        the three earlier fix attempts (sorting by the raw, still-wrapped
        GOFF) never addressed this. Values below are the exact ones found
        on the real cluster (snapnum 120): halo 139520 sits just under the
        2**31 boundary, halo 139521 (GOFF+GLEN of the previous halo) sits
        just over it and wraps.
        """
        true_goff_1 = 2_147_486_779  # halo 139521's true GOFF; exceeds 2**31 - 1
        wrapped_goff_1 = true_goff_1 - (1 << 32)  # what g3read actually returns
        assert wrapped_goff_1 < 0

        fetcher._goff = np.array([2_147_483_622, wrapped_goff_1])
        fetcher._glen = np.array([3157, 3157])
        fetcher._soff = np.array([])
        fetcher._slen = np.array([])
        fetcher._load_group_offsets()

        np.testing.assert_array_equal(fetcher._halo_goff, [2_147_483_622, true_goff_1])
        np.testing.assert_array_equal(fetcher._halo_gend, [2_147_486_779, true_goff_1 + 3157])
        np.testing.assert_array_equal(fetcher._halo_id_sorted, [0, 1])

        halo_owner, _ = fetcher._owners_for_chunk(2_147_483_622, 2_147_486_779 + 3157)
        np.testing.assert_array_equal(halo_owner, [0] * 3157 + [1] * 3157)

    def test_load_group_offsets_unwraps_soff_independently_of_parent_halo(self, fetcher):
        """
        Regression for a bug in an EARLIER attempt at this fix (never
        shipped): computing a subhalo's true offset as
        (raw SOFF - raw GOFF of its parent halo) assumes both wrap
        together. That's false whenever the 2**31 boundary falls in the
        MIDDLE of a halo's particle range — confirmed on real production
        data: halo 139520's own GOFF (2147483622) is still safely under
        2**31 and reads correctly, but a subhalo sitting further into that
        SAME halo's range (true SOFF 2147486277) has already crossed
        2**31 and wraps on its own, independently of its parent. The
        current fix (unwrap SOFF directly, with no dependency on the
        parent's GOFF) must get this right.
        """
        parent_true_goff = 2_147_483_622  # NOT wrapped
        parent_glen = 3157
        sub_true_soff = 2_147_486_277     # inside the parent's range, but wrapped
        sub_slen = 346
        wrapped_sub_soff = sub_true_soff - (1 << 32)
        assert wrapped_sub_soff < 0

        fetcher._goff = np.array([parent_true_goff])
        fetcher._glen = np.array([parent_glen])
        fetcher._soff = np.array([wrapped_sub_soff])
        fetcher._slen = np.array([sub_slen])
        fetcher._load_group_offsets()

        np.testing.assert_array_equal(fetcher._halo_goff, [parent_true_goff])
        np.testing.assert_array_equal(fetcher._sub_soff_sorted, [sub_true_soff])
        # the unwrapped subhalo must nest correctly inside its parent's range
        assert fetcher._sub_soff_sorted[0] >= fetcher._halo_goff[0]
        assert fetcher._sub_soff_sorted[0] + sub_slen <= fetcher._halo_goff[0] + parent_glen

    # ── _owners_for_chunk ───────────────────────────────────────────────────

    def test_owners_for_chunk_halo_exact_partition(self, fetcher):
        fetcher._goff = np.array([0, 5])
        fetcher._glen = np.array([5, 5])
        fetcher._soff = np.array([])
        fetcher._slen = np.array([])
        fetcher._load_group_offsets()

        halo_owner, sub_owner = fetcher._owners_for_chunk(0, 10)
        np.testing.assert_array_equal(halo_owner, [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(sub_owner, [-1] * 10)

    def test_owners_for_chunk_subhalo_gap_yields_minus_one(self, fetcher):
        fetcher._goff = np.array([0])
        fetcher._glen = np.array([5])
        fetcher._soff = np.array([0])   # subhalo 0 only covers positions 0-2
        fetcher._slen = np.array([3])   # positions 3-4 belong to no subhalo
        fetcher._load_group_offsets()

        _, sub_owner = fetcher._owners_for_chunk(0, 5)
        np.testing.assert_array_equal(sub_owner, [0, 0, 0, -1, -1])

    def test_owners_for_chunk_partial_boundary_halo(self, fetcher):
        """
        Group-file chunks don't align to halo boundaries — a chunk can start
        or end in the middle of a halo's range. Positions [3, 8) span the
        tail of halo 0 (positions 0-4) and the head of halo 1 (positions
        5-9); ownership for this chunk must be clipped correctly at both
        ends, not just for chunks that happen to align with GOFF/GLEN.
        """
        fetcher._goff = np.array([0, 5])
        fetcher._glen = np.array([5, 5])
        fetcher._soff = np.array([])
        fetcher._slen = np.array([])
        fetcher._load_group_offsets()

        halo_owner, _ = fetcher._owners_for_chunk(3, 8)
        np.testing.assert_array_equal(halo_owner, [0, 0, 1, 1, 1])

    # ── _stream_match ───────────────────────────────────────────────────────

    def test_stream_match_reads_each_group_file_exactly_once(self, fetcher):
        fetcher._goff = np.array([0, 25])
        fetcher._glen = np.array([25, 25])
        fetcher._soff = np.array([0, 25])
        fetcher._slen = np.array([25, 25])
        fetcher._load_group_offsets()

        pid_by_file = {
            "f1": np.arange(0, 25, dtype=np.int64),
            "f2": np.arange(25, 50, dtype=np.int64),
        }
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = pid_by_file

        ids = np.array([10, 30], dtype=np.int64)
        with patch("g3read.yield_all_files", return_value=["f1", "f2"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            halo_ids, subhalo_ids = fetcher._stream_match(ids)

        assert _FakeGadgetFile.read_calls == ["f1", "f2"]
        np.testing.assert_array_equal(halo_ids, [0, 1])
        np.testing.assert_array_equal(subhalo_ids, [0, 1])

    def test_stream_match_unmatched_ids_get_minus_one(self, fetcher):
        fetcher._goff = np.array([0])
        fetcher._glen = np.array([10])
        fetcher._soff = np.array([0])
        fetcher._slen = np.array([10])
        fetcher._load_group_offsets()

        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": np.arange(10, dtype=np.int64)}

        ids = np.array([5, 999_999], dtype=np.int64)  # second id never a real particle
        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            halo_ids, subhalo_ids = fetcher._stream_match(ids)

        np.testing.assert_array_equal(halo_ids, [0, -1])
        np.testing.assert_array_equal(subhalo_ids, [0, -1])

    # ── _assign_halo_ids / _assign_subhalo_ids ─────────────────────────────

    def test_assign_halo_ids_empty_input_returns_empty(self, fetcher):
        result = fetcher._assign_halo_ids(np.array([], dtype=np.int64))
        assert len(result) == 0

    def test_assign_subhalo_ids_empty_input_returns_empty(self, fetcher):
        result = fetcher._assign_subhalo_ids(np.array([], dtype=np.int64))
        assert len(result) == 0

    def test_assign_halo_ids_and_subhalo_ids_share_one_scan(self, fetcher):
        """
        Regression for a real bug: the cache guard originally compared
        id(ids) with `is not` instead of `!=`. Large ints aren't interned,
        so two separately-computed id() calls with the same VALUE are
        different int OBJECTS — `is not` was always True, silently
        defeating the cache and re-scanning the group catalogue on every
        _assign_subhalo_ids call. _store_particles() calls
        _assign_halo_ids(ids) then _assign_subhalo_ids(ids) with the exact
        same array once per snapshot file — this must cost one scan, not two.
        """
        fetcher._goff = np.array([0])
        fetcher._glen = np.array([10])
        fetcher._soff = np.array([0])
        fetcher._slen = np.array([10])
        fetcher._load_group_offsets()

        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": np.arange(10, dtype=np.int64)}

        ids = np.array([3, 7], dtype=np.int64)
        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            fetcher._assign_halo_ids(ids)
            fetcher._assign_subhalo_ids(ids)

        assert _FakeGadgetFile.read_calls == ["f1"], (
            f"expected exactly one scan for two calls sharing the same ids "
            f"array, got {len(_FakeGadgetFile.read_calls)}"
        )

    def test_assign_halo_ids_rescans_for_different_ids(self, fetcher):
        """The cache must not leak results across different id batches —
        each new snapshot-file chunk needs its own scan."""
        fetcher._goff = np.array([0])
        fetcher._glen = np.array([10])
        fetcher._soff = np.array([0])
        fetcher._slen = np.array([10])
        fetcher._load_group_offsets()

        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": np.arange(10, dtype=np.int64)}

        ids_a = np.array([1, 2], dtype=np.int64)
        ids_b = np.array([3, 4], dtype=np.int64)
        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
            fetcher._assign_halo_ids(ids_a)
            fetcher._assign_halo_ids(ids_b)

        assert _FakeGadgetFile.read_calls == ["f1", "f1"], (
            "a different ids array must trigger a fresh scan, not reuse "
            "the previous batch's cached result"
        )

    def test_assign_halo_ids_rejects_false_positive_that_v1_interval_would_accept(self, fetcher):
        """
        The actual bug, reproduced directly: halo 0's true pids are
        {5, 50, 100} -> V1's interval for halo 0 is [5, 100]. Particle 77
        was never a member of ANY halo (not present in the group catalogue
        at all), but V1's range test still matches it to halo 0 purely
        because 77 falls inside [5, 100].

        V1's own _assign_halo_ids (unchanged, exercised here directly)
        confirms the false positive; V2's exact-match streaming lookup on
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

        # V2 behaviour on equivalent data via the streaming path.
        fetcher._goff = np.array([0, 3])
        fetcher._glen = np.array([3, 2])
        fetcher._soff = np.array([0, 3])
        fetcher._slen = np.array([3, 2])
        fetcher._load_group_offsets()
        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"f1": pid_values}

        with patch("g3read.yield_all_files", return_value=["f1"]), \
             patch("g3read.GadgetFile", _FakeGadgetFile):
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
        # _cache_loaded and _particle_cache_loaded are independent flags —
        # set both so _store_particles() doesn't rebuild the pre-seeded
        # _snap_cache_config above via the real _load_particle_cache().
        fetcher._particle_cache_loaded = True

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

        # Only the two genuine members (100 -> halo 0, 200 -> halo 1) may be
        # written; the decoy (77) must have been dropped, not assigned to
        # halo 0 as V1's interval approximation would have done.
        assert len(df) == 2
        assert set(df["parent_id"].tolist()) == {0, 1}

        # _store_particles() calls _assign_halo_ids then _assign_subhalo_ids
        # with the same ids array for this one snapshot file — must be one
        # scan of the group catalogue, not two.
        assert _FakeGadgetFile.read_calls == ["group_f1"], (
            f"expected one group-catalogue scan for this snapshot file, "
            f"got {len(_FakeGadgetFile.read_calls)}"
        )


class TestVerifyAssignmentIntegrity:
    """
    verify_assignment_integrity runs the real _assign_halo_ids/
    _assign_subhalo_ids path end-to-end and checks its own output for two
    kinds of provable errors, without needing external ground truth:
    (1) a subhalo's recorded parent (GRNR) must match the halo_id assigned
    to the same particle, and (2) no halo/subhalo can be assigned more
    particles than its own GLEN/SLEN.
    """

    @staticmethod
    def _run(fetcher, grnr, group_pid_values, snap_ids):
        def fake_read_new(path, fields, ptype, **kwargs):
            if fields == ["GRNR"]:
                return {"GRNR": grnr}
            # Regression: this method only ever uses "ID  " — it must not
            # request POS/VEL/MASS (unnecessary I/O, and needlessly exposes
            # it to any field-naming issue in fields it doesn't use).
            assert fields == ["ID  "], f"expected snapshot read to request only ID, got {fields}"
            return {ptype: {"ID  ": snap_ids}}

        def fake_yield_all_files(base):
            return ["group_f1"] if base == fetcher._group_base else ["snap_f1"]

        _FakeGadgetFile.read_calls = []
        _FakeGadgetFile.pid_by_file = {"group_f1": group_pid_values}
        fetcher._snap_cache_config = {
            "stars": {"fields_padded": ["ID  ", "POS ", "MASS"], "ptype_int": 4}
        }
        fetcher._particle_cache_loaded = True

        with patch("g3read.yield_all_files", side_effect=fake_yield_all_files), \
             patch("g3read.GadgetFile", _FakeGadgetFile), \
             patch("g3read.read_new", side_effect=fake_read_new):
            return fetcher.verify_assignment_integrity(max_snap_files=1)

    def test_passes_on_consistent_assignment(self, fetcher):
        # halo0 = positions [0,3), halo1 = positions [3,5)
        fetcher._goff = np.array([0, 3])
        fetcher._glen = np.array([3, 2])
        # subhalo0 covers halo0's full range, subhalo1 covers halo1's —
        # consistent parentage
        fetcher._soff = np.array([0, 3])
        fetcher._slen = np.array([3, 2])

        group_pid_values = np.array([100, 5, 50, 200, 250], dtype=np.int64)
        snap_ids = np.array([100, 50, 200], dtype=np.int64)  # all 3 are real members
        grnr = np.array([0, 1], dtype=np.int64)  # subhalo0->halo0, subhalo1->halo1

        result = self._run(fetcher, grnr, group_pid_values, snap_ids)

        assert result["ok"] is True
        assert result["n_particles_checked"] == 3
        assert result["n_consistency_failures"] == 0
        assert result["n_halo_bound_failures"] == 0
        assert result["n_subhalo_bound_failures"] == 0

    def test_detects_grnr_mismatch(self, fetcher):
        """
        Regression target: if _assign_halo_ids and _assign_subhalo_ids ever
        disagreed about a particle's parentage (e.g. from a bug in the
        offset arithmetic), a subhalo's real parent (GRNR) would no longer
        match the halo_id assigned to the same particle. Simulated here by
        deliberately swapping GRNR so it's wrong for both subhalos — the
        method must catch and count every such mismatch, not just some.
        """
        fetcher._goff = np.array([0, 3])
        fetcher._glen = np.array([3, 2])
        fetcher._soff = np.array([0, 3])
        fetcher._slen = np.array([3, 2])

        group_pid_values = np.array([100, 5, 50, 200, 250], dtype=np.int64)
        snap_ids = np.array([100, 50, 200], dtype=np.int64)
        grnr = np.array([1, 0], dtype=np.int64)  # swapped: WRONG parents

        result = self._run(fetcher, grnr, group_pid_values, snap_ids)

        assert result["ok"] is False
        # 2 particles (100, 50) matched to (halo0, subhalo0): grnr[0]=1 != 0
        # 1 particle (200) matched to (halo1, subhalo1): grnr[1]=0 != 1
        assert result["n_consistency_failures"] == 3
        assert result["n_halo_bound_failures"] == 0
        assert result["n_subhalo_bound_failures"] == 0

    def test_no_star_particle_type_configured_returns_empty_ok_result(self, fetcher):
        fetcher.particle_fields = ["gas"]  # no "star"/"stars" entry
        result = fetcher.verify_assignment_integrity()
        assert result == {
            "n_particles_checked": 0, "n_consistency_failures": 0,
            "n_halo_bound_failures": 0, "n_subhalo_bound_failures": 0, "ok": True,
        }
