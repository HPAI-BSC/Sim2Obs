import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
import tempfile

from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.lazy_subhalo import LazySubhaloContext


# helpers
def make_mock_subhalo_dataset(path: str, n_halos: int = 5, subhalos_per_halo: int = 10):
    records = []
    subhalo_counter = 0

    for halo_id in range(n_halos):
        for _ in range(subhalos_per_halo):
            bucket = halo_id % 128
            records.append({
                "subhalo_id": subhalo_counter,
                # FIX: real subhalo catalog schema field is "id_parents",
                # not "parent_id" (see MagneticumFetcher._get_subhalos --
                # df["parent_bucket"] = df["id_parents"] % 128). Mock data
                # now mirrors production schema.
                "id_parents": halo_id,
                "parent_bucket": bucket,   # partition column — removed from file schema
                "parent_bucket_data": bucket,   # regular column — stays inside the parquet file
                "group_number": halo_id * 10,

                "spos": np.random.uniform(-1.0,  1.0, 3).tolist()
            })
            subhalo_counter += 1

    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_to_dataset(
        table,
        root_path=path,
        partition_cols=["parent_bucket"],   # only partition on parent_bucket
        use_threads=True,
    )


# fixtures
@pytest.fixture(scope="module")
def mock_dataset_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        make_mock_subhalo_dataset(tmpdir, n_halos=5, subhalos_per_halo=10)
        yield tmpdir


@pytest.fixture()
def ctx(mock_dataset_path):
    """Fresh context per test — avoids filter/transform state leaking between tests."""
    return LazySubhaloContext(mock_dataset_path)


# Lightweight stub used wherever a HaloContext is needed
class _FakeHaloContext:
    def __init__(self, halo_ids):
        self.data = pd.DataFrame({"halo_id": halo_ids})


# Lightweight stub used wherever a Halo object is needed
class _FakeHalo:
    def __init__(self, halo_id, gpos=None, group_number=None):
        self.halo_id = halo_id
        self.gpos = gpos if gpos is not None else [0.0, 0.0, 0.0]
        if group_number is not None:
            self.group_number = group_number


class TestLazySubhaloContext:

    # get_items_by_halo
    # NOTE: the `id_parents` kwarg name below is the method's external
    # parameter name -- matches the actual COLUMN name too here (unlike
    # the confusing parent_id/id_parents split earlier versions of this
    # class had), so no naming mismatch to keep straight this time.

    def test_get_items_by_halo_returns_correct_parent(self, ctx):
        result = ctx.get_items_by_halo(id_parents=2)
        assert not result.empty
        assert (result["id_parents"] == 2).all()

    def test_get_items_by_halo_row_count(self, ctx):
        result = ctx.get_items_by_halo(id_parents=2)
        assert len(result) == 10

    def test_get_items_by_halo_has_expected_columns(self, ctx):
        result = ctx.get_items_by_halo(id_parents=0)
        # parent_bucket_data is the in-file column; parent_bucket is the partition dir (absent)
        assert {"subhalo_id", "id_parents", "parent_bucket_data", "spos"}.issubset(result.columns)

    def test_get_items_by_halo_unknown_id_returns_empty(self, ctx):
        result = ctx.get_items_by_halo(id_parents=9999)
        assert result.empty

    def test_get_items_by_halo_bucket_data_matches_formula(self, ctx):
        # parent_bucket_data is a regular column so it IS readable
        result = ctx.get_items_by_halo(id_parents=3)
        assert (result["parent_bucket_data"] == 3 % 128).all()

    # get_subhalo_by_id

    def test_get_subhalo_by_id_with_halo(self, ctx):
        result = ctx.get_subhalo_by_id(subhalo_id=0, halo_id=0)
        assert not result.empty
        assert result.iloc[0]["subhalo_id"] == 0
        assert result.iloc[0]["id_parents"] == 0

    def test_get_subhalo_by_id_without_halo(self, ctx):
        result = ctx.get_subhalo_by_id(subhalo_id=15)
        assert not result.empty
        assert result.iloc[0]["subhalo_id"] == 15

    def test_get_subhalo_by_id_wrong_halo_returns_empty(self, ctx):
        result = ctx.get_subhalo_by_id(subhalo_id=0, halo_id=3)
        assert result.empty

    def test_get_subhalo_by_id_respects_pipeline_filters(self, ctx):
        ctx.add_filter(ds.field("id_parents") != 0)
        result = ctx.get_subhalo_by_id(subhalo_id=0, halo_id=0)
        assert result.empty

    # ── add_filter / _build_filter ─────────────────────────────────────────────

    def test_build_filter_returns_none_when_no_filters(self, ctx):
        assert ctx._build_filter() is None

    def test_build_filter_single_filter(self, ctx):
        ctx.add_filter(ds.field("id_parents") == 1)
        assert ctx._build_filter() is not None

    def test_build_filter_combines_multiple_filters(self, ctx):
        ctx.add_filter(ds.field("id_parents") == 1)
        ctx.add_filter(ds.field("subhalo_id") > 5)
        result = ctx.dataset.to_table(filter=ctx._build_filter()).to_pandas()
        assert (result["id_parents"] == 1).all()
        assert (result["subhalo_id"] > 5).all()

    # ── materialize ────────────────────────────────────────────────────────────

    def test_materialize_returns_full_dataset_without_filters(self, ctx):
        assert ctx.materialize().num_rows == 50

    def test_materialize_applies_registered_filters(self, ctx):
        ctx.add_filter(ds.field("id_parents") == 0)
        table = ctx.materialize()
        assert table.num_rows == 10
        assert all(v == 0 for v in table["id_parents"].to_pylist())

    def test_materialize_sets_self_data(self, ctx):
        assert ctx.data is None
        ctx.materialize()
        assert ctx.data is not None

    def test_materialize_applies_transforms_in_order(self, ctx):
        ctx.add_transform(lambda t: t.set_column(
            t.schema.get_field_index("spos"), "spos",
            pa.array([[1.0, 1.0, 1.0]] * t.num_rows)
        ))
        ctx.add_transform(lambda t: t.set_column(
            t.schema.get_field_index("spos"), "spos",
            pa.array([[2.0, 2.0, 2.0]] * t.num_rows)
        ))
        table = ctx.materialize()
        assert table["spos"].to_pylist()[0] == [2.0, 2.0, 2.0]

    # ── remove_subhalos_belonging_unvalid_halos ─────────────────────────────────
    # Functionally identical to filter_to_valid_halos (same purpose, no
    # bucket pre-filter) -- mirrored test coverage for parity, since this
    # method previously had none at all.

    def test_remove_subhalos_belonging_unvalid_halos_removes_invalid_rows(self, ctx):
        ctx.remove_subhalos_belonging_unvalid_halos(_FakeHaloContext(halo_ids=[0, 1]))
        parent_ids = set(ctx.materialize()["id_parents"].to_pylist())
        assert parent_ids == {0, 1}

    def test_remove_subhalos_belonging_unvalid_halos_keeps_all_rows_when_all_valid(self, ctx):
        ctx.remove_subhalos_belonging_unvalid_halos(_FakeHaloContext(halo_ids=[0, 1, 2, 3, 4]))
        assert ctx.materialize().num_rows == 50

    def test_remove_subhalos_belonging_unvalid_halos_empty_halo_context_gives_empty(self, ctx):
        ctx.remove_subhalos_belonging_unvalid_halos(_FakeHaloContext(halo_ids=[]))
        assert ctx.materialize().num_rows == 0

    def test_remove_subhalos_belonging_unvalid_halos_composes_with_other_filters(self, ctx):
        # Registering this alongside another filter should AND them
        # together, same as any other add_filter() call.
        ctx.add_filter(ds.field("subhalo_id") >= 5)
        ctx.remove_subhalos_belonging_unvalid_halos(_FakeHaloContext(halo_ids=[0]))
        table = ctx.materialize()
        assert all(v == 0 for v in table["id_parents"].to_pylist())
        assert all(v >= 5 for v in table["subhalo_id"].to_pylist())

    # preprocess

    def test_preprocess_filters_to_single_halo(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=2, gpos=[0.0, 0.0, 0.0]))
        unique_parents = set(ctx.materialize()["id_parents"].to_pylist())
        assert unique_parents == {2}

    def test_preprocess_filters_by_group_number_when_present(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=2, gpos=[0.0, 0.0, 0.0], group_number=20))
        table = ctx.materialize()
        assert all(v == 20 for v in table["group_number"].to_pylist())

    def test_preprocess_subtracts_gpos(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=0, gpos=[0.5, 0.5, 0.5]))
        spos = np.array(ctx.materialize()["spos"].to_pylist())
        assert spos.min() >= -1.5 - 1e-9
        assert spos.max() <= 0.5 + 1e-9

    def test_preprocess_lbox_wraps_positions(self, ctx):
        lbox = 2.0
        ctx.preprocess(_FakeHalo(halo_id=0, gpos=[0.9, 0.9, 0.9]), lbox=lbox)
        spos = np.array(ctx.materialize()["spos"].to_pylist())
        assert (spos > -lbox / 2 - 1e-9).all()
        assert (spos <= lbox / 2 + 1e-9).all()

    def test_preprocess_snapshot_isolates_halo_mutation(self, ctx):
        halo = _FakeHalo(halo_id=0, gpos=[0.0, 0.0, 0.0])
        ctx.preprocess(halo)
        halo.gpos = [999.0, 999.0, 999.0]
        spos = np.array(ctx.materialize()["spos"].to_pylist())
        assert spos.max() < 10.0

    def test_preprocess_raises_if_called_twice(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=0))
        with pytest.raises(RuntimeError, match="preprocess"):
            ctx.preprocess(_FakeHalo(halo_id=0))

    # rotate

    def test_rotate_transforms_spos(self, ctx):
        ctx.add_filter(ds.field("id_parents") == 0)
        ctx.rotate(z=np.array([0, 0, 1]), degree=90)
        spos = np.array(ctx.materialize()["spos"].to_pylist())
        assert spos.shape == (10, 3)
        assert np.isfinite(spos).all()

    def test_rotate_raises_if_called_twice(self, ctx):
        ctx.rotate(z=np.array([0, 0, 1]), degree=45)
        with pytest.raises(RuntimeError, match="rotate"):
            ctx.rotate(z=np.array([0, 0, 1]), degree=45)

    def test_rotate_snapshot_isolates_mutation(self, ctx):
        ctx.add_filter(ds.field("id_parents") == 0)
        ctx.rotate(z=np.array([0.0, 0.0, 1.0]), degree=90)
        ctx.materialize()  # must not raise

    # __add__

    def test_add_merges_two_contexts(self, mock_dataset_path):
        merged = LazySubhaloContext(mock_dataset_path) + LazySubhaloContext(mock_dataset_path)
        assert merged.materialize().num_rows == 100

    def test_add_returns_same_class(self, mock_dataset_path):
        merged = LazySubhaloContext(mock_dataset_path) + LazySubhaloContext(mock_dataset_path)
        assert isinstance(merged, LazySubhaloContext)

    def test_add_incompatible_type_returns_not_implemented(self, ctx):
        assert ctx.__add__("not-a-context") is NotImplemented

    # __len__

    def test_len_raises_before_materialize(self, ctx):
        with pytest.raises(RuntimeError, match="materialize"):
            len(ctx)

    def test_len_after_materialize(self, ctx):
        ctx.materialize()
        assert len(ctx) == 50

    def test_len_after_filtered_materialize(self, ctx):
        ctx.add_filter(ds.field("id_parents") == 1)
        ctx.materialize()
        assert len(ctx) == 10

    # properties

    def test_id_parents_raises_before_materialize(self, ctx):
        with pytest.raises(RuntimeError, match="materialize"):
            _ = ctx.id_parents

    def test_id_parents_after_materialize(self, ctx):
        ctx.materialize()
        assert len(ctx.id_parents) == 50

    def test_subhalo_id_raises_before_materialize(self, ctx):
        with pytest.raises(RuntimeError, match="materialize"):
            _ = ctx.subhalo_id

    def test_subhalo_id_after_materialize(self, ctx):
        ctx.materialize()
        assert len(ctx.subhalo_id) == 50