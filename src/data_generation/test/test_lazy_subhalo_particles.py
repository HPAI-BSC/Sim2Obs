import tempfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.lazy_subhalo import LazySubhaloParticles


# helpers

def make_mock_particles_dataset(path: str, n_halos: int = 5, particles_per_halo: int = 10):
    """
    Write a partitioned parquet dataset mimicking real subhalo particle data.
    Partition column: parent_bucket (parent_id % 128)
    parent_bucket_data is the in-file duplicate of the partition column.
    """
    records = []

    for halo_id in range(n_halos):
        bucket = halo_id % 128
        for _ in range(particles_per_halo):
            records.append({
                "parent_id":          halo_id,
                "parent_bucket":      bucket,
                "parent_bucket_data": bucket,
                "group_number":       halo_id * 10,

                "star_p": np.random.uniform(-1.0,  1.0, 3).tolist(),
                "star_m": np.random.uniform(1e8,   1e12),
                "star_v": np.random.uniform(-500, 500,  3).tolist(),
            })

    df    = pd.DataFrame(records)
    table = pa.Table.from_pandas(df, preserve_index=False)

    pq.write_to_dataset(
        table,
        root_path=path,
        partition_cols=["parent_bucket"],
        use_threads=True,
    )


# fixtures

@pytest.fixture(scope="module")
def mock_dataset_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        make_mock_particles_dataset(tmpdir, n_halos=5, particles_per_halo=10)
        yield tmpdir


@pytest.fixture()
def ctx(mock_dataset_path):
    """Fresh context per test — avoids filter/transform/flag state leaking between tests."""
    return LazySubhaloParticles(mock_dataset_path)


class _FakeHalo:
    def __init__(self, halo_id, gpos=None, group_number=None):
        self.halo_id = halo_id
        self.gpos = gpos if gpos is not None else [0.0, 0.0, 0.0]
        if group_number is not None:
            self.group_number = group_number


class TestLazySubhaloParticles:

    # add_filter / _build_filter

    def test_build_filter_returns_none_when_no_filters(self, ctx):
        assert ctx._build_filter() is None

    def test_build_filter_single_filter(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 1)
        assert ctx._build_filter() is not None

    def test_build_filter_combines_multiple_filters(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 1)
        ctx.add_filter(ds.field("group_number") == 10)
        result = ctx.dataset.to_table(filter=ctx._build_filter()).to_pandas()
        assert (result["parent_id"] == 1).all()
        assert (result["group_number"] == 10).all()

    # materialize

    def test_materialize_returns_full_dataset_without_filters(self, ctx):
        assert ctx.materialize().num_rows == 50  # 5 halos × 10 particles

    def test_materialize_applies_registered_filters(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 0)
        table = ctx.materialize()
        assert table.num_rows == 10
        assert all(v == 0 for v in table["parent_id"].to_pylist())

    def test_materialize_sets_self_data(self, ctx):
        assert ctx.data is None
        ctx.materialize()
        assert ctx.data is not None

    def test_materialize_applies_transforms_in_order(self, ctx):
        # First transform sets star_p to all [1,1,1], second to [2,2,2].
        # Final result must be [2,2,2], proving order is preserved.
        ctx.add_transform(lambda t: t.set_column(
            t.schema.get_field_index("star_p"), "star_p",
            pa.array([[1.0, 1.0, 1.0]] * t.num_rows)
        ))
        ctx.add_transform(lambda t: t.set_column(
            t.schema.get_field_index("star_p"), "star_p",
            pa.array([[2.0, 2.0, 2.0]] * t.num_rows)
        ))
        table = ctx.materialize()
        assert table["star_p"].to_pylist()[0] == [2.0, 2.0, 2.0]

    # __add__

    def test_add_merges_two_contexts(self, mock_dataset_path):
        merged = LazySubhaloParticles(mock_dataset_path) + LazySubhaloParticles(mock_dataset_path)
        assert merged.materialize().num_rows == 100

    def test_add_returns_same_class(self, mock_dataset_path):
        merged = LazySubhaloParticles(mock_dataset_path) + LazySubhaloParticles(mock_dataset_path)
        assert isinstance(merged, LazySubhaloParticles)

    def test_add_sets_overlapping_particles_id(self, mock_dataset_path):
        ctx_a = LazySubhaloParticles(mock_dataset_path)
        ctx_b = LazySubhaloParticles(mock_dataset_path)
        merged = ctx_a + ctx_b
        # Boundary should be the row count of ctx_a alone
        assert merged.overlapping_particles_id == 50

    def test_add_incompatible_type_returns_not_implemented(self, ctx):
        assert ctx.__add__("not-a-context") is NotImplemented

    # non_overlapping

    def test_non_overlapping_returns_self_when_no_merge(self, ctx):
        # No __add__ called — should log and return self
        result = ctx.non_overlapping
        assert result is ctx

    def test_non_overlapping_slices_to_boundary(self, mock_dataset_path):
        ctx_a = LazySubhaloParticles(mock_dataset_path)
        ctx_b = LazySubhaloParticles(mock_dataset_path)
        merged = ctx_a + ctx_b
        result = merged.non_overlapping.materialize()
        # Should contain only the first 50 rows (ctx_a's rows)
        assert result.num_rows == 50

    def test_non_overlapping_does_not_mutate_merged(self, mock_dataset_path):
        ctx_a = LazySubhaloParticles(mock_dataset_path)
        ctx_b = LazySubhaloParticles(mock_dataset_path)
        merged = ctx_a + ctx_b
        _ = merged.non_overlapping
        # Original merged context still returns all 100 rows
        assert merged.materialize().num_rows == 100

    # preprocess

    def test_preprocess_filters_to_single_halo(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=2, gpos=[0.0, 0.0, 0.0]))
        unique = set(ctx.materialize()["parent_id"].to_pylist())
        assert unique == {2}

    def test_preprocess_filters_by_group_number_when_present(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=2, gpos=[0.0, 0.0, 0.0], group_number=20))
        table = ctx.materialize()
        assert all(v == 20 for v in table["group_number"].to_pylist())

    def test_preprocess_subtracts_gpos(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=0, gpos=[0.5, 0.5, 0.5]))
        star_p = np.array(ctx.materialize()["star_p"].to_pylist())
        # uniform[-1,1] - 0.5 => values in [-1.5, 0.5]
        assert star_p.min() >= -1.5 - 1e-9
        assert star_p.max() <=  0.5 + 1e-9

    def test_preprocess_lbox_wraps_positions(self, ctx):
        lbox = 2.0
        ctx.preprocess(_FakeHalo(halo_id=0, gpos=[0.9, 0.9, 0.9]), lbox=lbox)
        star_p = np.array(ctx.materialize()["star_p"].to_pylist())
        assert (star_p > -lbox / 2 - 1e-9).all()
        assert (star_p <= lbox / 2 + 1e-9).all()

    def test_preprocess_snapshot_isolates_halo_mutation(self, ctx):
        halo = _FakeHalo(halo_id=0, gpos=[0.0, 0.0, 0.0])
        ctx.preprocess(halo)
        halo.gpos = [999.0, 999.0, 999.0]  # mutate after registration
        star_p = np.array(ctx.materialize()["star_p"].to_pylist())
        # Snapshot taken at registration — positions must not be shifted by 999
        assert star_p.max() < 10.0

    def test_preprocess_raises_if_called_twice(self, ctx):
        ctx.preprocess(_FakeHalo(halo_id=0))
        with pytest.raises(RuntimeError, match="preprocess"):
            ctx.preprocess(_FakeHalo(halo_id=0))

    # rotate

    def test_rotate_transforms_star_p(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 0)
        ctx.rotate(z=np.array([0.0, 0.0, 1.0]), degree=90)
        star_p = np.array(ctx.materialize()["star_p"].to_pylist())
        assert star_p.shape == (10, 3)
        assert np.isfinite(star_p).all()

    def test_rotate_raises_if_called_twice(self, ctx):
        ctx.rotate(z=np.array([0.0, 0.0, 1.0]), degree=45)
        with pytest.raises(RuntimeError, match="rotate"):
            ctx.rotate(z=np.array([0.0, 0.0, 1.0]), degree=45)

    def test_rotate_snapshot_isolates_mutation(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 0)
        ctx.rotate(z=np.array([0.0, 0.0, 1.0]), degree=90)
        ctx.materialize()  # must not raise or use mutated values

    def test_rotate_applied_after_preprocess(self, ctx):
        # preprocess then rotate — both transforms must compose correctly
        ctx.preprocess(_FakeHalo(halo_id=0, gpos=[0.0, 0.0, 0.0]))
        ctx.rotate(z=np.array([0.0, 0.0, 1.0]), degree=90)
        star_p = np.array(ctx.materialize()["star_p"].to_pylist())
        assert star_p.shape == (10, 3)
        assert np.isfinite(star_p).all()

    # is_empty

    def test_is_empty_before_materialize(self, ctx):
        assert ctx.is_empty is True

    def test_is_empty_after_materialize_with_data(self, ctx):
        ctx.materialize()
        assert ctx.is_empty is False

    def test_is_empty_after_materialize_with_no_matching_rows(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 9999)
        ctx.materialize()
        assert ctx.is_empty is True

    # __len__

    def test_len_raises_before_materialize(self, ctx):
        with pytest.raises(RuntimeError, match="materialize"):
            len(ctx)

    def test_len_after_materialize(self, ctx):
        ctx.materialize()
        assert len(ctx) == 50

    def test_len_after_filtered_materialize(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 1)
        ctx.materialize()
        assert len(ctx) == 10

    #  __repr__

    def test_repr_before_materialize_mentions_unmaterialized(self, ctx):
        r = repr(ctx)
        assert "unmaterialized" in r

    def test_repr_before_materialize_shows_filter_count(self, ctx):
        ctx.add_filter(ds.field("parent_id") == 0)
        ctx.add_filter(ds.field("group_number") == 0)
        r = repr(ctx)
        assert "2" in r  # 2 filters registered

    def test_repr_after_materialize_shows_columns(self, ctx):
        ctx.materialize()
        r = repr(ctx)
        assert "star_p" in r
        assert "star_m" in r