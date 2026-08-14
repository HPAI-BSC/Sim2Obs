import numpy as np
import pytest
from omegaconf import OmegaConf

from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.subhalo import SubhaloParticles


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_mock_particle_data(n_particles=10):
    """Create simple deterministic particle data for testing."""
    return {
        "star_p": np.array([
            [float(i), float(i + 1), float(i + 2)]
            for i in range(n_particles)
        ]),
        "star_m": np.arange(1, n_particles + 1, dtype=float),
        "star_v": np.array([
            [float(i), float(i + 1), float(i + 2)]
            for i in range(n_particles)
        ]),
    }


class _FakeHalo:
    def __init__(self, halo_id=0, gpos=None):
        self.halo_id = halo_id
        self.gpos = np.array(
            gpos if gpos is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def particle_data():
    return make_mock_particle_data(n_particles=10)


@pytest.fixture()
def ctx(particle_data):
    """Fresh SubhaloParticles object for every test."""
    return SubhaloParticles(
        data=particle_data,
        column_map=None,
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestSubhaloParticles:

    # __init__

    def test_init_creates_particle_attributes(self, ctx):
        assert hasattr(ctx, "star_p")
        assert hasattr(ctx, "star_m")
        assert hasattr(ctx, "star_v")

    def test_init_converts_lists_to_numpy_arrays(self):
        data = {
            "star_p": [[0, 1, 2], [3, 4, 5]],
            "star_m": [1.0, 2.0],
            "star_v": [[1, 0, 0], [0, 1, 0]],
        }

        ctx = SubhaloParticles(data=data, column_map=None)

        assert isinstance(ctx.star_p, np.ndarray)
        assert isinstance(ctx.star_m, np.ndarray)
        assert isinstance(ctx.star_v, np.ndarray)

    def test_init_preserves_numpy_arrays(self, particle_data):
        ctx = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )

        assert ctx.star_p.shape == (10, 3)
        assert ctx.star_m.shape == (10,)
        assert ctx.star_v.shape == (10, 3)

    def test_init_sets_overlapping_particles_id_to_none(self, ctx):
        assert ctx.overlapping_particles_id is None

    def test_init_supports_pandas_series(self):
        import pandas as pd

        data = {
            "star_p": pd.Series([
                [0, 1, 2],
                [3, 4, 5],
            ]),
            "star_m": pd.Series([1.0, 2.0]),
            "star_v": pd.Series([
                [1, 0, 0],
                [0, 1, 0],
            ]),
        }

        ctx = SubhaloParticles(
            data=data,
            column_map=None,
        )

        assert isinstance(ctx.star_p, np.ndarray)
        assert isinstance(ctx.star_m, np.ndarray)
        assert isinstance(ctx.star_v, np.ndarray)

    # non_overlapping
    #
    # `overlapping_particles_id` is a plain int boundary index (set by
    # __add__): particles [0:boundary] belong to the original subhalo,
    # everything after is the overlap that was appended.

    def test_non_overlapping_returns_self_when_no_overlap(self, ctx):
        result = ctx.non_overlapping

        assert result is ctx

    def test_non_overlapping_returns_self_when_boundary_is_zero(self, ctx):
        ctx.overlapping_particles_id = 0

        result = ctx.non_overlapping

        assert result is ctx

    def test_non_overlapping_keeps_first_particles(self, ctx):
        overlapping_particles = SubhaloParticles(
            data=make_mock_particle_data(n_particles=5),
            column_map=None,
        )
        merged = ctx + overlapping_particles
        result = merged.non_overlapping

        assert result.star_p.shape == (10, 3)
        assert result.star_m.shape == (10,)
        assert result.star_v.shape == (10, 3)
        np.testing.assert_array_equal(result.star_m, ctx.star_m)

    def test_non_overlapping_does_not_mutate_original(self, ctx):
        overlapping_particles = SubhaloParticles(
            data=make_mock_particle_data(n_particles=5),
            column_map=None,
        )
        merged = ctx + overlapping_particles

        _ = merged.non_overlapping

        assert merged.star_p.shape == (15, 3)
        assert merged.star_m.shape == (15,)

    # __add__

    def test_add_merges_particle_arrays(self, particle_data):
        ctx_a = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )
        ctx_b = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )

        merged = ctx_a + ctx_b

        assert merged.star_p.shape == (20, 3)
        assert merged.star_m.shape == (20,)
        assert merged.star_v.shape == (20, 3)

    def test_add_concatenates_values_in_order(self, particle_data):
        ctx_a = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )
        ctx_b = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )

        merged = ctx_a + ctx_b

        np.testing.assert_array_equal(
            merged.star_m,
            np.concatenate([
                particle_data["star_m"],
                particle_data["star_m"],
            ]),
        )

    def test_add_sets_overlapping_particles_boundary(self, particle_data):
        ctx_a = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )
        ctx_b = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )

        merged = ctx_a + ctx_b

        assert merged.overlapping_particles_id == 10

    def test_add_returns_same_class(self, particle_data):
        ctx_a = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )
        ctx_b = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )

        merged = ctx_a + ctx_b

        assert isinstance(merged, SubhaloParticles)

    def test_add_incompatible_type_returns_not_implemented(self, ctx):
        assert ctx.__add__("not-a-context") is NotImplemented

    def test_add_multiple_times(self, particle_data):
        ctx_a = SubhaloParticles(data=particle_data, column_map=None)
        ctx_b = SubhaloParticles(data=make_mock_particle_data(n_particles=5), column_map=None)
        ctx_c = SubhaloParticles(data=make_mock_particle_data(n_particles=3), column_map=None)

        merged = ctx_a + ctx_b + ctx_c

        # All particles are concatenated regardless of how many additions happen.
        assert merged.star_p.shape == (18, 3)
        assert merged.star_m.shape == (18,)
        assert merged.star_v.shape == (18, 3)

        # The boundary is only captured once, on the very first addition, so it
        # still reflects ctx_a's original length even after a third context is
        # folded in.
        assert merged.overlapping_particles_id == 10

        non_overlapping = merged.non_overlapping
        assert non_overlapping.star_p.shape == (10, 3)
        np.testing.assert_array_equal(non_overlapping.star_m, ctx_a.star_m)

    def test_add_order_affects_concatenation_order(self, particle_data):
        ctx_a = SubhaloParticles(data=particle_data, column_map=None)
        ctx_b = SubhaloParticles(data=make_mock_particle_data(n_particles=5), column_map=None)

        forward = ctx_a + ctx_b
        backward = ctx_b + ctx_a

        # Same total number of particles either way...
        assert forward.star_m.shape == backward.star_m.shape == (15,)

        # ...but the concatenation order follows `self + other`, so the first
        # block of values differs depending on which operand comes first.
        np.testing.assert_array_equal(forward.star_m[:10], ctx_a.star_m)
        np.testing.assert_array_equal(forward.star_m[10:], ctx_b.star_m)

        np.testing.assert_array_equal(backward.star_m[:5], ctx_b.star_m)
        np.testing.assert_array_equal(backward.star_m[5:], ctx_a.star_m)

        # The boundary is captured from whichever operand is `self`, so it
        # differs between the two orderings too.
        assert forward.overlapping_particles_id == 10
        assert backward.overlapping_particles_id == 5

    # preprocess

    def test_preprocess_subtracts_halo_position(self, ctx):
        halo = _FakeHalo(gpos=[1.0, 2.0, 3.0])

        ctx.preprocess(halo)

        expected = make_mock_particle_data(10)["star_p"] - halo.gpos

        np.testing.assert_array_equal(
            ctx.star_p,
            expected,
        )

    def test_preprocess_with_zero_center_keeps_positions(self, ctx):
        original = ctx.star_p.copy()

        halo = _FakeHalo(gpos=[0.0, 0.0, 0.0])

        ctx.preprocess(halo)

        np.testing.assert_array_equal(
            ctx.star_p,
            original,
        )

    def test_preprocess_applies_periodic_boundary_conditions(self):
        data = {
            "star_p": np.array([
                [9.0, 0.0, 0.0],
                [-9.0, 0.0, 0.0],
                [0.0, 6.0, 0.0],
            ]),
            "star_m": np.ones(3),
            "star_v": np.zeros((3, 3)),
        }

        ctx = SubhaloParticles(
            data=data,
            column_map=None,
        )

        halo = _FakeHalo(gpos=[0.0, 0.0, 0.0])

        ctx.preprocess(halo, lbox=10.0)

        expected = np.array([
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -4.0, 0.0],
        ])

        np.testing.assert_array_equal(
            ctx.star_p,
            expected,
        )

    def test_preprocess_does_not_change_mass(self, ctx):
        original_mass = ctx.star_m.copy()

        halo = _FakeHalo(gpos=[1.0, 2.0, 3.0])

        ctx.preprocess(halo)

        np.testing.assert_array_equal(
            ctx.star_m,
            original_mass,
        )

    # rotate

    def test_rotate_transforms_star_positions(self, ctx):
        ctx.rotate(
            z=np.array([0.0, 0.0, 1.0]),
            degree=90,
        )

        assert ctx.star_p.shape == (10, 3)
        assert np.isfinite(ctx.star_p).all()

    def test_rotate_preserves_number_of_particles(self, ctx):
        original_len = len(ctx)

        ctx.rotate(
            z=np.array([0.0, 0.0, 1.0]),
            degree=90,
        )

        assert len(ctx) == original_len

    # __repr__

    def test_repr_contains_numpy_array_shapes(self, ctx):
        result = repr(ctx)

        assert "star_p" in result
        assert "star_m" in result
        assert "star_v" in result

    def test_repr_contains_shapes(self, ctx):
        result = repr(ctx)

        assert "(10, 3)" in result
        assert "(10,)" in result

    # __len__

    def test_len_returns_number_of_particles(self, ctx):
        assert len(ctx) == 10

    def test_len_is_consistent_after_add(self, particle_data):
        ctx_a = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )
        ctx_b = SubhaloParticles(
            data=particle_data,
            column_map=None,
        )

        merged = ctx_a + ctx_b

        assert len(merged) == 20

    # is_empty

    def test_is_empty_is_false_when_data_exists(self, ctx):
        assert ctx.is_empty is False

    def test_is_empty_is_true_for_empty_context(self):
        ctx = SubhaloParticles(
            data={},
            column_map=None,
        )

        assert ctx.is_empty is True

    def test_init_with_column_map_depth_1(self):
        data = {
            "POS": np.array([
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
            ]),
            "MASS": np.array([1.0, 2.0]),
            "VEL": np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
        }

        column_map = {
            "POS": "p",
            "MASS": "m",
            "VEL": "v",
        }

        ctx = SubhaloParticles(
            data=data,
            column_map=column_map,
        )

        np.testing.assert_array_equal(
            ctx.star_p,
            data["POS"],
        )
        np.testing.assert_array_equal(
            ctx.star_m,
            data["MASS"],
        )
        np.testing.assert_array_equal(
            ctx.star_v,
            data["VEL"],
        )

    def test_init_with_column_map_depth_2(self):
        data = {
            'star': {
                "POS": np.array([
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                ]),
                "MASS": np.array([1.0, 2.0]),
                "VEL": np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]),
            },
            'gass': {}
        }

        column_map = {
            "star": { "POS": "p", "MASS": 'm', "VEL": "v"}
        }

        ctx = SubhaloParticles(
            data=data,
            column_map=column_map,
        )

        np.testing.assert_array_equal(
            ctx.star_p,
            data["star"]["POS"],
        )
        np.testing.assert_array_equal(
            ctx.star_m,
            data["star"]["MASS"],
        )
        np.testing.assert_array_equal(
            ctx.star_v,
            data["star"]["VEL"],
        )

    def test_init_with_column_map_depth_2_hydra(self):
        data = {
            'star': {
                "Coordinates": np.array([
                    [0.0, 1.0, 2.0],
                    [3.0, 4.0, 5.0],
                ]),
                "masses": np.array([1.0, 2.0]),
                "Velocities": np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]),
            },
            'gas': {}
        }

        column_map = OmegaConf.create({
            "star": {
                "Coordinates": {"name": "p"},
                "Velocities": {"name": "v"},
                "Masses": {"name": "m"},
            },
            "gas": {
                "Masses": {"name": "m"},
            },
        })

        ctx = SubhaloParticles(
            data=data,
            column_map=column_map,
        )

        np.testing.assert_array_equal(
            ctx.star_p,
            data["star"]["Coordinates"],
        )
        np.testing.assert_array_equal(
            ctx.star_m,
            data["star"]["Masses"],
        )
        np.testing.assert_array_equal(
            ctx.star_v,
            data["star"]["Velocities"],
        )

