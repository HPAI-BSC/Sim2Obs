"""
Regression tests for the Magneticum map-generation failure:

    MapStep - ERROR - Failed to create maps for halo 19980:
    setting an array element with a sequence.

Cause: Magneticum particles come back from parquet with POS/VEL as 1-D
object-dtype arrays (shape (N,), each element a length-3 array).
`VelocityDispersionProcessor.process` stored that array in `self.value`;
the ICL step later rebinds `subhalo_particles.star_v` to a stacked copy, so
`self.value` was left stale and `generate_map` ->
`np.asarray(self.value, dtype=np.float64)` raised.

Fix: `MagneticumFetcherV2._column_to_numpy` stacks those columns into (N, 3)
where the parquet is read (`_load_halo_particles` / `_load_subhalo_particles`),
so every downstream consumer sees real 2-D arrays from the start. Only the
Magneticum loaders are touched; other fetchers never go through this code.

Run:
    pytest src/data_generation/test/test_velocity_dispersion_object_star_v.py -v
"""
import numpy as np
import pandas as pd
import pytest

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.data_fetcher import MagneticumFetcherV2, MagneticumFetcherV2
from src.data_generation.src.feature import VelocityDispersionProcessor

HALO_ID = 19980
SUBHALO_ID = 7
N_PARTICLES = 500
BINS = 10
RANGING = ((-1.0, 1.0), (-1.0, 1.0))
AXIS = (0, 1)


def _write_particles_parquet(tmp_path, n=N_PARTICLES, seed=0, vec_dtype=np.float64):
    """Write a particles parquet shaped like MagneticumFetcherV2._store_particles output:
    POS/VEL as list-of-3 columns plus the parent/bucket bookkeeping columns."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 3)).astype(vec_dtype)
    vel = rng.normal(0, 300, (n, 3)).astype(vec_dtype)
    mass = rng.uniform(1e6, 1e7, n)

    path = tmp_path / "particles_parquet"
    pd.DataFrame({
        "POS": list(pos),
        "VEL": list(vel),
        "MASS": mass,
        "parent_id": np.full(n, HALO_ID),
        "subhalo_id": np.full(n, SUBHALO_ID),
        "parent_bucket": np.full(n, HALO_ID % 128),
    }).to_parquet(path)
    return str(path), pos, vel, mass


def _fetcher():
    """A Magneticum fetcher without running __init__ (no catalogue on disk needed):
    the loaders under test only use the parquet path and static helpers."""
    return object.__new__(MagneticumFetcherV2)


def _load_halo(tmp_path, **kwargs):
    path, pos, vel, mass = _write_particles_parquet(tmp_path, **kwargs)
    particles = _fetcher()._load_halo_particles(HALO_ID, path)
    return particles, pos, vel, mass


# ---------------------------------------------------------------------------
# why the fix exists
# ---------------------------------------------------------------------------

def test_precondition_raw_parquet_read_gives_object_dtype(tmp_path):
    """Documents the pandas behaviour the fix works around: a list column
    read from parquet and converted with .to_numpy() is an object array."""
    path, _, _, _ = _write_particles_parquet(tmp_path)
    vel = pd.read_parquet(path)["VEL"].to_numpy()

    assert vel.dtype == object
    assert vel.shape == (N_PARTICLES,)
    assert np.asarray(vel[0]).shape == (3,)


# ---------------------------------------------------------------------------
# the fix: MagneticumFetcherV2._column_to_numpy and the two loaders
# ---------------------------------------------------------------------------

def test_halo_loader_returns_2d_float_vectors(tmp_path):
    particles, pos, vel, mass = _load_halo(tmp_path)

    assert particles.star_v.shape == (N_PARTICLES, 3)
    assert particles.star_p.shape == (N_PARTICLES, 3)
    assert particles.star_v.dtype != object
    assert particles.star_p.dtype != object
    np.testing.assert_allclose(particles.star_v, vel)
    np.testing.assert_allclose(particles.star_p, pos)
    np.testing.assert_allclose(particles.star_m, mass)


def test_subhalo_loader_returns_2d_float_vectors(tmp_path):
    path, pos, vel, _ = _write_particles_parquet(tmp_path)
    fetcher = _fetcher()
    fetcher._get_halo_id_for_subhalo = lambda subhalo_id: HALO_ID

    particles = fetcher._load_subhalo_particles(SUBHALO_ID, path)

    assert particles.star_v.shape == (N_PARTICLES, 3)
    assert particles.star_p.shape == (N_PARTICLES, 3)
    assert particles.star_v.dtype != object
    np.testing.assert_allclose(particles.star_v, vel)
    np.testing.assert_allclose(particles.star_p, pos)


def test_loader_preserves_vector_dtype(tmp_path):
    """No forced dtype: the pipeline mixes float32/float64 across catalogues,
    so stacking must keep whatever the parquet stored."""
    particles, _, _, _ = _load_halo(tmp_path, vec_dtype=np.float32)

    assert particles.star_v.dtype == np.float32
    assert particles.star_p.dtype == np.float32


def test_column_to_numpy_leaves_other_columns_untouched():
    scalar = pd.Series(np.array([1.5, 2.5, 3.5]))
    ints = pd.Series(np.array([1, 2, 3], dtype=np.int64))
    empty = pd.Series([], dtype=object)

    out_scalar = MagneticumFetcherV2._column_to_numpy(scalar)
    out_ints = MagneticumFetcherV2._column_to_numpy(ints)
    out_empty = MagneticumFetcherV2._column_to_numpy(empty)

    np.testing.assert_array_equal(out_scalar, scalar.to_numpy())
    assert out_scalar.dtype == np.float64 and out_scalar.shape == (3,)
    assert out_ints.dtype == np.int64 and out_ints.shape == (3,)
    assert out_empty.shape == (0,)


# ---------------------------------------------------------------------------
# the original failure, end to end through the real loader
# ---------------------------------------------------------------------------

def test_control_native_float_star_v_generates_map(tmp_path):
    """Other simulations: star_v is already a native (N, 3) float array."""
    from src.data_generation.src.subhalo import SubhaloParticles

    rng = np.random.default_rng(0)
    particles = SubhaloParticles(
        data={
            "POS": rng.uniform(-1, 1, (N_PARTICLES, 3)),
            "VEL": rng.normal(0, 300, (N_PARTICLES, 3)),
            "MASS": rng.uniform(1e6, 1e7, N_PARTICLES),
        },
        column_map={"POS": "p", "MASS": "m", "VEL": "v"},
    )
    proc = VelocityDispersionProcessor({"name": "VelocityDispersion"})
    proc.process(halo=None, subhalo_particles=particles)

    input_ij, _, _ = proc.generate_map(None, None, particles, BINS, RANGING, axis=AXIS)

    assert input_ij.shape == (BINS, BINS)


def test_generate_map_with_magneticum_loaded_particles(tmp_path):
    """Previously raised ValueError: setting an array element with a sequence."""
    particles, _, _, _ = _load_halo(tmp_path)

    proc = VelocityDispersionProcessor({"name": "VelocityDispersion"})
    proc.process(halo=None, subhalo_particles=particles)

    input_ij, _, _ = proc.generate_map(None, None, particles, BINS, RANGING, axis=AXIS)

    assert input_ij.shape == (BINS, BINS)


def test_generate_map_after_icl_restacks_star_v(tmp_path):
    """Real pipeline order from the log: VelocityDispersion.process, then the ICL
    step's star_v restack (feature.py:898-900, inlined here so the 273 MB
    RandomForest pickle is not needed), then VelocityDispersion.generate_map.

    The processor's stored value must stay a valid (N, 3) float array even
    after the ICL step rebinds star_v on the shared particles object.
    """
    particles, _, _, _ = _load_halo(tmp_path)

    proc = VelocityDispersionProcessor({"name": "VelocityDispersion"})
    proc.process(halo=None, subhalo_particles=particles)

    # --- what ICLProcessor._compute_icl_mask does to the shared particles ---
    if (isinstance(particles.star_v, np.ndarray) and particles.star_v.dtype == object
            and len(particles.star_v) > 0 and isinstance(particles.star_v[0], (list, np.ndarray))):
        particles.star_v = np.stack(particles.star_v)
    # -------------------------------------------------------------------------

    assert proc.value.shape == (N_PARTICLES, 3)
    assert proc.value.dtype != object

    input_ij, _, _ = proc.generate_map(None, None, particles, BINS, RANGING, axis=AXIS)

    assert input_ij.shape == (BINS, BINS)
