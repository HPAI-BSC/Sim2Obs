import os
import tempfile

import h5py
import numpy as np
import pytest
from cython_fortran_file import FortranFile
from omegaconf import OmegaConf

from rootutils import rootutils
rootutils.setup_root(__file__, pythonpath=True)

from src.data_generation.src.data_fetcher import DataFetcherFactory, ObeliskFetcher
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext


# ── helpers ────────────────────────────────────────────────────────────────────

def make_mock_catalog(**overrides):
    cfg = {
        "name": "obelisk",
        "snapnum": 75,
        "a": 0.14289,
        "H": 0.7,
        "Lbox": 1000.0,
        "path": "/fake/obelisk/path",
        "fields": {
            "halo": {
                "mvir": {"input": "Msun", "output": "Msun/h", "scaling": "x * h", "name": "m200",
                          "filter": ["> 1e13 * 0.6774"]},
                "rvir": {"input": "kpc", "output": "kpc/h", "scaling": "x * h", "name": "r200"},
                "pos":  {"input": "box units", "output": "kpc/h", "scaling": "x * Lbox", "name": "gpos"},
            },
            "subhalo": {
                "mstar":   {"input": "Msun", "output": "Msun", "scaling": "x", "name": "mstr",
                             "filter": "> 1e10"},
                "r50_kpc": {"input": "kpc", "output": "kpc/h", "scaling": "x * h", "name": "shmr"},
                "pos":     {"input": "box units", "output": "kpc/h", "scaling": "x * Lbox", "name": "spos"},
            },
            "particles": {
                "stars": {
                    "position": {"input": None, "output": None, "scaling": None, "name": "p"},
                    "velocity": {"input": "km/s", "output": "km/s", "scaling": "x", "name": "v"},
                    "mass": {"input": "Msun", "output": "Msun/h", "scaling": "x * h", "name": "m"},
                },
            },
        },
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def make_fake_catalogue(n_halos: int = 2, n_sub_per_halo: int = 2, low_mass_halo: bool = False) -> dict:
    """
    Build a small AdaptaHOP-shaped catalogue: `n_halos` top-level (level=1,
    is_central=True) objects, each with `n_sub_per_halo` extra level=2
    (is_central=False) satellites sharing its `host`. IDs are deliberately
    NOT contiguous with row position, matching Obelisk's own ID namespace
    (host/hostsub reference `ID`, not row index). If `low_mass_halo` is
    set, an extra level=1/is_central row with mvir/mstar below both
    catalogue filters is appended, for filter-exclusion tests.

    `is_central` here is kept equal to `level == 1` for simplicity — real
    Obelisk data has cases where they diverge (see ObeliskFetcher's class
    docstring), but the fetcher reads `is_central` directly rather than
    deriving it from `level`, so that divergence doesn't need modelling
    here for these tests to be meaningful.
    """
    ID, level, is_central, host, mvir, rvir, mstar, r50_kpc, sigma = [], [], [], [], [], [], [], [], []
    x, y, z = [], [], []

    next_id = 100
    for i in range(n_halos):
        halo_id = next_id
        next_id += 1
        ID.append(halo_id); level.append(1); is_central.append(True); host.append(halo_id)
        mvir.append(5e13 + i * 1e13); rvir.append(200.0 + i * 10)
        mstar.append(2e10 + i * 1e9); r50_kpc.append(5.0 + i)
        sigma.append(300.0 + i * 10)
        x.append(0.1 * (i + 1)); y.append(0.2 * (i + 1)); z.append(0.3 * (i + 1))

        for j in range(n_sub_per_halo):
            sub_id = next_id
            next_id += 1
            ID.append(sub_id); level.append(2); is_central.append(False); host.append(halo_id)
            mvir.append(0.0); rvir.append(0.0)
            mstar.append(3e10 + j * 1e9); r50_kpc.append(2.0 + j)
            sigma.append(100.0 + j)
            x.append(0.1 * (i + 1) + 0.01 * (j + 1))
            y.append(0.2 * (i + 1) + 0.01 * (j + 1))
            z.append(0.3 * (i + 1) + 0.01 * (j + 1))

    if low_mass_halo:
        halo_id = next_id
        ID.append(halo_id); level.append(1); is_central.append(True); host.append(halo_id)
        mvir.append(1e10); rvir.append(10.0)
        mstar.append(5e9); r50_kpc.append(0.5); sigma.append(50.0)
        x.append(0.9); y.append(0.9); z.append(0.9)

    return {
        "ID": np.array(ID, dtype=np.int64),
        "level": np.array(level, dtype=np.int32),
        "is_central": np.array(is_central, dtype=bool),
        "host": np.array(host, dtype=np.int64),
        "mvir": np.array(mvir, dtype=np.float64),
        "rvir": np.array(rvir, dtype=np.float64),
        "mstar": np.array(mstar, dtype=np.float64),
        "r50_kpc": np.array(r50_kpc, dtype=np.float64),
        "sigma": np.array(sigma, dtype=np.float64),
        "x": np.array(x, dtype=np.float64),
        "y": np.array(y, dtype=np.float64),
        "z": np.array(z, dtype=np.float64),
    }


def write_fake_catalogue_h5(tmpdir: str, iout: int, data: dict) -> str:
    path = os.path.join(tmpdir, f"Obelisk_{iout:05d}_catalogue.h5")
    with h5py.File(path, "w") as hf:
        group = hf.create_group(f"output_{iout:05d}")
        for key, values in data.items():
            group.create_dataset(key, data=values)
    return path


def write_fake_gal_stars_file(tmpdir: str, iout: int, galaxy_id: int, n_star: int = 3,
                               header_id: int = None) -> str:
    """
    Write a minimal gal_stars_{ID:07d} Fortran binary file matching the
    record layout read by ObeliskFetcher._read_gal_stars_file (ported from
    src/data_generation/test/data/obelisk/galstars.py's read_galfile).
    `header_id` defaults to `galaxy_id`; pass a different value to test the
    header/filename mismatch check.
    """
    gal_dir = os.path.join(tmpdir, f"GAL_{iout:05d}")
    os.makedirs(gal_dir, exist_ok=True)
    path = os.path.join(gal_dir, f"gal_stars_{galaxy_id:07d}")
    hid = galaxy_id if header_id is None else header_id

    with FortranFile(path, mode='w') as ff:
        ff.write_vector(np.array([hid], dtype='i'))              # id
        ff.write_vector(np.array([1], dtype='i'))                # level
        ff.write_vector(np.array([1.0e10], dtype='d'))           # mass (galaxy-level)
        ff.write_vector(np.array([0.5, 0.5, 0.5], dtype='d'))    # position (galaxy-level)
        ff.write_vector(np.array([0.0, 0.0, 0.0], dtype='d'))    # velocity (galaxy-level)
        ff.write_vector(np.array([0.0, 0.0, 1.0], dtype='d'))    # angular_momentum
        ff.write_vector(np.array([n_star], dtype='q'))           # nparts (int64 — see _read_gal_stars_file)

        base = np.arange(n_star, dtype='d')
        ff.write_vector(base + 0.1)                              # position_x
        ff.write_vector(base + 0.2)                              # position_y
        ff.write_vector(base + 0.3)                              # position_z
        ff.write_vector(base + 1.0)                              # velocity_x
        ff.write_vector(base + 2.0)                              # velocity_y
        ff.write_vector(base + 3.0)                              # velocity_z
        ff.write_vector((base + 1.0) * 1e6)                      # mass (per-particle, Msun)
        ff.write_vector((base + 1).astype('q'))                  # particle id (int64)
        ff.write_vector(base + 5.0)                              # birth_time
        ff.write_vector(base * 0.01)                             # metallicity
    return path


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def catalogue_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def fetcher(catalogue_dir):
    write_fake_catalogue_h5(catalogue_dir, 75, make_fake_catalogue())
    catalog = make_mock_catalog(path=catalogue_dir)
    return ObeliskFetcher(catalog=catalog)


# ── tests ──────────────────────────────────────────────────────────────────────

class TestObeliskFetcher:

    # ── factory ─────────────────────────────────────────────────────────────

    def test_factory_returns_obelisk_fetcher(self, catalogue_dir):
        write_fake_catalogue_h5(catalogue_dir, 75, make_fake_catalogue())
        catalog = make_mock_catalog(path=catalogue_dir)
        fetcher = DataFetcherFactory.get_data_fetcher(catalog)
        assert isinstance(fetcher, ObeliskFetcher)

    # ── load_files ──────────────────────────────────────────────────────────

    def test_load_files_missing_path_raises(self):
        catalog = make_mock_catalog(path="/nonexistent/obelisk/path")
        with pytest.raises(Exception, match="does not exist"):
            ObeliskFetcher(catalog=catalog)

    def test_load_files_missing_group_raises(self, catalogue_dir):
        path = os.path.join(catalogue_dir, "Obelisk_00075_catalogue.h5")
        with h5py.File(path, "w") as hf:
            hf.create_group("output_00099")  # wrong snapshot group
        catalog = make_mock_catalog(path=catalogue_dir)
        with pytest.raises(Exception, match="not found"):
            ObeliskFetcher(catalog=catalog)

    # ── _get_halos ──────────────────────────────────────────────────────────

    def test_get_halos_returns_only_central_rows(self, fetcher):
        halo_context = fetcher.get_halos()
        assert isinstance(halo_context, HaloContext)
        # 2 central halos (IDs 100, 103 — see make_fake_catalogue layout)
        assert sorted(halo_context.halo_id) == [100, 103]

    def test_get_halos_assigns_halo_id_from_id_column(self, fetcher):
        halo_context = fetcher.get_halos()
        # halo_id must come from the catalogue's own ID, not a positional index
        assert set(halo_context.halo_id) == {100, 103}

    def test_get_halos_excludes_below_mass_filter(self, catalogue_dir):
        write_fake_catalogue_h5(catalogue_dir, 75, make_fake_catalogue(low_mass_halo=True))
        catalog = make_mock_catalog(path=catalogue_dir)
        fetcher = ObeliskFetcher(catalog=catalog)
        halo_context = fetcher.get_halos()
        # the extra low-mass halo (mvir=1e10) must be filtered out by
        # "m200 > 1e13 * 0.6774"
        assert 106 not in halo_context.halo_id

    def test_get_halos_position_scaled_by_lbox(self, fetcher):
        halo_context = fetcher.get_halos()
        halo_100 = halo_context.data.loc[100]
        # x=0.1, y=0.2, z=0.3 (box units) * Lbox=1000 -> [100, 200, 300]
        np.testing.assert_allclose(halo_100["gpos"], [100.0, 200.0, 300.0])

    def test_get_halos_mass_and_radius_scaled_by_h(self, fetcher):
        halo_context = fetcher.get_halos()
        halo_100 = halo_context.data.loc[100]
        assert halo_100["m200"] == pytest.approx(5e13 * 0.7)
        assert halo_100["r200"] == pytest.approx(200.0 * 0.7)

    # ── _get_subhalos ───────────────────────────────────────────────────────

    def test_get_subhalos_returns_all_rows(self, fetcher):
        subhalo_context = fetcher.get_subhalos()
        assert isinstance(subhalo_context, SubhaloContext)
        # 2 halos x (1 self + 2 satellites) = 6 rows total
        assert len(subhalo_context.data) == 6

    def test_get_subhalos_id_parents_self_for_centrals(self, fetcher):
        subhalo_context = fetcher.get_subhalos()
        central = subhalo_context.data.loc[(100, 100)]
        assert central["id_parents"] == 100

    def test_get_subhalos_id_parents_is_host_for_satellites(self, fetcher):
        subhalo_context = fetcher.get_subhalos()
        # satellites of halo 100 are IDs 101, 102 (see make_fake_catalogue)
        satellite = subhalo_context.data.loc[(100, 101)]
        assert satellite["id_parents"] == 100

    def test_get_subhalos_excludes_below_stellar_mass_filter(self, catalogue_dir):
        write_fake_catalogue_h5(catalogue_dir, 75, make_fake_catalogue(low_mass_halo=True))
        catalog = make_mock_catalog(path=catalogue_dir)
        fetcher = ObeliskFetcher(catalog=catalog)
        subhalo_context = fetcher.get_subhalos()
        # the extra low-mass object (mstar=5e9) must be filtered out by "mstr > 1e10"
        assert 106 not in subhalo_context.data["subhalo_id"].tolist()

    # ── get_p_subhalo / get_p_halo (star particles) ────────────────────────

    def test_get_p_subhalo_missing_kwarg_raises(self, fetcher):
        with pytest.raises(ValueError, match="subhalo_id"):
            fetcher.get_p_subhalo()

    def test_get_p_halo_missing_kwarg_raises(self, fetcher):
        with pytest.raises(ValueError, match="halo"):
            fetcher.get_p_halo()

    def test_get_p_subhalo_missing_file_raises(self, fetcher):
        with pytest.raises(Exception, match="does not exist"):
            fetcher.get_p_subhalo(subhalo_id=100)

    def test_get_p_subhalo_header_id_mismatch_raises(self, catalogue_dir, fetcher):
        # filename says 100, but the file's own header id is 999
        write_fake_gal_stars_file(catalogue_dir, 75, galaxy_id=100, header_id=999)
        with pytest.raises(Exception, match="expected 100"):
            fetcher.get_p_subhalo(subhalo_id=100)

    def test_get_p_subhalo_reads_star_particles(self, catalogue_dir, fetcher):
        write_fake_gal_stars_file(catalogue_dir, 75, galaxy_id=100, n_star=3)
        particles = fetcher.get_p_subhalo(subhalo_id=100)
        assert particles.star_m.shape == (3,)
        assert particles.star_p.shape == (3, 3)
        assert particles.star_v.shape == (3, 3)
        # mass scaling: "x * h", h=0.7 (see make_mock_catalog)
        np.testing.assert_allclose(particles.star_m, np.array([1.0, 2.0, 3.0]) * 1e6 * 0.7)

    def test_get_p_halo_aggregates_all_subhalos(self, catalogue_dir, fetcher):
        # halo 100 has itself (central) + satellites 101, 102 (see make_fake_catalogue)
        write_fake_gal_stars_file(catalogue_dir, 75, galaxy_id=100, n_star=2)
        write_fake_gal_stars_file(catalogue_dir, 75, galaxy_id=101, n_star=3)
        write_fake_gal_stars_file(catalogue_dir, 75, galaxy_id=102, n_star=4)

        halo_context = fetcher.get_halos()
        halo_100 = halo_context.get_halo_by_id(100)
        assert sorted(halo_100.subhalo_ids) == [100, 101, 102]

        particles = fetcher.get_p_halo(halo=halo_100)
        assert particles.star_m.shape == (2 + 3 + 4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
