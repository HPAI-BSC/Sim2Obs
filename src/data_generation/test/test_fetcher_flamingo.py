import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from hydra import initialize, compose
from omegaconf import OmegaConf

import rootutils
from src.data_generation.src.data_fetcher import FlamingoFetcher, DataFetcherFactory
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles

rootutils.setup_root(__file__, pythonpath=True)

PIPELINE_PATH = 'src.data_generation.src.data_fetcher.run_pipeline'
USE_REAL_DATA = 1
REAL_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")
REAL_SNAPNUM = "0073"


class FakeSwiftField:
    """
    Behaves like a real SWIFTDataset field.
    Supports boolean indexing, ==, np.isin, and unit conversion.
    """
    def __init__(self, values: np.ndarray):
        self._values = np.asarray(values)
        self.value = self._values
        self.convert_to_units = MagicMock()
        self.convert_to_physical = MagicMock()

    def __eq__(self, other):
        return self._values == other

    def __getitem__(self, idx):
        return self._values[idx]

    def __len__(self):
        return len(self._values)

    def __array__(self):
        return self._values


def make_swift_field(values: np.ndarray) -> FakeSwiftField:
    return FakeSwiftField(np.asarray(values))


def make_fake_swift_dataset(n: int = 10, host_index: np.ndarray = None):
    if host_index is None:
        host_index = np.array([-1] * n)
    ds = MagicMock()
    ds.soap.host_halo_index = make_swift_field(host_index)
    ds.input_halos.halo_centre = make_swift_field(np.random.uniform(0, 500, (n, 3)))
    ds.input_halos.halo_catalogue_index = make_swift_field(np.arange(n))
    ds.spherical_overdensity_200_crit.total_mass = make_swift_field(np.random.uniform(1e13, 1e15, n))
    ds.spherical_overdensity_200_crit.r_200_crit = make_swift_field(np.random.uniform(0.5, 2.0, n))
    return ds


def make_fake_particles(n: int = 100):
    particles = MagicMock()
    particles.stars.halo_catalogue_index = make_swift_field(np.repeat(np.arange(10), 10))
    particles.stars.coordinates = make_swift_field(np.random.uniform(0, 500, (n, 3)))
    particles.stars.masses = make_swift_field(np.random.uniform(1e8, 1e10, n))
    particles.stars.velocities = make_swift_field(np.random.uniform(-500, 500, (n, 3)))
    return particles


def make_fake_processed_halos(n: int = 10, host_index: np.ndarray = None) -> pd.DataFrame:
    if host_index is None:
        host_index = np.array([-1] * n)
    return pd.DataFrame({
        'halo_id':              np.arange(n),
        'fsub':                 np.arange(n),
        'gpos':                 [np.array([1., 2., 3.])] * n,
        'm200':                 np.random.uniform(1e13, 1e15, n),
        'r200':                 np.random.uniform(0.5, 2.0, n),
        'soap.host_halo_index': host_index,
    })


def make_fake_processed_subhalos(n: int = 10, host_index: np.ndarray = None) -> pd.DataFrame:
    if host_index is None:
        host_index = np.array([-1] * n)
    return pd.DataFrame({
        'subhalo_id':              np.arange(n),
        'id_parents':              np.zeros(n, dtype=int),
        'input_halos.halo_centre': [np.array([1., 2., 3.])] * n,
        'soap.host_halo_index':    host_index,
    })


def make_fake_processed_particles(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame({
        'stars.coordinates': [np.array([1., 2., 3.])] * n,
        'stars.masses':       np.random.uniform(1e8, 1e10, n),
    })


def _check_real_data_exists() -> tuple[bool, str]:
    """Check all required real data files exist."""
    simulation = "L1000N0900"
    soap_dir = os.path.join(REAL_DATA_PATH, "flamingo", simulation, "SOAP-HBT")
    snap_dir = os.path.join(REAL_DATA_PATH, "flamingo", simulation, "snapshots", f"flamingo_{REAL_SNAPNUM}")
    flamingo_file = os.path.join(soap_dir, f"membership_{REAL_SNAPNUM}", f"membership_{REAL_SNAPNUM}.0.hdf5")
    halo_props_file = os.path.join(soap_dir, f"halo_properties_{REAL_SNAPNUM}.hdf5")
    snap_file = os.path.join(snap_dir,  f"flamingo_{REAL_SNAPNUM}.0.hdf5")

    for path in (flamingo_file, halo_props_file, snap_file):
        if not os.path.exists(path):
            return False, f"Missing: {path}"
    return True, ""


def make_mock_catalog():
    return OmegaConf.create({
        'path':    '/fake/flamingo/path',
        'snapnum': '0077',
        'fields': {
            'halo': {
                'input_halos.halo_centre': {
                    'input': None, 'output': None, 'scaling': None, 'name': 'gpos'
                },
                'spherical_overdensity_200_crit.total_mass': {
                    'input': None, 'output': None, 'scaling': None, 'name': 'm200'
                },
                'spherical_overdensity_200_crit.r_200_crit': {
                    'input': None, 'output': None, 'scaling': None, 'name': 'r200'
                },
            },
            'subhalo': {
                'input_halos.halo_centre': {
                    'input': None, 'output': None, 'scaling': None, 'name': 'spos'
                },
            },
            'particles': {
                'stars': {
                    'coordinates': {
                        'input': None, 'output': None, 'scaling': None, 'name': 'p'
                    },
                    'masses': {
                        'input': None, 'output': None, 'scaling': None, 'name': 'm'
                    },
                }
            }
        }
    })


def make_real_catalog():
    with initialize(config_path="'../../../../../configs"):
        cfg_flamingo = compose(config_name="data_generation.yaml")

        # Load the local config for paths and assign them
        local_config = compose(config_name="paths/local.yaml")
        cfg_flamingo.paths = local_config.paths
        cfg_flamingo.paths.datasets_dir = REAL_DATA_PATH
        cfg_flamingo.catalog = compose("catalog/flamingo.yaml")['catalog']
    return cfg_flamingo


@pytest.fixture
def mock_catalog():
    return make_mock_catalog()


@pytest.fixture
def fetcher(mock_catalog):
    """Mock fetcher — used when USE_REAL_DATA=0."""
    fof = make_fake_swift_dataset(n=10)
    parts = make_fake_particles(n=100)

    with patch('os.path.exists', return_value=True), \
         patch('swiftsimio.load', side_effect=[fof, parts]):
        f = FlamingoFetcher(catalog=mock_catalog)

    f.fof_subhalos = fof
    f.particles = parts
    f.halo_fields = [
        'input_halos.halo_centre',
        'spherical_overdensity_200_crit.total_mass',
        'spherical_overdensity_200_crit.r_200_crit',
    ]
    f.subhalo_fields = ['input_halos.halo_centre']
    f.particle_fields = ['stars']
    f.column_map = {
        'input_halos.halo_centre':                   'gpos',
        'spherical_overdensity_200_crit.total_mass': 'm200',
        'spherical_overdensity_200_crit.r_200_crit': 'r200',
        'halo_id':                                   'halo_id',
        'fsub':                                      'fsub',
        'subhalo_id':                                'subhalo_id',
        'id_parents':                                'id_parents',
        'soap.host_halo_index':                      'soap.host_halo_index',
        'spos':                                      'spos',
    }
    return f


@pytest.fixture
def real_fetcher():
    """Real fetcher — used when USE_REAL_DATA=1."""
    if not USE_REAL_DATA:
        pytest.skip("USE_REAL_DATA=0 — skipping real data test")

    ok, reason = _check_real_data_exists()
    if not ok:
        pytest.skip(reason)
    config = make_real_catalog()

    required_keys = ['G']
    data_loader_dict = next((step for step in config.pipeline if step.name == 'data_loader'), None)
    attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
    fetcher = DataFetcherFactory.get_data_fetcher(config.catalog, **attr_dict)
    return fetcher


@pytest.fixture
def mock_halo():
    halo = MagicMock()
    halo.halo_id = 0
    halo.is_primary = -1
    halo.gpos = np.array([250.0, 250.0, 250.0])
    return halo


class TestFlamingoFetcher:

    def test_init_loads_files(self, mock_catalog):
        with patch('os.path.exists', return_value=True), \
             patch('swiftsimio.load') as mock_load:
            mock_load.return_value = MagicMock()
            FlamingoFetcher(catalog=mock_catalog)
            assert mock_load.call_count == 2

    def test_init_raises_if_file_missing(self, mock_catalog):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(Exception, match="does not exist"):
                FlamingoFetcher(catalog=mock_catalog)

    def test_init_sets_fof_and_particles(self, fetcher):
        assert fetcher.fof_subhalos is not None
        assert fetcher.particles is not None

    def test_get_halos_returns_halo_context(self, fetcher):
        mock_context = MagicMock(spec=HaloContext)
        mock_context.data = make_fake_processed_halos()
        with patch(PIPELINE_PATH, return_value=make_fake_processed_halos()), \
             patch('src.data_generation.src.data_fetcher.HaloContext',
                   return_value=mock_context) as mock_halo_ctx:
            result = fetcher.get_halos()
        assert mock_halo_ctx.called
        assert result is mock_context

    def test_get_halos_assigns_halo_ids(self, fetcher):
        mock_context = MagicMock(spec=HaloContext)
        mock_context.data = make_fake_processed_halos()
        with patch(PIPELINE_PATH, return_value=make_fake_processed_halos()), \
             patch('src.data_generation.src.data_fetcher.HaloContext',
                   return_value=mock_context):
            result = fetcher.get_halos()
        assert 'halo_id' in result.data.columns

    def test_get_halos_filters_non_primary(self, fetcher):
        host_index = np.array([-1, -1, 0, 0, -1, 1, -1, -1, -1, -1])
        fake_halos = make_fake_processed_halos(host_index=host_index)
        filtered = fake_halos[fake_halos['soap.host_halo_index'] == -1].copy()
        mock_context = MagicMock(spec=HaloContext)
        mock_context.data = filtered
        with patch(PIPELINE_PATH, return_value=fake_halos), \
             patch('src.data_generation.src.data_fetcher.HaloContext',
                   return_value=mock_context):
            result = fetcher.get_halos()
        assert len(result.data) == 7

    def test_get_halos_empty_dataset(self, fetcher):
        fetcher.fof_subhalos = "not a SWIFTDataset"
        with pytest.raises(Exception):
            fetcher.get_halos()

    def test_get_subhalos_returns_subhalo_context(self, fetcher):
        mock_context = MagicMock(spec=SubhaloContext)
        mock_context.data = make_fake_processed_subhalos()
        with patch(PIPELINE_PATH, side_effect=[
            make_fake_processed_halos(),
            make_fake_processed_subhalos(),
        ]), patch('src.data_generation.src.data_fetcher.SubhaloContext',
                  return_value=mock_context):
            result = fetcher.get_subhalos()
        assert isinstance(result, SubhaloContext)

    def test_get_subhalos_assigns_subhalo_ids(self, fetcher):
        mock_context = MagicMock(spec=SubhaloContext)
        mock_context.data = make_fake_processed_subhalos()
        with patch(PIPELINE_PATH, side_effect=[
            make_fake_processed_halos(),
            make_fake_processed_subhalos(),
        ]), patch('src.data_generation.src.data_fetcher.SubhaloContext',
                  return_value=mock_context):
            result = fetcher.get_subhalos()
        assert 'subhalo_id' in result.data.columns

    def test_get_subhalos_drops_host_halo_index_after_reindex(self, fetcher):
        host_index = np.array([-1, -1, 0, 0, -1, -1, -1, -1, -1, -1])
        fake_subhalos = make_fake_processed_subhalos(host_index=host_index)
        reindexed = fake_subhalos.drop(columns=['soap.host_halo_index'])
        mock_context = MagicMock(spec=SubhaloContext)
        mock_context.data = reindexed
        with patch(PIPELINE_PATH, side_effect=[
            make_fake_processed_halos(),
            fake_subhalos,
        ]), patch('src.data_generation.src.data_fetcher.SubhaloContext',
                  return_value=mock_context):
            result = fetcher.get_subhalos()
        assert 'soap.host_halo_index' not in result.data.columns

    def test_get_subhalos_empty_dataset(self, fetcher):
        fetcher.fof_subhalos = "not a SWIFTDataset"
        with pytest.raises(Exception):
            fetcher.get_subhalos()

    def test_get_p_subhalo_returns_subhalo_particles(self, fetcher):
        fetcher.fof_subhalos.soap.host_halo_index = make_swift_field(np.array([-1] * 10))
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        with patch(PIPELINE_PATH, return_value=make_fake_processed_particles()):
            result = fetcher.get_p_subhalo(subhalo_id=0)
        assert isinstance(result, SubhaloParticles)

    def test_get_p_subhalo_raises_if_not_found(self, fetcher):
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        with pytest.raises(ValueError, match="not a primary subhalo"):
            fetcher.get_p_subhalo(subhalo_id=999)

    def test_get_p_subhalo_raises_if_not_primary(self, fetcher):
        fetcher.fof_subhalos.soap.host_halo_index = make_swift_field(np.ones(10, dtype=int))
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        with pytest.raises(ValueError, match="not a primary subhalo"):
            fetcher.get_p_subhalo(subhalo_id=0)

    def test_get_p_subhalo_raises_if_id_missing(self, fetcher):
        with pytest.raises(ValueError, match="subhalo_id must be provided"):
            fetcher.get_p_subhalo()

    def test_get_p_halo_returns_subhalo_particles(self, fetcher, mock_halo):
        with patch(PIPELINE_PATH, return_value=make_fake_processed_particles()):
            result = fetcher.get_p_halo(halo=mock_halo)
        assert isinstance(result, SubhaloParticles)

    def test_get_p_halo_raises_if_not_primary(self, fetcher, mock_halo):
        mock_halo.is_primary = 1
        with pytest.raises(ValueError, match="not a primary subhalo"):
            fetcher.get_p_halo(halo=mock_halo)

    def test_get_p_halo_raises_if_halo_missing(self, fetcher):
        with pytest.raises(ValueError, match="halo must be provided"):
            fetcher.get_p_halo()

    def test_get_p_halo_includes_satellite_particles(self, fetcher, mock_halo):
        fetcher.fof_subhalos.soap.host_halo_index = make_swift_field(
            np.array([-1, -1, 0, 0, -1, -1, -1, -1, -1, -1])
        )
        fetcher.fof_subhalos.input_halos.halo_catalogue_index = make_swift_field(np.arange(10))
        fetcher.particles.stars.halo_catalogue_index = make_swift_field(
            np.repeat(np.arange(10), 10)
        )
        with patch(PIPELINE_PATH, return_value=make_fake_processed_particles(n=30)):
            result = fetcher.get_p_halo(halo=mock_halo)
        assert isinstance(result, SubhaloParticles)


@pytest.mark.skipif(not USE_REAL_DATA, reason="USE_REAL_DATA=0")
class TestFlamingoFetcherRealData:
    """
    Integration tests against real Flamingo data.

    Run with:
        USE_REAL_DATA=1 pytest tests/test_fetcher_flamingo.py -v -k RealData

    Expected structure:
        test/data/flamingo/
        ├── snapshots/
        │   └── flamingo_0073/
        │       └── flamingo_0073.0.hdf5
        └── SOAP-HBT/
            └── membership_0073/
                ├── flamingo_0073.hdf5
                └── halo_properties_0073.hdf5
    """

    def test_real_files_exist(self):
        ok, reason = _check_real_data_exists()
        assert ok, reason

    def test_real_get_halos_returns_halo_context(self, real_fetcher):
        result = real_fetcher.get_halos()
        assert isinstance(result, HaloContext)
        assert len(result.data) > 0

    def test_real_get_halos_has_required_columns(self, real_fetcher):
        result = real_fetcher.get_halos()
        for col in ('halo_id', 'fsub', 'gpos', 'm200', 'r200'):
            assert col in result.data.columns, f"Missing column: {col}"

    def test_real_get_halos_all_primary(self, real_fetcher):
        result = real_fetcher.get_halos()
        if 'soap.host_halo_index' in result.data.columns:
            assert (result.data['soap.host_halo_index'] == -1).all()

    def test_real_get_subhalos_returns_subhalo_context(self, real_fetcher):
        result = real_fetcher.get_subhalos()
        assert isinstance(result, SubhaloContext)
        assert len(result.data) > 0

    def test_real_get_subhalos_has_required_columns(self, real_fetcher):
        result = real_fetcher.get_subhalos()
        for col in ('subhalo_id', 'id_parents', 'spos'):
            assert col in result.data.columns, f"Missing column: {col}"

    def test_real_get_p_halo_returns_subhalo_particles(self, real_fetcher):
        halo_context = real_fetcher.get_halos()
        halo = halo_context.get_halo_by_id(halo_context.halo_id[0])
        result = real_fetcher.get_p_halo(halo=halo)
        assert isinstance(result, SubhaloParticles)
        assert len(result) > 0

    def test_real_get_p_halo_has_star_positions(self, real_fetcher):
        halo_context = real_fetcher.get_halos()
        halo = halo_context.get_halo_by_id(halo_context.halo_id[0])
        result = real_fetcher.get_p_halo(halo=halo)
        assert hasattr(result, 'star_p'), "star_p attribute missing"
        assert result.star_p.shape[1] == 3

    def test_real_get_p_subhalo_returns_subhalo_particles(self, real_fetcher):
        subhalo_context = real_fetcher.get_subhalos()
        subhalo_id = subhalo_context.subhalo_id[0]
        result = real_fetcher.get_p_subhalo(subhalo_id=subhalo_id)
        assert isinstance(result, SubhaloParticles)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])