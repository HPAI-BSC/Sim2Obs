import os
import pickle
import unittest

import pyarrow.dataset
import pytest
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import numpy as np
from dotenv import load_dotenv
from hydra import initialize, compose
from unittest.mock import patch
from omegaconf import OmegaConf

from src.data_generation.src.data_fetcher import DataFetcherFactory, MagneticumFetcherV2
from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.lazy_subhalo import LazySubhaloContext

load_dotenv('../.env')


@pytest.fixture
def mock_catalog():
    catalog = OmegaConf.create({
        'path':    '/fake/path',
        'snapnum': 120,
        'fields': {
            'halo': {
                'GOFF': {'input': None, 'output': None, 'scaling': None, 'name': 'goff'},
                'GLEN': {'input': None, 'output': None, 'scaling': None, 'name': 'glen'},
                'FSUB': {'input': None, 'output': None, 'scaling': None, 'name': 'fsub'},
            },
            'subhalo':   {},
            'particles': {}
        }
    })
    return catalog


@pytest.fixture
def fetcher(mock_catalog):
    # MagneticumFetcherV2 with cache pre-loaded — skips _load_catalog_cache.
    with patch('g3read.read_new'), patch('g3read.yield_all_files', return_value=[]):
        f = MagneticumFetcherV2(catalog=mock_catalog)

    # Inject fake metadata directly — bypasses file I/O
    f._cache_loaded = True
    f._group_base = '/fake/group_base'
    f._snap_base  = '/fake/snap_base'
    f._goff = np.array([0, 10, 20, 30, 40])
    f._glen = np.array([10, 10, 10, 10, 10])
    f._fsub = np.array([0,  2,  4,  6,  8])
    f._nsub = np.array([2,  2,  2,  2,  2])
    f._m200 = np.array([200000000, 3000000, 4000000, 5000000, 60000000])
    f._r200 = np.array([200000000, 3000000, 4000000, 5000000, 60000000])
    f.halo_fields = ['GOFF', 'GLEN', 'FSUB', 'MCRI', 'RCRI']
    f.column_map = {'GOFF': 'goff', 'GLEN': 'glen', 'FSUB': 'fsub', 'MCRI': 'm200', 'RCRI': 'r200'}
    f._pid_file_boundaries = []
    return f


def make_fake_halos(n=5):
    # Minimal raw halo dict as g3read would return.
    return {
        'GOFF': np.arange(n) * 10,
        'GLEN': np.full(n, 10),
        'FSUB': np.arange(n) * 2,
        'NSUB': np.full(n, 2),
        'MCRI': np.full(n, 100000000),
        'RCRI': np.full(n, 100000)
    }


class TestFetcherMagneticum(unittest.TestCase):
    cfg_magneticum = None

    @classmethod
    def setUpClass(cls):
        os.environ['PROJECT_ROOT'] = '../../..'

        # Initialize Hydra and load configuration
        with initialize(config_path="../../../configs"):
            cls.cfg_magneticum = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            cls.cfg_magneticum.paths = local_config.paths
            cls.cfg_magneticum.catalog = compose("catalog/magneticum.yaml")['catalog']
            cls.cfg_magneticum.catalog.path = 'data/magneticum'

        required_keys = ['G']
        data_loader_dict = next(
            (step for step in cls.cfg_magneticum.pipeline if step.name == 'data_loader'), None
        )
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        cls.fetcher_magneticum = DataFetcherFactory.get_data_fetcher(cls.cfg_magneticum.catalog, **attr_dict)

    def test_returns_halo_context(self):
        fetcher = self.fetcher_magneticum
        with patch('g3read.read_new', return_value=make_fake_halos()):
            result = fetcher.get_halos()
        assert isinstance(result, HaloContext)

    def test_correct_number_of_halos(self):
        fetcher = self.fetcher_magneticum
        with patch('g3read.read_new', return_value=make_fake_halos(5)):
            result = fetcher.get_halos()
        assert len(result.data) == 5

    def test_halo_ids_assigned(self):
        fetcher = self.fetcher_magneticum
        with patch('g3read.read_new', return_value=make_fake_halos(5)):
            result = fetcher.get_halos()
        assert 'halo_id' in result.data.columns
        assert sorted(result.data['halo_id'].tolist()) == list(range(5))

    def test_subhalo_ids_assigned(self):
        fetcher = self.fetcher_magneticum
        with patch('g3read.read_new', return_value=make_fake_halos(5)):
            result = fetcher.get_halos()
        assert 'subhalo_ids' in result.data.columns
        # halo 0 → fsub=0, nsub=2 → subhalo_ids=[0, 1]
        assert list(result.data.loc[result.data['halo_id'] == 0, 'subhalo_ids'].iloc[0]) == [0, 1]

    def test_fsub_added_if_missing(self):
        fetcher = self.fetcher_magneticum
        fetcher.halo_fields = ['MCRI', 'RCRI']  # FSUB missing
        with patch('g3read.read_new', return_value=make_fake_halos(5)):
            fetcher.get_halos()
        assert 'FSUB' in fetcher.halo_fields

    def test_empty_halos_returns_empty_context(self):
        fetcher = self.fetcher_magneticum
        with patch('g3read.read_new', return_value={}):
            with pytest.raises(Exception):
                _ = fetcher.get_halos()

    def test_load_catalog_cache_called_if_not_loaded(self):
        fetcher = self.fetcher_magneticum
        fetcher._cache_loaded = False
        with patch('g3read.read_new', return_value=make_fake_halos(5)):
            fetcher.get_halos()
        self.assertTrue(fetcher._cache_loaded)

    def test_get_subhalos(self):
        fetcher = self.fetcher_magneticum
        result = fetcher.get_subhalos()
        import pyarrow.dataset as ds
        self.assertIsInstance(result, LazySubhaloContext)
        self.assertIsInstance(result.dataset, ds.Dataset)

    def no_test_get_p_halo(self):
        mock_halo = {'halo_id': 0, 'm200': 26000000, 'r200': 744.56, 'fsub': 0, 'gpos': [200, 200, 200]}
        halo = Halo(**mock_halo)
        fetcher = self.fetcher_magneticum
        result = fetcher.get_p_halo(halo=halo)

    def test_store_particles(self):
        fetcher = self.fetcher_magneticum
        fetcher._store_particles()



