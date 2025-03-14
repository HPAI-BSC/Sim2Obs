import unittest

import numpy as np
from hydra import initialize, compose

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.utils import read_dataset, apply_conversion, compute_fov


def assert_dicts_equal(dict1, dict2, atol=1e-8):
    # Ensure the dictionaries have the same keys
    assert dict1.keys() == dict2.keys(), "Dictionaries have different keys"

    # Iterate over the keys and compare the arrays
    for key in dict1:
        arr1 = dict1[key]
        arr2 = dict2[key]

        # Check if the arrays are equal (use np.allclose for floating-point arrays)
        assert np.allclose(arr1, arr2, atol=atol), f"Arrays for key '{key}' are not equal"


class TestUtils(unittest.TestCase):
    def test_get_particle_data(self):
        # https://arxiv.org/pdf/1706.09899
        itype = 4   # PartType
        path = '../../../local/data/datasets/eagle/RefL0050N0752'
        gn = 1
        sgn = 0
        gns = read_dataset(itype, 'GroupNumber', path)
        sgns = read_dataset(itype, 'SubGroupNumber', path)
        mask = np.logical_and(gns == gn, sgns == sgn)

        data = {
            'mass': read_dataset(itype, 'Mass', path)[mask],
            'coords': read_dataset(itype, 'Coordinates', path)[mask],
            'velocity': read_dataset(itype, 'Velocity', path)[mask]
        }
        print(len(data))

    def test_compute_fov(self):
        with initialize(config_path="configs"):
            self.cfg_big_illustris = compose(config_name="data_generation.yaml")
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_big_illustris.paths = local_config.paths
            self.cfg_big_illustris.catalog = compose("catalog/bigillustris300.yaml")['catalog']
        required_keys = ['G']
        data_loader_dict = next((step for step in self.cfg_big_illustris.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        fetcher_illustris = DataFetcherFactory.get_data_fetcher(self.cfg_big_illustris.catalog, **attr_dict)
        halo_context = fetcher_illustris.get_halos()
        halo_positions = halo_context.data['gpos'].to_dict()
        positions = np.array(list(halo_positions.values()))
        halo_mass = halo_context.data['m200'].to_numpy()
        fov_radius = compute_fov(positions, halo_mass, percentage=90)
        print(fov_radius)

    def test_find_values(self):
        comoving_resolution = 2.66 # for Ref-L0050N0752
        scale_factor = 0.84551589459619
        spatial_resolution = comoving_resolution /  scale_factor # Convert to physical kpc
        print(spatial_resolution)

        # bin calculation
        fov = 526.68 # 526.68 (should we use the same fov ? than illustris )
        r = fov / 2  # Radius of the field of view

        bin_size = spatial_resolution * 1.8
        bins = int(fov / bin_size)
        print(f'spatial resolution: {spatial_resolution}, fov: {fov}, r: {r}, bins: {bins}, bin size: {bin_size}')

        # Illustris
        spatial_resolution = 2.02

        # bin calculation
        fov = 526.68  # 526.68 (should we use the same fov ? than illustris )
        r = fov / 2  # Radius of the field of view

        bin_size = spatial_resolution * 1.8
        bins = int(fov / bin_size)
        print(f'spatial resolution: {spatial_resolution}, fov: {fov}, r: {r}, bins: {bins}, bin size: {bin_size}')

