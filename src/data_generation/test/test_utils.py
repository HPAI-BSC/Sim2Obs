import unittest

import numpy as np
from src.data_generation.src.utils import read_dataset, to_list, to_list_in_batches


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

    def test_to_list_batches(self):
        random_array = np.random.rand(10, 3)
        col = 'random_column'
        d = {col: random_array}
        converted = to_list(d, col)
        converted_ii = to_list_in_batches(d, col, batch_size=2)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(converted, converted_ii)))

    def test_find_values(self):
        comoving_resolution = 2.66 # for Ref-L0050N0752
        scale_factor = 0.84551589459619
        spatial_resolution = comoving_resolution / scale_factor # Convert to physical kpc
        print(spatial_resolution)

        # bin calculation
        fov = 526.68 # 526.68 (should we use the same fov ? than illustris )
        r = fov / 2  # Radius of the field of view

        bin_size = spatial_resolution * 3
        bins = int(fov / bin_size)
        print(f'spatial resolution: {spatial_resolution}, fov: {fov}, r: {r}, bins: {bins}, bin size: {bin_size}')

    def test_ckdtree(self):
        import numpy as np
        from scipy.spatial import cKDTree

        positions = np.array([
            [0, 0, 0],  # index 0 → maybe halo ID 100
            [0, 1, 0],  # index 1 → maybe halo ID 101
            [2, 2, 2],  # index 2 → maybe halo ID 102
        ])
        halo_ids = np.array([100, 101, 102])

        tree = cKDTree(positions)

        # Query neighbors within distance 1.5 of point [0,0,0]
        indices = tree.query_ball_point([0, 0, 0], r=1.5)

        print(indices)  # ➝ [0, 1]
        print(halo_ids[indices])  # ➝ [100, 101] ← you get the actual halo IDs here

    def test_plot_transition_radius(self):
        # Initialize Hydra and load configuration
        with initialize(config_path="configs"):
            # Initialize the cfg_illustris and cfg_eagle
            cfg_illustris = compose(config_name="data_generation.yaml")
            cfg_eagle = compose(config_name="data_generation.yaml")
            cfg_flamingo = compose(config_name="data_generation.yaml")
            cfg_millenium = compose(config_name="data_generation.yaml")

        transition_radius_list = np.arange(10)
        plot_transition_radius(cfg_illustris, transition_radius_list, log_normalization=True)



