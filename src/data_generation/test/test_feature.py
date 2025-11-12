import os

from matplotlib import pyplot as plt
from rootutils import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import pickle
import unittest
from unittest.mock import patch

import joblib
from numpy.testing import assert_array_equal

import numpy as np
from hydra import initialize, compose

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.feature import ICLProcessor, VelocityDispersionProcessor, FeatureProcessorFactory
from src.data_generation.src.utils import muse_bins_r


class TestFeature(unittest.TestCase):
    def setUp(self):
        # Illustris objs
        with open("data/illustris/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_illustris = pickle.load(file)

        with open("data/illustris/halo_0.pkl", "rb") as file:
            self.halo_illustris = pickle.load(file)

        # Eagle objs
        with open("data/eagle/subhalo_particles_0.pkl", "rb") as file:
            self.subhalo_particles_eagle = pickle.load(file)

        with open("data/eagle/halo_0.pkl", "rb") as file:
            self.halo_eagle = pickle.load(file)

        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")
            self.cfg_millenium = compose(config_name="data_generation.yaml")
            self.cfg_flamingo = compose(config_name="data_generation.yaml")
            self.cfg_tng_cluster = compose(config_name="data_generation.yaml")

            # Load the local config for paths and assign them
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_illustris.paths = local_config.paths
            self.cfg_eagle.paths = local_config.paths
            self.cfg_millenium.paths = local_config.paths
            self.cfg_flamingo.paths = local_config.paths
            self.cfg_tng_cluster.paths = local_config.paths

            # Dynamic catalog loading for both Illustris and Eagle
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']

            # Enforce RefL0050N0752 simulation code
            self.cfg_eagle.catalog.simulation_code = "RefL0050N0752"
            self.cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'

            self.cfg_millenium.catalog = compose("catalog/millenium.yaml")['catalog']
            self.cfg_millenium.catalog.path = '../../../local/data/datasets/millenium/MTNG-L31.25-270-A'

            self.cfg_flamingo.catalog = compose("catalog/flamingo.yaml")['catalog']
            self.cfg_flamingo.catalog.path = '../../../local/data/datasets/flamingo/L1000N0900'

            self.cfg_tng_cluster.catalog = compose("catalog/tngcluster.yaml")['catalog']
            self.cfg_tng_cluster.catalog.path = '../../../local/data/datasets/' # not used

        os.environ['PROJECT_ROOT'] = '../../..'
        required_keys = ['G']
        data_loader_dict = next((step for step in self.cfg_eagle.pipeline if step.name == 'data_loader'), None)
        attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        self.fetcher_flamingo = DataFetcherFactory.get_data_fetcher(self.cfg_flamingo.catalog, **attr_dict)
        self.fetcher_illustris = DataFetcherFactory.get_data_fetcher(self.cfg_illustris.catalog, **attr_dict)
        self.fetcher_millenium = DataFetcherFactory.get_data_fetcher(self.cfg_millenium.catalog, **attr_dict)
        self.fetcher_eagle = DataFetcherFactory.get_data_fetcher(self.cfg_eagle.catalog, **attr_dict)
        self.fetcher_tng_cluster = DataFetcherFactory.get_data_fetcher(self.cfg_tng_cluster.catalog, **attr_dict)

        model_path = 'src/data_generation/model/Best_Model-Random-3par.pkl'
        project_root = os.getenv('PROJECT_ROOT')
        feature_input_name, feature_output_name = self.cfg_illustris.feature_names
        self.input_feature_illustris = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_illustris)
        self.output_feature_illustris = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_illustris)
        self.output_feature_illustris.model = joblib.load(os.path.join(project_root, model_path))

        self.input_feature_eagle = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_eagle)
        self.output_feature_eagle = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_eagle)
        self.output_feature_eagle.model = joblib.load(os.path.join(project_root, model_path))

        self.input_feature_millenium = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_millenium)
        self.output_feature_millenium = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_millenium)
        self.output_feature_millenium.model = joblib.load(os.path.join(project_root, model_path))

        self.input_feature_tng_cluster = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg_tng_cluster)
        self.output_feature_tng_cluster = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg_tng_cluster)
        self.output_feature_tng_cluster.model = joblib.load(os.path.join(project_root, model_path))

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_ICLProcessor_process(self, mock_halo, mock_subhalo_particles):
        # Illustris
        mock_halo_instance = self.halo_illustris
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        length = mock_subhalo_particles_instance.star_v.shape[0]

        expected_mask = np.ones((length, ), dtype=int)  # ICL mask

        self.output_feature_illustris.process(mock_halo_instance, mock_subhalo_particles_instance)
        result = self.output_feature_illustris.value

        self.assertEqual(result.shape, (length, ))
        self.assertTrue(np.all(np.isin(result, [0, 1])))  # Check if it's a binary mask
        assert_array_equal(result, expected_mask)

        # Eagle
        mock_halo_instance = self.halo_eagle
        mock_halo.return_value = mock_halo_instance
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        length = mock_subhalo_particles_instance.star_v.shape[0]
        expected_mask = np.ones((length, ), dtype=int)  # ICL mask

        self.output_feature_eagle.process(mock_halo_instance, mock_subhalo_particles_instance)
        result = self.output_feature_eagle.value
        self.assertEqual(result.shape, (length, ))
        self.assertTrue(np.all(np.isin(result, [0, 1])))  # Check if it's a binary mask
        assert_array_equal(result, expected_mask)

    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_ICLProcessor_generate_maps(self, mock_subhalo_particles):
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance

        r = 263.34
        # Defined by Ilaria
        bins = muse_bins_r(2 * 263.34, 2.02, 1)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

        self.output_feature_illustris.value = np.ones((mock_subhalo_particles_instance.star_p.shape[0], ), dtype=int)  # ICL mask
        # Axis x-y
        icl_map, _, biny = self.output_feature_illustris.generate_map(subhalo_particles=mock_subhalo_particles_instance,
                                                                      bins=bins, ranging=ranging, axis=(0, 1))
        assert (len(icl_map)) == bins

        # Eagle
        self.output_feature_eagle.value = np.ones((self.subhalo_particles_eagle.star_p.shape[0], ), dtype=int)  # ICL mask
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        icl_map, _, biny = self.output_feature_eagle.generate_map(subhalo_particles=mock_subhalo_particles_instance,
                                                                  bins=bins, ranging=ranging, axis=(0, 1))
        assert (len(icl_map)) == bins

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_VelocityDispersionProcessor_process(self, mock_halo, mock_subhalo_particles):
        # Illustris
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        expected_shape = mock_subhalo_particles_instance.star_v.shape

        self.input_feature_illustris.process(mock_halo.return_value, mock_subhalo_particles_instance)
        result = self.input_feature_illustris.value
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

        # Eagle
        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance
        expected_shape = mock_subhalo_particles_instance.star_v.shape

        self.input_feature_eagle.process(mock_halo.return_value, mock_subhalo_particles_instance)
        result = self.input_feature_eagle.value
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.issubdtype(result.dtype, np.floating))

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_VelocityDispersionProcessor_generate_maps(self, mock_halo, mock_subhalo_particles):
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance

        mock_halo.return_value = self.halo_illustris
        self.input_feature_illustris.process(mock_halo.return_value, mock_subhalo_particles_instance)

        # Defined by Ilaria
        r = 263.34
        bins = muse_bins_r(2 * 263.34, 2.02, 1)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps
        # axis x-y
        vel_map, _, biny = self.input_feature_illustris.generate_map(mock_subhalo_particles_instance, bins,
                                                                ranging, axis=(0, 1))
        assert (len(vel_map)) == bins

        mock_halo.return_value = self.halo_eagle
        mock_subhalo_particles_instance = self.subhalo_particles_eagle
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance

        self.input_feature_eagle.process(mock_halo.return_value, mock_subhalo_particles_instance)
        vel_map, _, biny = self.input_feature_eagle.generate_map(mock_subhalo_particles_instance, bins,
                                                                ranging, axis=(0, 1))
        assert (len(vel_map)) == bins

    @patch('src.data_generation.src.halo.Halo')
    @patch('src.data_generation.src.subhalo.SubhaloParticles')
    def test_ICLProcessor_get_transition_radius(self, mock_halo, mock_subhalo_particles):
        # Illustris
        mock_halo_instance = self.halo_illustris
        mock_halo.return_value = self.halo_illustris
        mock_subhalo_particles_instance = self.subhalo_particles_illustris
        mock_subhalo_particles.return_value = mock_subhalo_particles_instance

        self.output_feature_illustris.process(mock_halo_instance, mock_subhalo_particles_instance)
        icl_mask = self.output_feature_illustris.value
        self.output_feature_illustris._compute_transition_radius(mock_halo.return_value,
                                                                 mock_subhalo_particles.return_value,
                                                                 icl_mask)
        self.assertIsNotNone(self.output_feature_illustris.transition_radius)

    def test_exploration_transition_radius(self):
        def plot_particles(x, y, z, halo_center):
            # Create a 3D scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot all the particles
            ax.scatter(x, y, z, c='r', marker='o')  # Particles in red

            # Plot the halo center with a different color (e.g., blue) and a larger marker
            ax.scatter(halo_center[0], halo_center[1], halo_center[2], c='b', marker='*',
                       s=100)  # Halo center in blue, larger marker

            # Add labels for clarity
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()

        halo_id = 1017
        path = f'../../../transition_radius_exploration/halo_{halo_id}_MTNG-L125-1080-A.joblib'
        halo = joblib.load(path)
        path = f'../../../transition_radius_exploration/subhalo_particles_{halo_id}_MTNG-L125-1080-A.joblib'
        # Already preprocessed position = position - halo position
        subhalo_particles = joblib.load(path)
        path = f'../../../transition_radius_exploration/icl_mask_{halo_id}_MTNG-L125-1080-A.joblib'
        icl_mask = joblib.load(path)

        self.output_feature_millenium._compute_transition_radius(halo, subhalo_particles, icl_mask)
        print(self.output_feature_millenium.transition_radius)

        # Extract x, y, z coordinates
        x = subhalo_particles.star_p[:, 0]
        y = subhalo_particles.star_p[:, 1]
        z = subhalo_particles.star_p[:, 2]

        plot_particles(x, y, z, halo_center=[0, 0, 0])

    def test_icl_never_dominates(self):
        def plot_particles(x, y, z, halo_center):
            # Create a 3D scatter plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot all the particles
            ax.scatter(x, y, z, c='r', marker='o')  # Particles in red

            # Plot the halo center with a different color (e.g., blue) and a larger marker
            ax.scatter(halo_center[0], halo_center[1], halo_center[2], c='b', marker='*',
                       s=100)  # Halo center in blue, larger marker

            # Add labels for clarity
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()

        halo_id = 7869072
        path = f'data/tng_cluster/halo_illustris-300_tng_cluster_{halo_id}.pkl'
        # Load the pickle file
        with open(path, "rb") as f:
            halo = pickle.load(f)

        path = f'data/tng_cluster/subhalo_particles_illustris-300_tng_cluster_{halo_id}.pkl'
        with open(path, "rb") as f:
            subhalo_particles = pickle.load(f)

        path = f'data/tng_cluster/icl_mask_illustris-300_tng_cluster_{halo_id}.npy'
        icl_mask = np.load(path)

        try:
            self.output_feature_tng_cluster._compute_transition_radius(halo, subhalo_particles, icl_mask)
            print(self.output_feature_tng_cluster.transition_radius)
        except Exception as e:
            print(e)

        # Extract x, y, z coordinates
        x = subhalo_particles.star_p[:, 0]
        y = subhalo_particles.star_p[:, 1]
        z = subhalo_particles.star_p[:, 2]

        plot_particles(x, y, z, halo_center=[0, 0, 0])

    def test_icl_mask_empty(self):
        halo_id = 3423689
        path = f'data/tng_cluster/halo_illustris-300_tng_cluster_{halo_id}.pkl'
        # Load the pickle file
        with open(path, "rb") as f:
            halo = pickle.load(f)

        path = f'data/tng_cluster/subhalo_particles_illustris-300_tng_cluster_{halo_id}.pkl'
        with open(path, "rb") as f:
            subhalo_particles = pickle.load(f)

        path = f'data/tng_cluster/icl_mask_illustris-300_tng_cluster_{halo_id}.npy'
        icl_mask = np.load(path)

        self.output_feature_tng_cluster.process(halo, subhalo_particles)
        _ = self.output_feature_illustris.value

    def test_generate_maps_halo_0(self):
        halo_id = 0
        path = f'../../../transition_radius_exploration/subhalo_particles_1017_MTNG-L125-1080-A.joblib'
        # Already preprocessed position = position - halo position
        subhalo_particles = joblib.load(path)
        path = f'../../../transition_radius_exploration/icl_mask_1017_MTNG-L125-1080-A.joblib'
        icl_mask = joblib.load(path)

        r = 637.031
        spatial_resolution = 1.480  # kpc
        bin_spatial_resolution_scale = 1.5
        bins = muse_bins_r(2 * r, spatial_resolution, bin_spatial_resolution_scale)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps
        # axis x-y
        self.input_feature_illustris.value = subhalo_particles.star_v
        vel_map, _, biny = self.input_feature_illustris.generate_map(subhalo_particles, bins, ranging, axis=(0, 2))
        plt.imshow(vel_map)
        plt.show()

        self.output_feature_illustris.value = icl_mask
        icl_map, _, biny = self.output_feature_illustris.generate_map(subhalo_particles, bins, ranging, axis=(0, 2))
        plt.imshow(icl_map)
        plt.show()





