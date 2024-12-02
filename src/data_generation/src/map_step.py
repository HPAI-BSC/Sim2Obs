import os
from typing import List

import numpy as np
from omegaconf import DictConfig

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.utils import muse_bins_r, circle, setup_output_directories


class MapStep(PipelineStep):
    def __init__(self, cfg: DictConfig):
        self.map_configuration = self._build_map_configuration(cfg)

    @staticmethod
    def _build_map_configuration(cfg: DictConfig) -> dict:
        """
        Builds the configuration for the mapping step.

        Args:
            cfg (DictConfig): Configuration object.

        Returns:
            dict: A dictionary containing mapping configuration.
        """
        required_keys = ['path', 'name', 'n_zooms']
        for key in required_keys:
            if key not in cfg:
                if key not in cfg.get('catalog', {}):
                    raise ValueError(f"Missing required configuration key: {key}")

        return {
            'root_path': cfg.catalog.path,
            'number_zooms': cfg.n_zooms,
            'catalog_name': cfg.catalog.name,
            'rotation_angle': cfg.rotation_group[0]
        }

    # TODO to be tested
    # TODO change input param
    @staticmethod
    def _create_maps(cosmos_data: CosmosData, input_feature: FeatureProcessor,
                     output_feature: FeatureProcessor) -> List:
        bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
        ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

        # Create three different maps based on projections onto different coordinate planes (X-Y, Y-Z, X-Z).
        # Break up the 2D spatial plane into small regions (bins), compute the mean of the squared velocities
        # in each region, and return the resulting grid of values.
        pos_rot = cosmos_data.subhalo_particle_positions

        # Map for the input feature
        input_xy, _, biny_xy = input_feature.generate_map(pos_rot, bins, ranging, axis=(0, 1))
        input_yz, _, biny_yz = input_feature.generate_map(pos_rot, bins, ranging, axis=(1, 2))
        input_xz, _, biny_xz = input_feature.generate_map(pos_rot, bins, ranging, axis=(0, 2))

        # Map for the output feature
        output_xy, _, _ = output_feature.generate_map(pos_rot, bins, ranging, axis=(0, 1))
        output_yz, _, _ = output_feature.generate_map(pos_rot, bins, ranging, axis=(1, 2))
        output_xz, _, _ = output_feature.generate_map(pos_rot, bins, ranging, axis=(0, 2))

        maps = [
            (input_xy, output_xy, biny_xy),
            (input_yz, output_yz, biny_yz),
            (input_xz, output_xz, biny_xz)
        ]
        return maps

    @staticmethod
    def _get_interlopers_id(cosmos_data: CosmosData):
        bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius

        # Considering as interloper any subhalo contained in a box defined by [-r, r] in the x and y directions and
        # inside [-R_200, R_200] along the z axis

        spos = np.array(cosmos_data.subhalos.data['spos'].tolist())

        mask = (
                (spos[:, 0] < r) &  # x condition
                (spos[:, 1] < r) &  # y condition
                (spos[:, 2] < cosmos_data.halo.r200)  # z condition
        )

        interlopers_id = cosmos_data.subhalos.data['subhalo_id'][mask]

        # removing first galaxy (central/BCG), since it's usually the central galaxy or BCG
        # TODO is it possible to have no interlopers ?
        interlopers_id = interlopers_id.iloc[1:]

        if len(interlopers_id) == 0:
            pass
            #log.warning('There is no interlopers.')
        return interlopers_id

    @staticmethod
    def _get_interlopers_mask(cosmos_data: CosmosData, map_shape, interlopers_id: np.ndarray, biny: np.ndarray) -> np.ndarray:
        """
        Create a mask to identify and exclude interloper subhalos from calculations.

        Parameters:
        ----------
        label : np.ndarray
            A 2D array representing the labels of the particles.

        vel_disp : np.ndarray
            A 2D array representing the velocity dispersion.

        spos_rot : np.ndarray
            A 2D array containing the spatial positions of each subhalo.

        shmr : pd.DataFrame
            A DataFrame containing the SubhaloHalfmassRadType of subhalos.

        interlopers_id : np.ndarray
            An array of identifiers for the interloper subhalos.

        biny : np.ndarray
            An array representing the y-coordinates for the masking grid.

        Returns:
        -------
        np.ndarray
            A boolean mask identifying interloper regions.
        """
        bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius for the grid
        interlopers_mask = np.zeros(map_shape)

        # Generate X and Y based on maschera's shape for consistency
        rows, cols = interlopers_mask.shape
        X, Y = np.meshgrid(
            np.linspace(-r, r, cols),  # X coordinates matching columns in maschera
            np.linspace(biny[0], biny[-1], rows)  # Y coordinates matching rows in maschera
        )

        for i in interlopers_id:
            # For each interloper get its center position (h, k) and its radius
            subhalo = cosmos_data.subhalos.get_item(i)
            h = subhalo.spos[i, 0]
            k = subhalo.spos[i, 1]
            rradius = subhalo.shmr[i, 0]
            for cc in range(rows):
                for dd in range(cols):
                    radius_circle = circle(X[cc, dd], Y[cc, dd], h, k, rradius)
                    if radius_circle <= 1:
                        interlopers_mask[cc, dd] = 1

        # Check if the mask is just zero
        if np.all(interlopers_mask == 0):
            pass
            #log.warning('Mask is full of zeros.')
        interlopers_mask = interlopers_mask.astype(bool)
        return interlopers_mask

    @staticmethod
    def _create_random_zoom(input_feature: FeatureProcessor, output_feature: FeatureProcessor,
                            interlopers_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a random zoom of the masked labels and velocity dispersion maps.

        Parameters:
        ----------
        label_masked : np.ndarray
            A 2D array representing the masked labels.

        vel_disp_masked : np.ndarray
            A 2D array representing the masked velocity dispersion.

        Returns:
        -------
        tuple
            A tuple containing:
            - label_masked1 : np.ndarray
                A smaller 2D array of masked labels.
            - vel_disp_masked1 : np.ndarray
                A smaller 2D array of masked velocity dispersion.
        """
        bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
        xo, yo = (int(bins / 2), int(bins / 2))
        xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)

        input_feature_zoom = input_feature.map[xoprime - int(bins / 4):xoprime + int(bins / 4),
                     yoprime - int(bins / 4): yoprime + int(bins / 4)]

        output_feature_zoom = output_feature.map[xoprime - int(bins / 4):xoprime + int(bins / 4),
                  yoprime - int(bins / 4): yoprime + int(bins / 4)]

        mask_zoom = interlopers_mask[xoprime - int(bins / 4):xoprime + int(bins / 4),
                yoprime - int(bins / 4):yoprime + int(bins / 4)]
        return input_feature_zoom, output_feature_zoom, mask_zoom

    def _create_n_zooms(self, projection_str: str, cosmos_data: CosmosData, input_feature: FeatureProcessor,
                        output_feature: FeatureProcessor, interlopers_mask: np.ndarray) -> None:

        def get_path(feature_name, halo_id_, subhalo_id_, projection_, i_):
            if self.map_configuration['rotation_angle'] is not None:
                path = os.path.join(self.map_configuration['root_path'],
                                    f'rotation_{self.map_configuration["rotation_angle"]}',
                                    feature_name,
                                    f'halo-{halo_id_}_subhalo-{subhalo_id_}_{projection_}_zoom_{i_}')
            else:
                path = os.path.join(self.map_configuration['root_path'],
                                    feature_name,
                                    f'halo-{halo_id_}_subhalo-{subhalo_id_}_{projection_}_zoom_{i_}')
            return path

        setup_output_directories(self.map_configuration['root_path'],
                                 self.map_configuration['rotation_angle'],
                                 input_feature.name,
                                 output_feature.name)

        for i in range(self.map_configuration['number_zooms']):
            input_feature_i, output_feature_i, mask_i = self._create_random_zoom(input_feature, output_feature,
                                                                                 interlopers_mask)

            # Expand the dimensions of the mask to match the shape of the other arrays
            mask_expanded = np.expand_dims(mask_i, axis=-1)  # Shape: (H, W, 1)

            # Expand the dimensions of vel_disp_i and label_i to make them 3D
            input_featute_i_expanded = np.expand_dims(input_feature_i, axis=-1)  # Shape: (H, W, 1)
            output_feature_i_expanded = np.expand_dims(output_feature_i, axis=-1)  # Shape: (H, W, 1)

            # Concatenate the mask as an additional channel to the zoomed arrays
            input_feature_with_mask = np.concatenate((input_featute_i_expanded, mask_expanded), axis=-1)  # Shape: (H, W, 2)
            output_feature_with_mask = np.concatenate((output_feature_i_expanded, mask_expanded), axis=-1)  # Shape: (H, W, 2)

            input_path = get_path(input_feature.name, cosmos_data.halo.halo_id, cosmos_data.subhalo_id,
                                  projection_str, i)
            output_path = get_path(output_feature.name, cosmos_data.halo.halo_id, cosmos_data.subhalo_id,
                                   projection_str, i)

            if os.path.exists(input_path):
                print()
            np.savez(input_path, input_feature_with_mask)
            np.savez(output_path, output_feature_with_mask)

    def _process_projection(self, feature_size, interlopers_id: np.ndarray, biny_i, projection: str,
                            cosmos_data: CosmosData, input_feature: FeatureProcessor, output_feature: FeatureProcessor):
        interlopers_mask = self._get_interlopers_mask(cosmos_data, feature_size, interlopers_id, biny_i)
        self._create_n_zooms(projection, cosmos_data, input_feature, output_feature, interlopers_mask)

    # TODO see how to generalize this type4
    def process(self, input_feature: FeatureProcessor, output_feature: FeatureProcessor, cosmos_data: CosmosData):
        interlopers_id = self._get_interlopers_id(cosmos_data)
        io_biny_projections = self._create_maps(cosmos_data, input_feature, output_feature)
        projections = ["xy", "yz", "xz"]

        for j, (map_f_i, map_f_o, biny_i) in enumerate(io_biny_projections):
            projection_name = projections[j]
            input_feature.map = map_f_i
            output_feature.map = map_f_o
            self._process_projection(map_f_o.shape, interlopers_id, biny_i, projection_name, cosmos_data, input_feature,
                                     output_feature)


