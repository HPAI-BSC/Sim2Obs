import os
import re
from typing import List, Generator, Callable

import numpy as np
from omegaconf import DictConfig
from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.rotate_step import RotateStep
from src.data_generation.src.subhalo import SubhaloContext
from src.data_generation.src.utils import muse_bins_r, circle, setup_output_directories, setup_logger

logger = setup_logger("MapStep")


class MapStep(PipelineStep):
    name = 'map_step'

    def __init__(self, cfg: DictConfig):
        self.map_configuration = self._build_map_configuration(cfg)

        self.transition_radius = cfg.catalog.get('transition_radius', None)
        # TODO review
        self.simulate_muse = cfg.feature_names[0].get('simulate_muse', None)
        self.cfg = cfg

        # Compute bins and radius
        self.spatial_resolution = self.map_configuration['spatial_resolution']
        self.bin_spatial_resolution_scale = self.map_configuration['bin_spatial_resolution_scale']
        self.dim = self.map_configuration['dim']

    @classmethod
    def _build_map_configuration(cls, cfg: DictConfig) -> dict:
        """
        Builds the configuration for the mapping step.

        Args:
            cfg (DictConfig): Configuration object.

        Returns:
            dict: A dictionary containing mapping configuration.
        """
        if cls.name not in [e['name'] for e in cfg.pipeline]:
            raise ValueError('Error, To execute a map step you must specify the map step in the pipeline of'
                             ' the configuration.')

        map_step_dict = next((step for step in cfg.pipeline if step.name == cls.name), None)

        if 'n_zooms' not in map_step_dict:
            raise ValueError('Error, n_zooms can not be found in the map step of the pipeline.')

        required_keys = ['path', 'name', 'spatial_resolution']
        for key in required_keys:
            if key not in cfg:
                if key not in cfg.get('catalog', {}):
                    raise ValueError(f"Error, missing required configuration key: {key}")

        # Check if there are a previous rotate_step
        rotate_step_dict = next((step for step in cfg.pipeline if step.name == RotateStep.name), None)
        rotation_angle = None
        if rotate_step_dict is not None:
            rotation_angle, _, _ = rotate_step_dict.rotation_group

        return {
            'dim': map_step_dict.get('dim', 2),
            'root_path': cfg.catalog.path,
            'number_zooms': map_step_dict.n_zooms,
            'catalog_name': cfg.catalog.name,
            'rotation_angle': rotation_angle,
            'spatial_resolution': cfg.catalog.spatial_resolution,
            'bin_spatial_resolution_scale': cfg.catalog.bin_spatial_resolution_scale,
        }

    def _create_maps(self, cosmos_data: CosmosData, input_feature: FeatureProcessor,
                     output_feature: FeatureProcessor) -> List:
        """
        Generates projection maps for both input and output features.

        Args:
            cosmos_data (CosmosData): Cosmos data containing subhalo and particle information.
            input_feature (FeatureProcessor): Feature processor for input data.
            output_feature (FeatureProcessor): Feature processor for output data.

        Returns:
            List: A list of tuples containing the generated maps for each projection.
        """
        # Subhalo particles with overlapping particles added
        subhalo_particles = cosmos_data.subhalo_particles

        # Subhalo particles with NO overlapping particles
        non_overlapping = cosmos_data.non_overlapping

        r = self.transition_radius
        bins = cosmos_data.metadata['map']['bins']
        logger.debug(f'Generating maps for r={r}, bins={bins} and num particles={len(subhalo_particles.star_p)}')

        if self.dim == 2:
            ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

            # Create three different maps based on projections onto different coordinate planes (X-Y, Y-Z, X-Z).
            # Break up the 2D spatial plane into small regions (bins), compute the mean of the squared velocities
            # in each region, and return the resulting grid of values.

            # Map for the input feature
            halo = cosmos_data.halo
            # TODO REMOVE CFG AND HALO after debugging
            input_xy, _, biny_xy = input_feature.generate_map(self.cfg, halo, non_overlapping, bins, ranging, axis=(0, 1), muse_filter=self.simulate_muse)
            input_yz, _, biny_yz = input_feature.generate_map(self.cfg, halo, non_overlapping, bins, ranging, axis=(1, 2), muse_filter=self.simulate_muse)
            input_xz, _, biny_xz = input_feature.generate_map(self.cfg, halo, non_overlapping, bins, ranging, axis=(0, 2), muse_filter=self.simulate_muse)

            # Map for the output feature
            output_xy, _, _ = output_feature.generate_map(self.cfg, halo, subhalo_particles, bins, ranging, axis=(0, 1))
            output_yz, _, _ = output_feature.generate_map(self.cfg, halo, subhalo_particles, bins, ranging, axis=(1, 2))
            output_xz, _, _ = output_feature.generate_map(self.cfg, halo, subhalo_particles, bins, ranging, axis=(0, 2))

            maps = [
                (input_xy, output_xy, biny_xy),
                (input_yz, output_yz, biny_yz),
                (input_xz, output_xz, biny_xz)
            ]
            return maps
        elif self.dim == 3:
            ranging = ((-r, r), (-r, r), (-r, r))  # Defines the range for the 2D maps
            input, i_bin_edges, i_bin_number = input_feature.generate_map(subhalo_particles, bins, ranging, axis=None)
            output, o_bin_edges, o_bin_number = output_feature.generate_map(subhalo_particles,  bins, ranging,
                                                                            axis=None)
            return [(input, output, i_bin_number)]

    def _get_interlopers_id(self, cosmos_data: CosmosData) -> np.ndarray:
        """
        Identifies the interloper subhalos within the given data.

        Args:
            cosmos_data (CosmosData): Cosmos data containing subhalo information.

        Returns:
            np.ndarray: Array of subhalo IDs identified as interlopers.
        """

        # Considering as interloper any subhalo contained in a box defined by [-r, r] in the x and y directions and
        # inside [-R_200, R_200] along the z axis
        subhalos: SubhaloContext = cosmos_data.subhalos
        spos = np.array(subhalos.data['spos'].tolist())
        r = self.transition_radius
        mask = (
                (spos[:, 0] < r) &  # x condition
                (spos[:, 1] < r) &  # y condition
                (spos[:, 2] < cosmos_data.halo.r200)  # z condition
        )

        if 'group_number' in subhalos.data.columns:
            interlopers_id = subhalos.data[['subhalo_id', 'group_number']][mask]
        else:
            interlopers_id = subhalos.data['subhalo_id'][mask]

        # removing first galaxy (central/BCG), since it's usually the central galaxy or BCG
        # TODO is it possible to have no interlopers ?
        interlopers_id = interlopers_id.iloc[1:]

        if len(interlopers_id) == 0:
            pass
            # logger.warning(f'There is no interlopers for particles linked to {cosmos_data.subhalo_id}')
        return interlopers_id.to_numpy()

    def _get_interlopers_mask(self, cosmos_data: CosmosData, map_shape, interlopers_id: np.ndarray, biny: np.ndarray) -> np.ndarray:
        """
        Create a mask to identify and exclude interloper subhalos from calculations.

        Parameters:
        ----------
        cosmos_data : CosmosData
            The data containing subhalos and particle positions.

        map_shape : tuple
            The shape of the map to be created (height, width).

        interlopers_id : np.ndarray
            Array of identifiers for the interloper subhalos.

        biny : np.ndarray
            Array representing the y-coordinates for the masking grid.

        Returns:
        -------
        np.ndarray
            A boolean mask identifying interloper regions.
        """
        interlopers_mask = np.zeros(map_shape)
        r = self.transition_radius
        # Generate X and Y based on maschera's shape for consistency
        rows, cols = interlopers_mask.shape
        X, Y = np.meshgrid(
            np.linspace(-r, r, cols),  # X coordinates matching columns in maschera
            np.linspace(biny[0], biny[-1], rows)  # Y coordinates matching rows in maschera
        )

        try:
            for i in interlopers_id:
                # For each interloper get its center position (h, k) and its radius
                subhalo = None
                if isinstance(i, np.ndarray):
                    # [subhalo_id, halo_id]
                    subhalo = cosmos_data.subhalos.get_subhalo_by_id(subhalo_id=i[0], halo_id=i[1])
                elif isinstance(i, int):
                    subhalo = cosmos_data.subhalos.get_subhalo_by_id(subhalo_id=i, halo_id=None)
                assert subhalo is not None
                h = subhalo.spos[0]
                k = subhalo.spos[1]
                rradius = subhalo.shmr
                for cc in range(rows):
                    for dd in range(cols):
                        radius_circle = circle(X[cc, dd], Y[cc, dd], h, k, rradius)
                        if radius_circle <= 1:
                            interlopers_mask[cc, dd] = 1

            # Check if the mask is just zero
            if np.all(interlopers_mask == 0):
                logger.warning(f'Interlopers mask is full of zeros for halo: {cosmos_data.halo.halo_id}')
            interlopers_mask = interlopers_mask.astype(bool)
        except Exception as e:
            print(e)

        return interlopers_mask

    def _create_random_zoom(self, input_feature: FeatureProcessor, output_feature: FeatureProcessor,
                            interlopers_mask: np.ndarray, cosmos_data: CosmosData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates a random zoom of the masked labels and velocity dispersion maps.

        Parameters:
        ----------
        input_feature : FeatureProcessor
            The input feature processor.

        output_feature : FeatureProcessor
            The output feature processor.

        interlopers_mask : np.ndarray
            The mask identifying interloper regions.

        cosmos_data : CosmosData
            The data containing subhalos and particle positions.

        Returns:
        -------
        tuple
            A tuple containing:
            - input_feature_zoom : np.ndarray
                A smaller 2D array of the input feature.
            - output_feature_zoom : np.ndarray
                A smaller 2D array of the output feature.
            - mask_zoom : np.ndarray
                The zoomed-in mask.
        """
        bins = cosmos_data.metadata['map']['bins']
        xo, yo = (int(bins / 2), int(bins / 2))
        xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)

        # Ensure indices stay within bounds
        xoprime = np.clip(xoprime, int(bins / 4), input_feature.map.shape[0] - int(bins / 4))
        yoprime = np.clip(yoprime, int(bins / 4), input_feature.map.shape[1] - int(bins / 4))

        input_feature_zoom = input_feature.map[
                             xoprime - int(bins / 4):xoprime + int(bins / 4),
                             yoprime - int(bins / 4): yoprime + int(bins / 4)]

        output_feature_zoom = output_feature.map[
                              xoprime - int(bins / 4):xoprime + int(bins / 4),
                              yoprime - int(bins / 4): yoprime + int(bins / 4)]

        mask_zoom = interlopers_mask[
                    xoprime - int(bins / 4):xoprime + int(bins / 4),
                    yoprime - int(bins / 4):yoprime + int(bins / 4)]
        return input_feature_zoom, output_feature_zoom, mask_zoom

    def _create_n_zooms(self, projection_str: str, rotation_axis: str, rotation_iteration: int, cosmos_data: CosmosData,
                        input_feature: FeatureProcessor, output_feature: FeatureProcessor,
                        interlopers_mask: np.ndarray) -> None:
        """
        Creates multiple zoomed-in regions of the maps for a given projection.

        Args:
            projection_str (str): The name of the projection (e.g., "xy", "yz", "xz").
            cosmos_data (CosmosData): The cosmos data containing subhalo and particle information.
            input_feature (FeatureProcessor): The input feature processor.
            output_feature (FeatureProcessor): The output feature processor.
            interlopers_mask (np.ndarray): Mask of interloper regions.
        """
        # logger.info(f'Generating {self.map_configuration["number_zooms"]} zooms for projection {projection_str}...')

        def get_path(feature_name, halo_id_, subhalo_id_, projection_, rotation_axis_, rotation_iteration_, i_):
            if self.simulate_muse:
                if self.map_configuration['rotation_angle'] is not None:
                    path = os.path.join(self.map_configuration['root_path'], 'simulate_muse',
                                        f'rotation_{self.map_configuration["rotation_angle"]}_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                        feature_name,
                                        f'halo_{halo_id_}_subhalo_{subhalo_id_}_axis_{rotation_axis_}_'
                                        f'it_{rotation_iteration_}_{projection_}_zoom_{i_}')
                else:
                    path = os.path.join(self.map_configuration['root_path'], 'simulate_muse',
                                        f'no_rotation_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                        feature_name,
                                        f'halo_{halo_id_}_subhalo_{subhalo_id_}_{projection_}_zoom_{i_}')
            else:
                if self.map_configuration['rotation_angle'] is not None:
                    path = os.path.join(self.map_configuration['root_path'],
                                        f'rotation_{self.map_configuration["rotation_angle"]}_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                        feature_name,
                                        f'halo_{halo_id_}_subhalo_{subhalo_id_}_axis_{rotation_axis_}_'
                                        f'it_{rotation_iteration_}_{projection_}_zoom_{i_}')
                else:
                    path = os.path.join(self.map_configuration['root_path'],
                                        f'no_rotation_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                        feature_name,
                                        f'halo_{halo_id_}_subhalo_{subhalo_id_}_{projection_}_zoom_{i_}')
            return path

        setup_output_directories(self.map_configuration['root_path'],
                                 self.map_configuration['rotation_angle'],
                                 self.map_configuration['bin_spatial_resolution_scale'],
                                 input_feature.name,
                                 output_feature.name)

        for i in range(self.map_configuration['number_zooms']):
            input_feature_i, output_feature_i, mask_i = self._create_random_zoom(input_feature, output_feature,
                                                                                 interlopers_mask, cosmos_data)
            # Expand the dimensions of the mask to match the shape of the other arrays
            mask_expanded = np.expand_dims(mask_i, axis=-1)  # Shape: (H, W, 1)

            # Expand the dimensions of vel_disp_i and label_i to make them 3D
            input_featute_i_expanded = np.expand_dims(input_feature_i, axis=-1)  # Shape: (H, W, 1)
            output_feature_i_expanded = np.expand_dims(output_feature_i, axis=-1)  # Shape: (H, W, 1)

            # Concatenate the mask as an additional channel to the zoomed arrays
            input_feature_with_mask = np.concatenate((input_featute_i_expanded, mask_expanded), axis=-1)  # Shape: (H, W, 2)
            output_feature_with_mask = np.concatenate((output_feature_i_expanded, mask_expanded), axis=-1)  # Shape: (H, W, 2)

            # Add rotation axis
            input_path = get_path(input_feature.name, cosmos_data.halo.halo_id, cosmos_data.subhalo_id,
                                  projection_str, rotation_axis, rotation_iteration, i)
            output_path = get_path(output_feature.name, cosmos_data.halo.halo_id, cosmos_data.subhalo_id,
                                   projection_str, rotation_axis, rotation_iteration, i)
            logger.debug(f'input_path: {input_path}')
            logger.debug(f'output_path: {output_path}')

            np.savez(input_path, input_feature_with_mask)
            np.savez(output_path, output_feature_with_mask)

    def _process_projection(self, feature_size, interlopers_id: np.ndarray, biny_i, projection: str, rotation_axis: str,
                            rotation_iteration: int,  cosmos_data: CosmosData, input_feature: FeatureProcessor,
                            output_feature: FeatureProcessor):
        """
        Processes each projection (XY, YZ, XZ) for the given data, generating maps and saving zoomed-in versions.

        Args:
            cosmos_data (CosmosData): The data to be used for generating projections.
            input_feature (FeatureProcessor): The processor for input features.
            output_feature (FeatureProcessor): The processor for output features.
        """
        interlopers_mask = self._get_interlopers_mask(cosmos_data, feature_size, interlopers_id, biny_i)
        self._create_n_zooms(projection, rotation_axis, rotation_iteration, cosmos_data, input_feature, output_feature,
                             interlopers_mask)

    def _is_halo_already_processed(self, feature, halo, cosmos_data):
        feature_name = feature.name
        file = f'halo_{halo.halo_id}_subhalo_{cosmos_data.subhalo_id}'

        if self.simulate_muse:
            if self.map_configuration['rotation_angle'] is not None:
                path = os.path.join(self.map_configuration['root_path'], 'simulate_muse',
                                    f'rotation_{self.map_configuration["rotation_angle"]}_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                    feature_name)
            else:
                path = os.path.join(self.map_configuration['root_path'], 'simulate_muse',
                                    f'no_rotation_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                    feature_name)
        else:
            if self.map_configuration['rotation_angle'] is not None:
                path = os.path.join(self.map_configuration['root_path'],
                                    f'rotation_{self.map_configuration["rotation_angle"]}_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                    feature_name)
            else:
                path = os.path.join(self.map_configuration['root_path'],
                                    f'rotation_{self.map_configuration["rotation_angle"]}_s{self.map_configuration["bin_spatial_resolution_scale"]}',
                                    feature_name)

        pattern = re.compile(re.escape(file) + ".*")  # escape special chars in string
        print(pattern)
        if not os.path.exists(path):
            return False

        for filename in os.listdir(path):
            if pattern.match(filename):  # match at the start of the filename
                return True
        return False

    def process(self, g: Generator, next_step: Callable[[Generator], Generator], **kwargs):
        """
        Main method to process projections for the given cosmos data, generating maps and processing them.

        This method generates projection maps ("xy", "yz", "xz") for the cosmos data and processes them
        accordingly. It updates the input and output feature processors with the generated maps, and applies
        specific transformations based on the map projections.

        Parameters:
        -----------
        g : Generator
            The generator that yields the data to be processed.
        next_step : Callable[[Generator], Generator]
            A lambda expression representing the next step in the pipeline, which processes the data
            after the map generation step. The lambda receives a generator and returns another generator.

        Yields:
        -------
        tuple
            A tuple containing the updated input_feature, output_feature, and cosmos_data after map generation and processing.

        Notes:
        ------
        This method generates three different map projections, processes them, and updates the features and metadata accordingly.
        """
        rotation_axis = kwargs.get('rotation_axis', None)
        rotation_iteration = kwargs.get('rotation_iteration', None)
        errors = []
        for halo, _, cosmos_data in g:
            logger.debug(f'Halo: {halo.halo_id}, {cosmos_data.halo.halo_id}')
            try:
                # Check if this halo has been processed already
                if self.cfg.skip_existing_halos:
                    if self._is_halo_already_processed(cosmos_data.input_feature, halo, cosmos_data):
                        logger.debug(f'Halo {halo.halo_id} has been already processed, skipping it.')
                        continue

                if self.transition_radius is None:
                    raise ValueError('Transition radius cannot be None.')

                # Compute number of bins and store it in the metadata
                fov = self.transition_radius * 2
                bins = muse_bins_r(fov, self.spatial_resolution, self.bin_spatial_resolution_scale)

                # Store num_bins
                cosmos_data.metadata['map']['bins'] = bins
                input_feature = cosmos_data.input_feature
                output_feature = cosmos_data.output_feature
                logger.debug(f'Number of bins: {bins}')

                interlopers_id = self._get_interlopers_id(cosmos_data)
                io_biny_projections = self._create_maps(cosmos_data, input_feature, output_feature)

                if self.dim == 2:
                    projections = ["xy", "yz", "xz"]
                    for j, (map_f_i, map_f_o, biny_i) in enumerate(io_biny_projections):
                        projection_name = projections[j]
                        input_feature.map = map_f_i
                        output_feature.map = map_f_o
                        self._process_projection(map_f_o.shape, interlopers_id, biny_i, projection_name, rotation_axis,
                                                 rotation_iteration, cosmos_data, input_feature, output_feature)
                elif self.dim == 3:
                    pass

                # Got to the next step
                if next_step is not None:
                    yield from next_step([(halo, _, cosmos_data)], **kwargs)
                else:
                    yield cosmos_data.halo, cosmos_data.input_feature, cosmos_data.output_feature

            except Exception as e:
                logger.error(f"Failed to create maps for halo {cosmos_data.halo.halo_id}: {e}")
                errors.append((halo.halo_id, str(e), self.name))
                continue
        yield errors


