import cProfile
import pstats
from datetime import datetime
from typing import Generator, Callable, Tuple
from omegaconf import DictConfig

import numpy as np

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.utils import rotate_vector, get_transform_matrix, rotate_3dvec, setup_logger

logger = setup_logger("RotateStep")


class RotateStep(PipelineStep):
    """
    A pipeline step for rotating data in the cosmos dataset.

    This class takes a rotation angle and a list of rotation axes, and applies
    rotations to the positions and velocities of the objects in the cosmos data.

    Attributes:
    -----------
    rotation_angle : float
        The angle by which the objects are rotated during each step.
    z : np.ndarray
        The original z-axis vector (should be a standard reference vector).
    rotation_directions : np.ndarray
        A set of axes along which the rotation is applied.
    """
    name = 'rotate_step'

    def __init__(self, cfg: DictConfig):
        """
        Initializes the RotateStep object.

        Parameters:
        -----------
        rotation_angle : float
            The angle in degrees by which the objects will be rotated.
        z : np.ndarray
            A 3D vector representing the original z-axis vector.
        rotation_directions : np.ndarray
            A set of rotation axes for the objects.
        """
        self.rotation_angle, self.z, self.rotation_directions = self._build_configuration(cfg)
        self.path = cfg.paths.datasets_dir + f'/rotate_step_{datetime}.txt'


    @classmethod
    def _build_configuration(cls, cfg: DictConfig) -> Tuple:
        """
        Builds the configuration for the mapping step.

        Args:
            cfg (DictConfig): Configuration object.

        Returns:
            Tuple: A tuple containing rotate step configuration.
        """

        if cls.name not in [e['name'] for e in cfg.pipeline]:
            raise ValueError('Error, To execute a rotation step you must specify the rotation step in the pipeline of'
                             ' the configuration.')

        rotate_step_dict = next((step for step in cfg.pipeline if step.name == cls.name), None)

        if rotate_step_dict is None:
            raise ValueError('The step rotate_step could not be found in the pipeline, make sure it is properly defined'
                             'in the pipeline of the configuration file.')

        return rotate_step_dict.rotation_group

    def _rotate(self, axis, degree, X):
        """
        Helper function to rotate positions or velocities based on the provided axis and degree.

        Parameters:
        -----------
        axis : np.ndarray
            The axis around which the rotation is performed.
        degree : float
            The angle in degrees for the rotation.
        X : np.ndarray
            The array of positions or velocities to be rotated.

        Returns:
        --------
        np.ndarray
            The rotated array.
        """
        new_z = rotate_vector(v=self.z, axis=axis, angle_=degree)
        Q = get_transform_matrix(new_z, self.z, normalized=False)
        X_rot = np.float32(rotate_3dvec(Q, X))
        return X_rot

    def process(self, input_feature: FeatureProcessor, output_feature: FeatureProcessor,
                cosmos_data: CosmosData, g: Generator, next_step: Callable[[Generator], Generator]):
        """
        Rotates the data in the cosmos dataset and updates the features accordingly.

        This method applies rotations to the positions and velocities of the objects in the
        cosmos dataset and also updates the metadata to indicate the rotation state.

        Parameters:
        -----------
        input_feature : FeatureProcessor
            The feature processor object that holds the features to be rotated.
        output_feature : FeatureProcessor
            The feature processor object that holds the features to be updated after rotation.
        cosmos_data : CosmosData
            The data object that contains the positions, velocities, and metadata of the cosmos data.
        g : Generator
            The generator that yields data for processing, likely from a pipeline or data stream.
        next_step : Callable[[Generator], Generator]
            A lambda expression representing the next step in the pipeline that will process the data
            after this step completes. The lambda receives a generator and returns another generator.

        Yields:
        -------
        tuple
            A tuple containing the updated input_feature, output_feature, and cosmos_data after the rotation is applied.

        Notes:
        ------
        The method applies a series of rotations in specified directions and angles, updating the positions and velocities
        in the cosmos data, as well as updating the rotation-related metadata.
        """
        profiler = cProfile.Profile()
        profiler.enable()
        logger.info('Starting rotations...')
        for _ in g:
            try:
                iterations = 360 // int(self.rotation_angle)
                for rotation_axis in self.rotation_directions:
                    degree = 0
                    for i in range(iterations):
                        rotation_axis = np.array(rotation_axis)

                        # Update cosmos data, subhalos data and subhalo particle positions
                        spos = np.array(cosmos_data.subhalos['spos'].tolist())
                        subhalos_spos = self._rotate(rotation_axis, degree, spos)
                        cosmos_data.subhalos['spos'] = list([spos[i] for i in range(subhalos_spos.shape[0])])

                        cosmos_data.subhalo_particles_positions = (
                            self._rotate(rotation_axis, degree, cosmos_data.subhalo_particle_positions))

                        input_feature.value = self._rotate(rotation_axis, degree, input_feature.value)
                        cosmos_data.metadata['rotation'] = {'has_rotated': 1, 'angle': degree}

                        if next_step is not None:
                            yield from next_step(g)
                        else:
                            yield input_feature, output_feature, cosmos_data
                        degree += self.rotation_angle
            except Exception as e:
                logger.error(f"Failed to execute rotations for halo {cosmos_data.halo.halo_id}: {e}")
                continue
        profiler.disable()
        with open(self.path, "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.strip_dirs()
            stats.sort_stats("cumulative")
            stats.print_stats()
