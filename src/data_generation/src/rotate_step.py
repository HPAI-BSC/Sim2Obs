from typing import Generator, Callable, Tuple
from omegaconf import DictConfig

import numpy as np

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.utils import setup_logger

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

    def process(self, g: Generator, next_step: Callable[[Generator], Generator], **kwargs):
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
        errors = []
        for halo, subhalo_ctx, cosmos_data in g:
            logger.info(f"Processing rotations for halo {halo.halo_id} ...")
            try:
                logger.info(f"Generating maps for halo {halo.halo_id} without rotation")
                #yield from next_step([(halo, subhalo_ctx, cosmos_data)], **kwargs)

                logger.info(f"Starting rotations for halo {halo.halo_id} ...")
                iterations = 360 // int(self.rotation_angle)
                for rt_i, rotation_axis in enumerate(self.rotation_directions):
                    logger.info(f"Starting rotation axis {rt_i} of {len(self.rotation_directions) - 1} for halo {halo.halo_id} ...")
                    kwargs['rotation_axis'] = rt_i
                    degree = int(self.rotation_angle)
                    for i in range(iterations):
                        rotation_axis = np.array(rotation_axis)
                        kwargs['rotation_iteration'] = i

                        logger.info(f"Iteration {i} of {iterations} for halo {halo.halo_id} in rotation axis {rt_i} with angle {degree} degrees")

                        # Rotate the appropiate data in subhalos context, subhalo particles and input feature
                        cosmos_data.subhalos.rotate(rotation_axis, degree)
                        cosmos_data.subhalo_particles.rotate(rotation_axis, degree)
                        cosmos_data.input_feature.rotate(rotation_axis, degree)
                        cosmos_data.metadata['rotation'] = {'has_rotated': 1, 'angle': degree}

                        logger.info(f"Out - Iteration {i} of {iterations} for halo {halo.halo_id} in rotation axis {rt_i} with angle {degree} degrees")

                        # Got to the next step
                        if next_step is not None:
                            logger.info(f"Generating maps for halo {halo.halo_id}, iteration={i}/{iterations - 1},"
                                        f" rotation_axis={rt_i}/{len(self.rotation_directions) - 1}")

                            yield from next_step([(halo, subhalo_ctx, cosmos_data)], **kwargs)
                        else:
                            yield cosmos_data.halo, cosmos_data.input_feature, cosmos_data.output_feature

                        #degree += self.rotation_angle

            except Exception as e:
                logger.error(f"Failed to execute rotations for halo {cosmos_data.halo.halo_id}: {e}")
                errors.append((halo, str(e), self.name))
                continue
        yield errors



