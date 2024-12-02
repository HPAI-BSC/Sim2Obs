import numpy as np

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.utils import rotate_vector, get_transform_matrix, rotate_3dvec


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
    def __init__(self, rotation_angle: float, z: np.ndarray, rotation_directions: np.ndarray):
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
        self.rotation_angle = rotation_angle
        self.z = z
        self.rotation_directions = rotation_directions

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

    def process(self, input_feature: FeatureProcessor, cosmos_data: CosmosData):
        """
        Rotates the data in the cosmos dataset and updates the features accordingly.

        This method applies rotations to the positions and velocities of the objects in the
        cosmos dataset and also updates the metadata to indicate the rotation state.

        Parameters:
        -----------
        input_feature : FeatureProcessor
            The feature processor object that holds the features to be rotated.
        cosmos_data : CosmosData
            The data object that contains the positions, velocities, and metadata of the cosmos data.

        Yields:
        -------
        tuple
            A tuple containing the updated cosmos_data and input_feature after the rotation is applied.
        """
        iterations = 360 // int(self.rotation_angle)
        # log.debug(f"Total iterations: {iterations} with rotation angle: {self.rotation_angle} degrees.")
        for rotation_axis in self.rotation_directions:
            degree = 0
            for i in range(iterations):
                # log.debug(f"Rotation iteration {i + 1}/{iterations}: rotating by {degree} degrees.")
                rotation_axis = np.array(rotation_axis)

                # Update cosmos data, subhalos data and subhalo particle positions
                spos = np.array(cosmos_data.subhalos.data['spos'].tolist())
                subhalos_spos = self._rotate(rotation_axis, degree, spos)
                cosmos_data.subhalos.data['spos'] = list([spos[i] for i in range(subhalos_spos.shape[0])])

                cosmos_data.subhalo_particles_positions = (
                    self._rotate(rotation_axis, degree, cosmos_data.subhalo_particle_positions))

                input_feature.value = self._rotate(rotation_axis, degree, input_feature.value)
                # log.debug(f"Rotation metadata updated: {cosmos_data.metadata['rotation']}")
                cosmos_data.metadata['rotation'] = {'has_rotated': 1, 'angle': degree}
                degree += self.rotation_angle
                yield cosmos_data, input_feature
            # log.info("Rotation process completed.")
