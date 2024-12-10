import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import joblib
from omegaconf import DictConfig
from scipy import stats
from dotenv import load_dotenv, find_dotenv

from src.data_generation.src.halo import Halo
from src.data_generation.src.utils import setup_logger

logger = setup_logger("FeatureProcessor")

dotenv_path = find_dotenv('../.env')
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print(".env file not found, skipping load.")


class FeatureProcessor(ABC):
    """
    Abstract base class for feature processors.

    A feature processor is responsible for processing specific features
    from input data, typically in the form of particles or halo information,
    and generating the corresponding output.

    Subclasses must implement the `process` method.
    """
    @abstractmethod
    def process(self, **kwargs) -> np.ndarray:
        """
        Process the feature data.

        Args:
            **kwargs: Keyword arguments, expected to include the necessary data for processing.

        Returns:
            np.ndarray: A numpy array containing the processed feature data.
        """
        pass

    @abstractmethod
    def generate_map(self, particle_positions: np.ndarray, bins: int, ranging: tuple, axis: tuple) -> tuple:
        """
        Generate a 2D map for the feature based on particle positions.

        Args:
            particle_positions (np.ndarray): Positions of particles.
            bins (int): Number of bins for the map.
            ranging (tuple): Range for the 2D map.
            axis (tuple): Axes to project (e.g., (0, 1) for x-y plane).

        Returns:
            tuple: A tuple containing the map and additional data like bin edges.
        """
        pass

    @staticmethod
    def _build_configuration(cfg: DictConfig, required_keys: list) -> dict:
        if 'data_loader' not in [e['name'] for e in cfg.pipeline]:
            raise ValueError('Error, To execute a data_loader you must specify the map step in the pipeline of'
                             ' the configuration.')

        data_loader_dict = next((step for step in cfg.pipeline if step.name == 'data_loader'), None)
        for key in required_keys:
            if key not in data_loader_dict:
                raise ValueError(f"Error, missing required configuration key: {key}")
        subset_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
        return subset_dict

    @classmethod
    def get_empty(cls) -> np.ndarray:
        """
        Get an empty or default value for the feature.

        Returns:
            np.ndarray: An empty or default value for the feature.
        """
        pass


class VelocityDispersionProcessor(FeatureProcessor):
    """
    Class to process the velocity dispersion feature for particles.

    This processor computes the velocity dispersion based on the input particle velocities.
    It assumes that the velocity dispersion is directly available in the data.

    Attributes:
        name (str): The name of the feature ('VelocityDispersion').
        map ([]): Velocity Dispersion map.
        value ([]): Placeholder for storing computed value.
    """
    def __init__(self):
        self.name = 'VelocityDispersion'
        self.map = []
        self.value = []

    # TODO review this computation
    def process(self, **kwargs) -> np.ndarray:
        """
        Calculate the velocity dispersion of particles.

        Args:
            **kwargs: Keyword arguments, expected to include the necessary data for processing, in this case
            the velocities of the particles.

        Returns:
            np.ndarray: The velocity for each particle.

        Raises:
            ValueError: If the input particle_velocity is empty.
        """
        if 'particle_velocity' not in kwargs:
            logger.error('VelocityDispersion needs particle_velocity as input parameter.')
            raise Exception('VelocityDispersion needs particle_velocity as input parameter.')

        particle_velocity = kwargs['particle_velocity']

        if particle_velocity.size == 0:
            logger.error("SubhaloParticles cannot have 'v' (velocity) empty.")
            raise ValueError("SubhaloParticles cannot have 'v' (velocity) empty.")

        logger.info(f'Computed velocity dispersion for {len(particle_velocity)} particles.')
        return particle_velocity

    def generate_map(self, particle_positions: np.ndarray, bins: int, ranging: tuple, axis: tuple) -> tuple:
        """
        Generate a 2D map of processed particle features using binned statistics.

        This method calculates a 2D histogram-like map of a feature derived from particle data.
        It uses the positions of particles and bins them into a grid, computing the mean of a
        specific feature for each grid cell. The resulting feature values are processed further
        to compute their normalized root mean square (RMS).

        Args:
            particle_positions (np.ndarray): A 2D array of shape (N, 3) representing the positions
                of particles in 3D space, where N is the number of particles.
            bins (int): The number of bins to use along each axis for the grid.
            ranging (tuple): A tuple of two tuples defining the range for binning along each axis,
                e.g., ((xmin, xmax), (ymin, ymax)).
            axis (tuple): A tuple of two integers indicating which axes to use for the 2D grid
                (e.g., (0, 1) for x and y).

        Returns:
            tuple: A tuple containing:
                - input_ij (np.ndarray): A 2D array representing the grid of processed feature values.
                - binx (np.ndarray): The bin edges along the first axis.
                - biny (np.ndarray): The bin edges along the second axis.

        Raises:
            Exception: If the `self.value` attribute is empty, indicating that the `process`
                method needs to be called before generating the map.
        """

        i, j = axis
        if len(self.value) == 0:
            logger.error('FeatureProcessor value is empty, call first process method.')
            raise Exception('FeatureProcessor value is empty, call first process method.')

        if isinstance(self.value, List):
            self.value = np.array(self.value)

        components = {0, 1, 2}
        missing_component = components - set(axis)
        k = next(iter(missing_component))

        pos_rot = particle_positions
        input_ij, binx, biny, _ = stats.binned_statistic_2d(
            pos_rot[:, i], pos_rot[:, j],
            (self.value[:, k]) ** 2,  # Access the relevant feature
            statistic="mean", bins=bins, range=ranging
        )
        input_ij = np.sqrt(input_ij / 3)
        input_ij[np.isnan(input_ij)] = 0
        return input_ij, binx, biny

    @classmethod
    def get_empty(cls) -> np.ndarray:
        """
        Return an empty value for velocity dispersion (an empty array).

        Returns:
            np.ndarray: Empty array for velocity dispersion.
        """
        return np.array([])


class ICLProcessor(FeatureProcessor):
    """
    Class to process the IntraCluster Light (ICL) feature for particles in a halo.

    This processor calculates a mask for particles belonging to the ICL (Inter-cluster Light)
    using a trained machine learning model. It computes the necessary features, applies
    the model to predict ICL/BCG (Brightest Cluster Galaxy), and returns a mask.

    Attributes:
        name (str): The name of the feature ('IntraClusterLight').
        model_path (str): Path to the trained model that classifies particle data into BCG or ICL.
        cfg (DictConfig): Configuration object for retrieving constant G.
        value (None): Placeholder for storing computed value.
    """
    def __init__(self, cfg: DictConfig):
        self.name = 'IntraClusterLight'

        project_root = os.getenv('PROJECT_ROOT')
        if not project_root:
            raise EnvironmentError("PROJECT_ROOT is not set in the environment variables.")
        self.model_path = os.path.join(project_root, 'src/data_generation/model/Best_Model-Random-3par.pkl')
        self.cfg = cfg
        self.map = []
        self.value = []

        attr_list = self._build_configuration(cfg, ['G'])
        for key, value in attr_list.items():
            setattr(self, key, value)

    def _predict(self, v_delta: np.ndarray, distance: np.ndarray, halo: Halo):
        """
        Predict whether particles belong to the IntraCluster Light (ICL) or the Brightest Cluster Galaxy (BCG).

        This method uses a pre-trained machine learning model to classify particles based on their velocity
        and distance from the center of the halo. It calculates features such as the scaled velocity,
        scaled distance, and log-scaled halo mass, which are then passed to the model for classification.

        Args:
            v_delta (np.ndarray): The velocity difference (in 3D space) between each particle and the halo's
                                   center-of-mass velocity.
            distance (np.ndarray): The Euclidean distance of each particle from the center of the halo.
            halo (Halo): The halo object containing parameters like `m200` and `r200`, which are used to compute
                         scaling factors for velocity and distance.

        Returns:
            np.ndarray: An array of predictions where each value indicates the predicted class for a particle:
                        0 for ICL (IntraCluster Light) and 1 for BCG (Brightest Cluster Galaxy).

        Raises:
            ValueError: If the scaled particle data is empty or contains only missing values, the method raises
                        an exception indicating that the prediction cannot be made.
        """
        final_model = joblib.load(self.model_path)
        # Define simulation parameters
        V200 = np.sqrt(self.G * halo.m200 / halo.r200)

        # Create a DataFrame with features required by the model
        # Features are: log-scaled distance (r/R200), scaled velocity (modulo_v/V200), and log-scaled halo mass (M200)
        Data = pd.DataFrame({
            'distance': np.log10(distance / halo.r200),  # Log-scaled distance
            'velocity': v_delta / V200,  # Scaled velocity
            'm200': [np.log10(halo.m200 * 1e-14)] * len(v_delta)
            # Log-scaled halo mass (m200, constant for all particles)
        })

        # Replace any infinite values in the DataFrame with NaN, then drop all rows with NaN values
        Data = Data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        if Data.size == 0:
            logger.error("Scaled particle data is empty, it's likely that your particle data has a "
                         "considerable number of missing values")
            raise ValueError("Scaled particle data is empty, it's likely that your particle data has a "
                             "considerable number of missing values")

        # Use the trained model to make predictions on the data
        # The model predicts whether each particle belongs to BCG (label 1) or ICL (label 0)
        logger.info("Labeling particles ...")
        final_predictions = final_model.predict(Data.to_numpy())
        return final_predictions

    # Subhalo particles, halo
    def process(self, **kwargs) -> np.ndarray:
        """
        Process the IntraCluster Light (ICL) for a given halo and its particles.

        Args:
            halo (Halo): The halo object containing halo parameters.
            particle_velocity (np.ndarray): The velocity of each particle.
            particle_position (np.ndarray): The position of each particle.

        Returns:
            np.ndarray: A mask (1 or 0) indicating whether each particle belongs to the ICL (1) or BCG (0).

        Raises:
            ValueError: If the scaled particle data is empty.
        """
        if not {'halo', 'particle_velocity', 'particle_position'}.issubset(kwargs):
            logger.error("Missing required keys: 'halo', 'particle_velocity', or 'particle_position' in kwargs")
            raise Exception("Missing required keys: 'halo', 'particle_velocity', or 'particle_position' in kwargs")

        halo = kwargs['halo']
        particle_velocity = kwargs['particle_velocity']
        particle_position = kwargs['particle_position']

        logger.info(f'Processing ICL for Halo ID {halo.halo_id}.')

        v_cm = np.average(particle_velocity, axis=0)  # calculate the center-of-mass velocity (avg velocity)
        # Computing the Euclidean distance from the new origin to each particle
        distance = np.sqrt((particle_position ** 2).sum(axis=1))
        # Computing the delta vel. from each particle in comparison to the mean-vel.
        v_delta = np.sqrt(((particle_velocity - v_cm) ** 2).sum(axis=1))
        final_predictions = self._predict(v_delta, distance, halo)
        # Extract indices of particles predicted to be part of the ICL
        ids = np.where(final_predictions == 0)[0]
        rows, _ = particle_velocity.shape
        icl_mask = np.zeros((rows, ))
        if ids.size > 0:
            # if particle belongs to ICL=1, otherwise BCG=0
            icl_mask[ids] = 1
        else:
            logger.warning('No particles predicted as ICL.')
            raise ValueError("icl_mask is empty; no masking applied.")

        logger.info(f'Generated ICL mask for {len(icl_mask)} particles.')
        return icl_mask

    def generate_map(self, particle_positions: np.ndarray, bins: int, ranging: tuple, axis: tuple) -> tuple:
        """
        Generate a 2D map of particle feature data using binned statistics.

        This method computes a 2D map by summing particle features within a grid defined
        by specified bins and axis ranges. The summed feature values are normalized by
        the count of particles in each bin to provide an average value for each grid cell.

        Args:
            particle_positions (np.ndarray): A 2D array of shape (N, 3) representing the positions
                of particles in 3D space, where N is the number of particles.
            bins (int): The number of bins to use along each axis for the grid.
            ranging (tuple): A tuple of two tuples defining the range for binning along each axis,
                e.g., ((xmin, xmax), (ymin, ymax)).
            axis (tuple): A tuple of two integers indicating which axes to use for the 2D grid
                (e.g., (0, 1) for x and y).

        Returns:
            tuple: A tuple containing:
                - output_ij (np.ndarray): A 2D array representing the grid of averaged feature values.
                - binx (np.ndarray): The bin edges along the first axis.
                - biny (np.ndarray): The bin edges along the second axis.

        Raises:
            Exception: If the `self.value` attribute is empty, indicating that the `process`
                method needs to be called before generating the map.
            AssertionError: If the size of the input data along the specified axes and the length
                of `self.value` do not match.
        """
        i, j = axis
        if len(self.value) == 0:
            logger.error('FeatureProcessor value is empty, call first process method.')
            raise Exception('FeatureProcessor value is empty, call first process method.')
        pos_rot = particle_positions

        assert pos_rot[:, i].size == pos_rot[:, j].size == len(self.value)
        output_ij, binx, biny, _ = stats.binned_statistic_2d(
            pos_rot[:, i], pos_rot[:, j],
            self.value,  # Access the ICL feature data
            statistic="mean", bins=bins, range=ranging
        )
        output_ij[np.isnan(output_ij)] = 0
        return output_ij, binx, biny

    @classmethod
    def get_empty(cls) -> np.ndarray:
        """
        Return an empty value for ICL (an empty array).

        Returns:
            np.ndarray: Empty array for ICL.
        """
        return np.array([])


class FeatureProcessorFactory:
    """
    Factory class to generate specific feature processors.

    This factory class provides an easy way to instantiate and return the correct
    feature processor based on the feature type passed.

    Methods:
        get_feature: Returns an instance of the required FeatureProcessor.
    """
    @staticmethod
    def get_feature(feature: str, **kwargs) -> FeatureProcessor:
        """
        Get an instance of the appropriate FeatureProcessor based on the feature name.

        Args:
            feature (str): The feature type ('velocity_dispersion' or 'icl').
            **kwargs: Additional keyword arguments passed to the feature processor.

        Returns:
            FeatureProcessor: An instance of the appropriate feature processor.

        Raises:
            ValueError: If an invalid feature name is provided.
        """
        # Add more feature processors as needed
        feature_processors = {
            'velocity_dispersion': VelocityDispersionProcessor(),
            'icl': ICLProcessor(cfg=kwargs.get('cfg', None)),
            # Add more features if required
        }

        if feature not in feature_processors:
            raise ValueError(f"Invalid feature name {feature} provided")
        logger.info(f"Feature processor '{feature}' created.")
        return feature_processors[feature]
