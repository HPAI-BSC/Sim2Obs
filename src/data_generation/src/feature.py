from abc import ABC, abstractmethod
import rootutils

import numpy as np
import pandas as pd
import joblib
from omegaconf import DictConfig


from src.data_generation.src.halo import Halo
from src import utils


rootutils.setup_root(__file__, pythonpath=True)
log = utils.get_pylogger(__name__)


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


class VelocityDispersionProcessor(FeatureProcessor):
    """
    Class to process the velocity dispersion feature for particles.

    This processor computes the velocity dispersion based on the input particle velocities.
    It assumes that the velocity dispersion is directly available in the data.

    Attributes:
        name (str): The name of the feature ('VelocityDispersion').
        value (None): Placeholder for storing computed value.
    """
    def __init__(self):
        self.name = 'VelocityDispersion'
        self.value = None

    # TODO review this computation
    def process(self, particle_velocity: np.ndarray) -> np.ndarray:
        """
        Calculate the velocity dispersion of particles.

        Args:
            particle_velocity (np.ndarray): The velocities of the particles.

        Returns:
            np.ndarray: The velocity dispersion for each particle.

        Raises:
            ValueError: If the input particle_velocity is empty.
        """

        if particle_velocity.size == 0:
            raise ValueError("SubhaloParticles cannot have 'v' (velocity) empty.")

        v_cm = np.average(particle_velocity, axis=0)  # calculate the center-of-mass velocity (avg velocity)
        # Computing the delta vel. from each particle in comparison to the mean-vel.
        velocity_disp = np.sqrt(((particle_velocity - v_cm) ** 2).sum(axis=1))
        log.info(f'Computed velocity dispersion for {len(particle_velocity)} particles.')
        return velocity_disp


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
        self.model_path = '../model/Best_Model-Random-3par.pkl'
        self.cfg = cfg
        self.value = None

    # Subhalo particles, halo
    def process(self, halo: Halo, particle_velocity: np.ndarray, particle_position: np.ndarray) -> np.ndarray:
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
        log.info(f'Processing ICL for Halo ID {halo.halo_id}.')
        v_cm = np.average(particle_velocity, axis=0)  # calculate the center-of-mass velocity (avg velocity)
        # Computing the Euclidean distance from the new origin to each particle
        distance = np.sqrt((particle_position ** 2).sum(axis=1))
        # Computing the delta vel. from each particle in comparison to the mean-vel.
        v_delta = np.sqrt(((particle_velocity - v_cm) ** 2).sum(axis=1))

        final_model = joblib.load(self.model_path)
        # Define simulation parameters
        V200 = np.sqrt(self.cfg.G * halo.m200 / halo.r200)

        # Create a DataFrame with features required by the model
        # Features are: log-scaled distance (r/R200), scaled velocity (modulo_v/V200), and log-scaled halo mass (M200)
        Data = pd.DataFrame({
            'distance': np.log10(distance / halo.r200),  # Log-scaled distance
            'velocity': v_delta / V200,  # Scaled velocity
            'm200': [np.log10(halo.m200 * 1e-14)] * len(v_delta)  # Log-scaled halo mass (m200, constant for all particles)
        })

        # Replace any infinite values in the DataFrame with NaN, then drop all rows with NaN values
        Data = Data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        if Data.size == 0:
            raise ValueError("Scaled particle data is empty, it's likely that your particle data has a "
                             "considerable number of missing values")

        # Use the trained model to make predictions on the data
        # The model predicts whether each particle belongs to BCG (label 1) or ICL (label 0)
        # log.info("Labeling particles ...")
        final_predictions = final_model.predict(Data.to_numpy())
        # Extract indices of particles predicted to be part of the ICL
        ids = np.where(final_predictions == 0)[0]
        rows, _ = particle_velocity.shape
        icl_mask = np.zeros((rows, 1))
        if ids.size > 0:
            # if particle belongs to ICL=1, otherwise BCG=0
            icl_mask[ids] = 1
        else:
            log.warning("No particles predicted as ICL.")
            raise ValueError("icl_mask is empty; no masking applied.")
        log.info(f'Generated ICL mask for {len(icl_mask)} particles.')
        return icl_mask


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
        log.info(f"Feature processor '{feature}' created.")
        return feature_processors[feature]