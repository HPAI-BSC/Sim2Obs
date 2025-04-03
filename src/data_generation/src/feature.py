import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt, ticker
from omegaconf import DictConfig
from scipy import stats
from dotenv import load_dotenv, find_dotenv

from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.utils import setup_logger, rotate

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
    def process(self, halo: Halo, subhalo_particles: SubhaloParticles) -> np.ndarray:
        """
        Process the feature data.

        Args:
            **kwargs: Keyword arguments, expected to include the necessary data for processing.

        Returns:
            np.ndarray: A numpy array containing the processed feature data.
        """
        pass
    def rotate(self, z, axis, degree):
        if len(self.value) == 0:
            logger.error('FeatureProcessor value is empty, call first process method.')
            raise Exception('FeatureProcessor value is empty, call first process method.')
        self.value = rotate(z, axis, degree, self.value)

    # TODO update
    @abstractmethod
    def generate_map(self, subhalo_particles: SubhaloParticles, bins: int, ranging: tuple, axis: tuple) -> tuple:
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

    def process(self, halo: Halo, subhalo_particles: SubhaloParticles) -> np.ndarray:
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
        if not hasattr(subhalo_particles, 'star_v'):
            logger.error('VelocityDispersion needs star_v as input parameter.')
            raise Exception('VelocityDispersion needs star_v as input parameter.')

        if subhalo_particles.star_v.size == 0:
            logger.error("SubhaloParticles cannot have 'v' (velocity) empty.")
            raise ValueError("SubhaloParticles cannot have 'v' (velocity) empty.")

        logger.info(f'Computed velocity dispersion for {len(subhalo_particles.star_v)} particles.')
        return subhalo_particles.star_v

    def generate_map(self, subhalo_particles: SubhaloParticles, bins: int, ranging: tuple, axis: tuple) -> tuple:
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

        if axis is not None:
            i, j = axis
            if len(self.value) == 0:
                logger.error('FeatureProcessor value is empty, call first process method.')
                raise Exception('FeatureProcessor value is empty, call first process method.')

            if isinstance(self.value, List):
                self.value = np.array(self.value)

            components = {0, 1, 2}
            missing_component = components - set(axis)
            k = next(iter(missing_component))

            pos_rot = subhalo_particles.star_p
            input_ij, binx, biny, _ = stats.binned_statistic_2d(
                pos_rot[:, i], pos_rot[:, j],
                (self.value[:, k]) ** 2,  # Access the relevant feature
                statistic="mean", bins=bins, range=ranging
            )
            input_ij = np.sqrt(input_ij / 3)
            input_ij[np.isnan(input_ij)] = 0
            return input_ij, binx, biny
        else:
            # 3D
            pass

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
        self.cfg = cfg
        self.map = []
        self.value = []
        self.model = joblib.load(os.path.join(project_root, 'src/data_generation/model/Best_Model-Random-3par.pkl'))

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
        final_predictions = self.model.predict(Data.to_numpy())
        return final_predictions

    @staticmethod
    def get_transition_radius(halo: Halo, subhalo_particles: SubhaloParticles, icl_mask: np.ndarray):
        """
        Computes the transition radius where the intracluster light (ICL) dominates over the brightest cluster galaxy (BCG).

        The transition radius is defined as the point where the cumulative count of ICL particles surpasses
        that of BCG particles. This method calculates the Euclidean distances of particles from the halo center,
        normalizes them by the halo's R200 radius, and determines the transition point based on sorted
        cumulative distributions of ICL and BCG particles.

        Parameters:
        -----------
        halo : Halo
            The halo object containing global properties, including `r200` (the characteristic radius).
        subhalo_particles : SubhaloParticles
            Object containing star particle positions (`star_p`), which should be centered by subtracting `halo.gpos`.
        icl_mask : np.ndarray
            A boolean array indicating whether each star particle belongs to the ICL (1 for ICL, 0 for BCG).

        Returns:
        --------
        float or None
            The transition radius (in units of `r200`) where ICL starts to dominate, or `None` if no transition is found.
        """
        particle_position = subhalo_particles.star_p
        # Computing the Euclidean distance from the new origin to each particle
        distance = np.sqrt((particle_position ** 2).sum(axis=1))
        normalized_distance = distance #/ halo.r200

        # Compute cumulative mass profiles
        sorted_indices = np.argsort(normalized_distance)
        masses = subhalo_particles.star_m[sorted_indices]
        sorted_distances = normalized_distance[sorted_indices]
        sorted_icl_mask = icl_mask[sorted_indices]
        inverse_icl_mask = 1 - sorted_icl_mask

        # Compute cumulative counts for ICL and BCG
        icl_cumulative = np.cumsum(sorted_icl_mask * masses)
        bcg_cumulative = np.cumsum(inverse_icl_mask * masses)

        # Identify transition radius where ICL surpasses BCG
        transition_index = np.where(icl_cumulative > bcg_cumulative)[0]

        if len(transition_index) > 0:
            max_diff = -1
            max_pos = -1
            for i, index in enumerate(transition_index):
                if i > 0:
                    diff = transition_index[i] - transition_index[i - 1]
                    if diff > max_diff:
                        max_diff = diff
                        max_pos = i

            scaled_transition_radius = sorted_distances / halo.r200

            # Create the plot
            plt.figure(figsize=(8, 5))
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(scaled_transition_radius, icl_cumulative, label="cumsum icl", color="blue")
            ax.plot(scaled_transition_radius, bcg_cumulative, label="cumsum bcg", color="orange")
            ax.set_xlabel("r/R200")
            ax.set_ylabel("cumsum")
            ax.legend()

            # Set scientific notation and scale Y-axis by 10^5
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))  # Force scientific notation

            ax.yaxis.set_major_formatter(formatter)  # Apply ScalarFormatter
            ax.yaxis.get_offset_text().set_size(12)  # Adjust exponent text size

            # Show the plot
            plt.savefig(f"cumsum_transition_radius_{halo.halo_id}.png", dpi=300, bbox_inches="tight")
            plt.show()

            transition_radius = sorted_distances[transition_index[max_pos]]
            normalized_transition_radius = transition_radius / halo.r200
            logger.debug(f'Transition radius for halo: {halo.halo_id}: {transition_radius} '
                         f'({round(normalized_transition_radius, 2)} % r200), r200: {halo.r200}')
        else:
            logger.warning("No transition radius found; ICL never dominates.")
            transition_radius = None

        return transition_radius

    # Subhalo particles, halo
    def process(self, halo: Halo, subhalo_particles: SubhaloParticles) -> np.ndarray:
        """
        Process the IntraCluster Light (ICL) for a given halo and its particles.

        Args:
            halo (Halo): The halo object containing halo parameters.

        Returns:
            np.ndarray: A mask (1 or 0) indicating whether each particle belongs to the ICL (1) or BCG (0).

        Raises:
            ValueError: If the scaled particle data is empty.
        """
        particle_velocity = subhalo_particles.star_v
        particle_position = subhalo_particles.star_p

        logger.info(f'Processing ICL for Halo ID {halo.halo_id}.')

        v_cm = np.average(particle_velocity, axis=0)  # calculate the center-of-mass velocity (avg velocity)
        # Computing the Euclidean distance from the new origin to each particle,
        # this value will be normalized inside predict method
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

    def generate_map(self, subhalo_particles: SubhaloParticles, bins: int, ranging: tuple, axis: tuple) -> tuple:
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
        pos_rot = subhalo_particles.star_p
        if axis is not None:
            i, j = axis
            if len(self.value) == 0:
                logger.error('FeatureProcessor value is empty, call first process method.')
                raise Exception('FeatureProcessor value is empty, call first process method.')

            assert pos_rot[:, i].size == pos_rot[:, j].size == len(self.value)

            # Check the range of your data and the bins
            print("Range of pos_rot[:, i]:", np.min(pos_rot[:, i]), np.max(pos_rot[:, i]))
            print("Range of pos_rot[:, j]:", np.min(pos_rot[:, j]), np.max(pos_rot[:, j]))

            output_ij, binx, biny, _ = stats.binned_statistic_2d(
                pos_rot[:, i], pos_rot[:, j],
                self.value,  # Access the ICL feature data
                statistic="mean", bins=bins, range=ranging
            )
            non_nan_bins = np.count_nonzero(~np.isnan(output_ij))
            print(f"Number of bins with data: {non_nan_bins}")

            output_ij[np.isnan(output_ij)] = 0
            return output_ij, binx, biny
        else:
            assert pos_rot[:, 0].size == pos_rot[:, 1].size == pos_rot[:, 2].size == len(self.value)
            statistic_ijk, bin_edges, bin_number = stats.binned_statistic_dd(pos_rot, self.value, statistic="mean",
                                                                             bins=bins, range=ranging)
            statistic_ijk[np.isnan(statistic_ijk)] = 0
            return statistic_ijk, bin_edges, bin_number

    @classmethod
    def get_empty(cls) -> np.ndarray:
        """
        Return an empty value for ICL (an empty array).

        Returns:
            np.ndarray: Empty array for ICL.
        """
        return np.array([])


class StellarMassDensityProcessor(FeatureProcessor):
    """
    Class to process the ...
    Attributes:
        name (str): The name of the feature ('VelocityDispersion').
        map ([]): Velocity Dispersion map.
        value ([]): Placeholder for storing computed value.
    """
    def __init__(self):
        self.name = 'StellarMassDensity'
        self.map = []
        self.value = []

    def process(self, halo: Halo, subhalo_particles: SubhaloParticles) -> np.ndarray:
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
        if not hasattr(subhalo_particles, 'star_m'):
            logger.error('StellarMassDensity needs star_m as input parameter.')
            raise Exception('StellarMassDensity needs star_m as input parameter.')

        if subhalo_particles.star_m.size == 0:
            logger.error("SubhaloParticles cannot have 'm' (mass) empty.")
            raise ValueError("SubhaloParticles cannot have 'm' (mass) empty.")

        return subhalo_particles.star_m

    def generate_map(self, subhalo_particles: SubhaloParticles, bins: int, ranging: tuple, axis: tuple) -> tuple:
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
        pos_rot = subhalo_particles.star_p
        if axis is not None:
            i, j = axis
            if len(self.value) == 0:
                logger.error('FeatureProcessor value is empty, call first process method.')
                raise Exception('FeatureProcessor value is empty, call first process method.')

            if isinstance(self.value, List):
                self.value = np.array(self.value)

            components = {0, 1, 2}
            missing_component = components - set(axis)
            k = next(iter(missing_component))

            input_ij, binx, biny, _ = stats.binned_statistic_2d(
                pos_rot[:, i], pos_rot[:, j],
                (self.value[:, k]) ** 2,  # Access the relevant feature
                statistic="sum", bins=bins, range=ranging
            )
            input_ij = np.sqrt(input_ij / 3)
            input_ij[np.isnan(input_ij)] = 0
            return input_ij, binx, biny
        else:
            # TODO to review computation and output
            assert pos_rot[:, 0].size == pos_rot[:, 1].size == pos_rot[:, 2].size == len(self.value)
            statistic_ijk, bin_edges, bin_number = stats.binned_statistic_dd(pos_rot, self.value, statistic="sum",
                                                                             bins=bins, range=ranging)
            statistic_ijk[np.isnan(statistic_ijk)] = 0
            return statistic_ijk, bin_edges, bin_number

    @classmethod
    def get_empty(cls) -> np.ndarray:
        """
        Return an empty value for velocity dispersion (an empty array).

        Returns:
            np.ndarray: Empty array for velocity dispersion.
        """
        return np.array([])


class DMMassDensityProcessor(FeatureProcessor):
    """
    Class to process the ...
    Attributes:
        name (str): The name of the feature ('VelocityDispersion').
        map ([]): Velocity Dispersion map.
        value ([]): Placeholder for storing computed value.
    """
    def __init__(self):
        self.name = 'DMMassDensity'
        self.map = []
        self.value = []

    def process(self, halo: Halo, subhalo_particles: SubhaloParticles) -> np.ndarray:
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
        if not hasattr(subhalo_particles, 'dm_m'):
            logger.error('StellarMassDensity needs dm_m as input parameter.')
            raise Exception('StellarMassDensity needs dm_m as input parameter.')

        if subhalo_particles.dm_m.size == 0:
            logger.error("SubhaloParticles cannot have 'm' (mass) empty.")
            raise ValueError("SubhaloParticles cannot have 'm' (mass) empty.")

        return subhalo_particles.dm_m

    def generate_map(self, subhalo_particles: SubhaloParticles, bins: int, ranging: tuple, axis: tuple) -> tuple:
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
        pos_rot = subhalo_particles.star_p
        if axis is not None:
            i, j = axis
            if len(self.value) == 0:
                logger.error('FeatureProcessor value is empty, call first process method.')
                raise Exception('FeatureProcessor value is empty, call first process method.')

            if isinstance(self.value, List):
                self.value = np.array(self.value)

            components = {0, 1, 2}
            missing_component = components - set(axis)
            k = next(iter(missing_component))

            input_ij, binx, biny, _ = stats.binned_statistic_2d(
                pos_rot[:, i], pos_rot[:, j],
                (self.value[:, k]) ** 2,  # Access the relevant feature
                statistic="sum", bins=bins, range=ranging
            )
            input_ij = np.sqrt(input_ij / 3)
            input_ij[np.isnan(input_ij)] = 0
            return input_ij, binx, biny
        else:
            # TODO to review computation and output
            assert pos_rot[:, 0].size == pos_rot[:, 1].size == pos_rot[:, 2].size == len(self.value)
            statistic_ijk, bin_edges, bin_number = stats.binned_statistic_dd(pos_rot, self.value, statistic="sum",
                                                                             bins=bins, range=ranging)
            statistic_ijk[np.isnan(statistic_ijk)] = 0
            return statistic_ijk, bin_edges, bin_number

    @classmethod
    def get_empty(cls) -> np.ndarray:
        """
        Return an empty value for velocity dispersion (an empty array).

        Returns:
            np.ndarray: Empty array for velocity dispersion.
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
            'stellar_mass_density': StellarMassDensityProcessor(),
            'dark_matter_mass_density': DMMassDensityProcessor()
            # Add more features if required
        }

        if feature not in feature_processors:
            raise ValueError(f"Invalid feature name {feature} provided")
        logger.info(f"Feature processor '{feature}' created.")
        return feature_processors[feature]
