import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd
import joblib
from omegaconf import DictConfig, OmegaConf
from scipy import stats
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.utils import setup_logger, rotate, plot_icl_bcg_profiles, store_tr_obj, \
    plot_icl_bcg_profiles_conflictive, convolve_and_pixelize_from_bins_1d, lsf_blur_via_sum_of_gaussians, avg_pool_2x2

logger = setup_logger("FeatureProcessor")

# Uncomment for local testing, internet connection required
# dotenv_path = find_dotenv('../.env')
# if dotenv_path:
#    load_dotenv(dotenv_path)
# else:
#    print(".env file not found, skipping load.")


# Add more models if any
MODELS = {
    'IntraClusterLight': 'src/data_generation/model/Best_Model-Random-3par.pkl'
}


def load_model(path):
    print(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=DataConversionWarning)
        warnings.simplefilter("ignore", category=UserWarning)

        project_root = os.getenv('PROJECT_ROOT')
        print('model_path', os.path.join(project_root, path))
        model = joblib.load(os.path.join(project_root, path))
    return model


def load_models(features):
    models = {}
    for feature in features:
        feature_name = feature['name']
        if feature_name in MODELS:
            model_path = MODELS[feature_name]
            model = load_model(model_path)
            models[feature_name] = model
    return models


def attach_models_to_features(features: list, models: dict):
    for feature in features:
        feature_name = getattr(feature, 'name', None)  # or use feature.__class__.__name__
        if feature_name in models:
            feature.model = models[feature_name]


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

    # TODO update
    #def rotate(self, z, axis, degree):  # Old rotate call
    def rotate(self, z, degree):
        if len(self.value) == 0:
            logger.error('FeatureProcessor value is empty, call first process method.')
            raise Exception('FeatureProcessor value is empty, call first process method.')
        self.value = rotate(z, degree, self.value)

    # TODO update
    @abstractmethod
    def generate_map(self, subhalo_particles: SubhaloParticles, bins: int, ranging: tuple, axis: tuple, muse_filter: bool) -> tuple:
        """
        Generate a 2D map for the feature based on particle positions.

        Args:
            particle_positions (np.ndarray): Positions of particles.
            bins (int): Number of bins for the map.
            ranging (tuple): Range for the 2D map.
            axis (tuple): Axes to project (e.g., (0, 1) for x-y plane).
            muse_filter (bool): Whether to apply muse filter for realistic maps. If true, number of bins duplicated

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

    @staticmethod
    def _neighbors(i, j, shape):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < shape[0] and 0 <= nj < shape[1]:
                    yield ni, nj

    def _apply_muse_filter(self, pos_rot_i, pos_rot_j, feature, bins, ranging, final_aggr='mean', n_particles_threshold=100):
        """
        This simulates alterations in telescope imaging to produce realistic maps.

        - Create the maps, similarly to what we have done so far, but at higher resolution. This means that we should
        probably refine the binning, maybe that we have a bin of side x2 the softening length (done in generate_map).
        - Apply a Gaussian Kernel to each bin to mimic an instrument effect called Line Spread Function. What this does
        is to increase the size of the distribution in the bin. The kernel size has an approximated sigma that I can pass
        to you.
        - Apply pixelization: meaning decrease the number of bins with the number of bins measured by the instrument.
        This would mean, reapplying the binning to what we are already doing.
        - Fit a Gaussian to each bin to get the velocity dispersion (i.e. the standard deviation) as we are already doing.
        """
        # We need higher resolution to apply muse filter
        bins = 2 * bins

        logger.debug(f"MUSE filter with n bins: {bins}")

        # This outputs per-bin-distributions, without collapsing them (mean/std) as stats.binned_statistic_2d does
        bin_features, binx, biny = self._bin_particles_2d(
            pos_rot_i, pos_rot_j,  # positions
            feature,  # Velocity
            bins=bins,
            xrange=ranging[0],
            yrange=ranging[1]
        )

        logger.debug(f"MUSE filter bin_features: {bin_features.shape}, {binx[:3]}, {biny[:3]}")

        visited = np.zeros(bin_features.shape[:2], dtype=bool)
        output_ij = np.zeros((bin_features.shape[0], bin_features.shape[1]))

        """
        # Now we apply per-bin 1d gaussian kernel (Line Spread Function)
        for i in range(bin_features.shape[0]):
            for j in range(bin_features.shape[1]):
                velocities = bin_features[i, j]
                if len(velocities) == 0:
                    # No particles in bin
                    #map_mean[i, j] = 0
                    #map_std[i, j] = 0
                    #map_pix[i, j] = 0
                    output_ij[i, j] = 0
                    continue
                if len(velocities) > n_particles_threshold:
                    # To simulate LSF we need to apply the conv on a velocity distribution
                    hist, edges = np.histogram(np.array(velocities), bins=int(np.ceil(np.sqrt(len(velocities)))))

                    # Apply your 1D convolution / pixelization function
                    F_intr, F_conv, F_pix = convolve_and_pixelize_from_bins_1d(
                        hist,
                        edges,
                        stat_mode="std",  # Std since we are handling velocities
                        FWHM=2.5,  # ToDo: Adjust to MUSE
                        ksize=None,  # kernel size, or let function compute
                        bin_factor=2,  # rebin spectral dimension if desired
                        agg_pixel="mean"  # aggregation method for rebin
                    )
                    #_, spectrum = lsf_blur_via_sum_of_gaussians(np.array(velocities))
                    #F_pix = spectrum

                    if final_aggr == "std":
                        output_ij[i, j] = np.std(F_pix)
                    else:
                        output_ij[i, j] = np.mean(F_pix)
                else:
                    output_ij[i, j] = 0
            """

        for i in range(bin_features.shape[0]):
            for j in range(bin_features.shape[1]):

                if visited[i, j]:
                    continue

                velocities = bin_features[i, j]

                # ------------------------
                # CASE 1: empty cell
                # ------------------------
                if len(velocities) == 0:
                    output_ij[i, j] = 0
                    visited[i, j] = True
                    continue

                # ------------------------
                # CASE 2: enough particles → normal processing
                # ------------------------
                if len(velocities) >= n_particles_threshold:
                    hist, edges = np.histogram(
                        np.array(velocities),
                        bins=int(np.ceil(np.sqrt(len(velocities))))
                    )

                    F_intr, F_conv, F_pix = convolve_and_pixelize_from_bins_1d(
                        hist,
                        edges,
                        stat_mode="std",
                        FWHM=2.5,
                        ksize=None,
                        bin_factor=2,
                        agg_pixel="mean"
                    )

                    output_ij[i, j] = np.std(F_pix) if final_aggr == "std" else np.mean(F_pix)
                    visited[i, j] = True
                    continue

                # ------------------------
                # CASE 3: not enough particles → grow region
                # ------------------------
                region = []
                merged_velocities = []
                queue = [(i, j)]

                visited[i, j] = True

                while queue and len(merged_velocities) < n_particles_threshold:
                    x, y = queue.pop(0)

                    region.append((x, y))
                    merged_velocities.extend(bin_features[x, y])

                    for nx, ny in self._neighbors(x, y, bin_features.shape[:2]):
                        if not visited[nx, ny]:
                            visited[nx, ny] = True
                            queue.append((nx, ny))

                # ------------------------
                # Apply SAME operation on merged region
                # ------------------------
                if len(merged_velocities) == 0:
                    value = 0
                else:
                    hist, edges = np.histogram(
                        np.array(merged_velocities),
                        bins=int(np.ceil(np.sqrt(len(merged_velocities))))
                    )

                    F_intr, F_conv, F_pix = convolve_and_pixelize_from_bins_1d(
                        hist,
                        edges,
                        stat_mode="std",
                        FWHM=2.5,
                        ksize=None,
                        bin_factor=2,
                        agg_pixel="mean"
                    )

                    value = np.std(F_pix) if final_aggr == "std" else np.mean(F_pix)

                # ------------------------
                # Copy result to all merged cells
                # ------------------------
                for x, y in region:
                    output_ij[x, y] = value

        output_ij = avg_pool_2x2(output_ij)

        return output_ij, binx, biny

    def _bin_particles_2d(self, pos_x, pos_y, feature, bins, xrange, yrange):
        """
        Bin particle data in 2D **without collapsing statistics**.

        Parameters
        ----------
        pos_x, pos_y : array-like, shape (N,)
            Particle positions in two axes.
        feature : array-like, shape (N,)
            Particle values to store (e.g., LOS velocities)
        bins : int or tuple
            Number of bins in x and y (int or (nx, ny))
        xrange, yrange : 2-element lists
            Min/max of each axis

        Returns
        -------
        bin_features : 2D array of lists, shape (ny, nx)
            Each entry is a list of particle feature values in that bin.
        xedges, yedges : bin edges
        """
        pos_x = np.asarray(pos_x)
        pos_y = np.asarray(pos_y)
        feature = np.asarray(feature)

        if isinstance(bins, int):
            nx = ny = bins
        else:
            nx, ny = bins

        x_min = np.asarray(xrange[0]).item()
        x_max = np.asarray(xrange[1]).item()
        xedges = np.linspace(x_min, x_max, nx + 1)
        y_min = np.asarray(yrange[0]).item()
        y_max = np.asarray(yrange[1]).item()
        yedges = np.linspace(y_min, y_max, ny + 1)

        # Initialize 2D array of lists
        bin_features = np.empty((ny, nx), dtype=object)
        for i in range(ny):
            for j in range(nx):
                bin_features[i, j] = []

        # Digitize particle positions
        ix = np.digitize(pos_x, xedges) - 1
        iy = np.digitize(pos_y, yedges) - 1


        # Mask particles outside the range
        mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix = ix[mask]; iy = iy[mask]; f = feature[mask]

        """
        # Assign features to bins
        for x_idx, y_idx, val in zip(ix, iy, f):
            bin_features[y_idx, x_idx].append(val)
        """
        """
        indices = np.column_stack([iy, ix])
        unique_bins, inverse = np.unique(indices, axis=0, return_inverse=True)

        for bin_idx, (y_idx, x_idx) in enumerate(unique_bins):
            if bin_idx % 500000 == 0:
                print(f"Processed {bin_idx}/{len(ix)} particles")
            bin_features[y_idx, x_idx] = f[inverse == bin_idx].tolist()
        """
        # Assign features to bins
        for x_idx, y_idx, val in zip(ix, iy, f):
            bin_features[y_idx, x_idx].append(val)

        return bin_features, xedges, yedges


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
    def __init__(self, kwargs):
        self.name = 'VelocityDispersion'
        self.map = []
        self.value = []

    def process(self, halo: Halo, subhalo_particles: SubhaloParticles):
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
        self.value = subhalo_particles.star_v

    def generate_map(self, subhalo_particles: SubhaloParticles, bins: int, ranging: tuple, axis: tuple, muse_filter: bool = False) -> tuple:
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
        feature = self.value

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


            if muse_filter:
                logger.debug(f"Applying MUSE filter...")

                print(pos_rot[:,i].shape)
                print(pos_rot[:,j].shape)
                print((feature[:, k]) ** 2)

                input_ij, binx, biny = self._apply_muse_filter(
                    pos_rot[:, i], pos_rot[:, j],
                    feature=(feature[:, k]) ** 2,  # Access the relevant feature,
                    bins=bins,
                    ranging=ranging,
                    final_aggr="mean",
                    n_particles_threshold=10
                )
                input_ij = np.sqrt(input_ij / 3)
                input_ij[np.isnan(input_ij)] = 0
                logger.debug(f"MUSE filter successfully applied -- Shapes: input_ij={input_ij.shape}, binx={binx.shape}, biny={biny.shape}")
                return input_ij, binx, biny

            else:
                input_ij, binx, biny, _ = stats.binned_statistic_2d(
                    pos_rot[:, i], pos_rot[:, j],
                    (feature[:, k]) ** 2,  # Access the relevant feature
                    statistic="mean", bins=bins, range=ranging
                )
                input_ij = np.sqrt(input_ij / 3)
                input_ij[np.isnan(input_ij)] = 0
                logger.debug(f"Shapes: input_ij={input_ij.shape}, binx={binx.shape}, biny={biny.shape}")
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
    def __init__(self, cfg: DictConfig, kwargs):
        # TODO init with the cfg if exists
        self.transition_radius = None
        self.name = 'IntraClusterLight'
        self.cfg = cfg
        self.map = []
        self.value = []
        self.model = None

        attr_list = self._build_configuration(cfg, ['G'])
        for key, value in attr_list.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
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
        final_predictions = self.model.predict(Data.to_numpy())
        return final_predictions

    def _compute_icl_mask(self, halo: Halo, subhalo_particles: SubhaloParticles):
        particle_velocity = subhalo_particles.star_v
        particle_position = subhalo_particles.star_p

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
        icl_mask = np.zeros((rows,))
        if ids.size > 0:
            # if particle belongs to ICL=1, otherwise BCG=0
            icl_mask[ids] = 1
        else:
            logger.warning('No particles predicted as ICL.')
            store_tr_obj(self.cfg, halo, subhalo_particles, icl_mask)
            raise ValueError("icl_mask is empty; no masking applied.")

        logger.info(f'Generated ICL mask for {len(icl_mask)} particles.')
        return icl_mask

    def _compute_transition_radius(self, halo: Halo, subhalo_particles: SubhaloParticles, icl_mask: np.ndarray):
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
        logger.debug(f'Computing transition radius for: {halo.halo_id}')
        store_tr_obj(self.cfg, halo, subhalo_particles, icl_mask)
        particle_position = subhalo_particles.star_p
        # Computing the Euclidean distance from the new origin to each particle
        distance = np.sqrt((particle_position ** 2).sum(axis=1))
        normalized_distance = distance

        # Compute cumulative mass profiles
        sorted_indices = np.argsort(normalized_distance)
        if not hasattr(subhalo_particles, 'star_m'):
            raise ValueError('The object subhalos_particles should have an attribute called star_m in order to'
                             ' compute the transition radius')

        masses = subhalo_particles.star_m[sorted_indices]
        sorted_distances = normalized_distance[sorted_indices]
        sorted_icl_mask = icl_mask[sorted_indices]
        inverse_icl_mask = 1 - sorted_icl_mask

        # Compute cumulative counts for ICL and BCG
        icl_cumulative = np.cumsum(sorted_icl_mask * masses)
        bcg_cumulative = np.cumsum(inverse_icl_mask * masses)

        # Identify transition radius where ICL surpasses BCG
        transition_index = np.where(icl_cumulative > bcg_cumulative)[0]

        # To analyze cumsum profiles
        r_by_r200 = sorted_distances #/ halo.r200
        self.log_r = np.log10(r_by_r200)

        max_pos = -1
        if len(transition_index) > 0:
            max_diff = -1
            for i, index in enumerate(transition_index):
                if i > 0:
                    diff = transition_index[i] - transition_index[i - 1]
                    if diff > max_diff:
                        max_diff = diff
                        max_pos = i

            logger.debug(f'Index for the transition radius: {transition_index[max_pos]}')
            transition_radius = sorted_distances[transition_index[max_pos]]
            normalized_transition_radius = transition_radius / halo.r200
            logger.debug(f'Transition radius for halo: {halo.halo_id}: {transition_radius} '
                         f'({round(normalized_transition_radius, 2)} % r200), r200: {halo.r200}')
        else:
            plot_icl_bcg_profiles_conflictive(self.cfg, icl_cumulative, bcg_cumulative, self.log_r, halo.halo_id)
            logger.error('No transition radius found; ICL never dominates.')
            raise Exception('No transition radius found; ICL never dominates.')

        self.icl_cumulative = icl_cumulative
        self.bcg_cumulative = bcg_cumulative
        self.tr_index = transition_index[max_pos]

        # Store values within the feature
        self.transition_radius = transition_radius

        # Build cumsum profiles
        plot_icl_bcg_profiles(self.cfg, icl_cumulative, bcg_cumulative, self.log_r, transition_index[max_pos], halo.halo_id)

    def process(self, halo: Halo, subhalo_particles: SubhaloParticles):
        """
        Process the IntraCluster Light (ICL) for a given halo and its particles.

        Args:
            halo (Halo): The halo object containing halo parameters.

        Returns:
            np.ndarray: A mask (1 or 0) indicating whether each particle belongs to the ICL (1) or BCG (0).

        Raises:
            ValueError: If the scaled particle data is empty.
        """
        # Non overlapping particles
        icl_mask = self._compute_icl_mask(halo, subhalo_particles)
        self.value = icl_mask

        if hasattr(self, 'transition_radius_compute'):
            if self.transition_radius_compute:
                self._compute_transition_radius(halo, subhalo_particles, icl_mask)
            elif self.cfg.catalog.get('transition_radius', None) is None:
                logger.warning('Warning, you must define a transition radius for generating the maps')

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
        feature = self.value

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
                feature,  # Access the ICL feature data
                statistic="mean", bins=bins, range=ranging
            )
            non_nan_bins = np.count_nonzero(~np.isnan(output_ij))
            print(f"Number of bins with data: {non_nan_bins}")

            output_ij[np.isnan(output_ij)] = 0

            return output_ij, binx, biny
        else:
            # TODO
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
    def __init__(self, kwargs):
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
    def __init__(self, kwargs):
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
    def get_feature(feature: DictConfig, **kwargs) -> FeatureProcessor:
        """
        Get an instance of the appropriate FeatureProcessor based on the feature name.

        Args:
            feature (DictConfig): The feature type ('velocity_dispersion' or 'icl').
            **kwargs: Additional keyword arguments passed to the feature processor.

        Returns:
            FeatureProcessor: An instance of the appropriate feature processor.

        Raises:
            ValueError: If an invalid feature name is provided.
        """

        cfg = kwargs.get('cfg', None)
        # Add more feature processors as needed
        feature_name = feature['name']
        kwargs = feature.copy()
        OmegaConf.set_struct(kwargs, False)
        del kwargs['name']

        match feature_name:
            case 'VelocityDispersion':
                feature_processor = VelocityDispersionProcessor(kwargs=kwargs)
            case'IntraClusterLight':
                feature_processor = ICLProcessor(cfg=cfg, kwargs=kwargs)
            case 'StellarMassDensity':
                feature_processor = StellarMassDensityProcessor(kwargs=kwargs)
            case 'DarkMatterMassDensity':
                feature_processor = DMMassDensityProcessor(kwargs=kwargs)
            # Add more features if required
            case _:
                raise ValueError(f"Invalid feature name {feature} provided")

        logger.info(f"Feature processor '{feature}' created.")
        return feature_processor
