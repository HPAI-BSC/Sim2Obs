import json
import logging
import os
import pickle
import re
import traceback
from pathlib import Path
import eagleSqlTools as sql
import h5py

import numpy as np

# do not delete
import math
import pandas as pd
import matplotlib.pyplot as plt

# from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig
from scipy.spatial.distance import cosine
from scipy.signal import fftconvolve

# This will search for the .env file and load it if found
# Uncomment to connect to eagle db for local test,
# however MN5 does not have internet connection, then is not needed
# dotenv_path = find_dotenv('../.env')
# if dotenv_path:
#    load_dotenv(dotenv_path)
# else:
#    print(".env file not found, skipping load.")


def setup_logger(name: str) -> logging.Logger:
    """
    Sets up and returns a logger instance with both console and file handlers.

    Parameters:
    -----------
    name : str
        The name of the logger instance.

    Returns:
    --------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Only add a handler if none exists
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


logger = setup_logger("Utils")


def get_connection():
    """
    Establishes a connection to the EAGLE SQL database using credentials from environment variables.

    Returns:
    --------
    sql.Connection
        Connection object to interact with the EAGLE SQL database.
    """
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    con = sql.connect(username, password)
    return con


# TODO optimize
def read_dataset(itype: int, att: str, path: str, nfiles: int =16) -> np.ndarray:
    """
    Reads and processes data from HDF5 snapshot files for a specific particle type and attribute.
    # https://arxiv.org/pdf/1706.09899

    Parameters:
    -----------
    itype : int
        Particle type index (e.g., 0 for gas, 1 for dark matter).
    att : str
        Attribute to extract (e.g., 'Mass', 'Coordinates', 'Velocity').
    path : str
        Path to the directory containing snapshot files.
    nfiles : int, optional
        Number of snapshot files to process, default is 16.

    Returns:
    --------
    np.ndarray
        Combined and processed data array for the specified attribute.

    Notes:
    ------
    This function handles unit conversions using attributes such as CGSConversionFactor, aexp-scale-exponent,
    and h-scale-exponent, which are retrieved from the HDF5 file metadata.
    """
    data = []

    def find_folders_with_snapshot(parent_folder: str) -> str:
        """
        Finds a folder containing 'snapshot' within the parent folder.

        Parameters:
        -----------
        parent_folder : str
            Path to the parent folder to search in.

        Returns:
        --------
        str
            Name of the folder containing 'snapshot', or None if not found.
        """
        pattern = re.compile(r'snapshot_*', re.IGNORECASE)
        found_folder = [item for item in os.listdir(parent_folder) if pattern.search(item)]
        return found_folder[0] if found_folder else None

    snapshot_folder = find_folders_with_snapshot(path)
    if not snapshot_folder:
        raise FileNotFoundError("Snapshot folder not found in the specified path.")
    else:
        pass
        #logger.debug(snapshot_folder)

    # Update nfiles
    snapdir = os.path.join(path, snapshot_folder)
    nfiles = sum(1 for entry in Path(snapdir).iterdir() if entry.is_file())
    #logger.debug(f'nfiles: {nfiles}')

    try:
        for i in range(nfiles):
            file = os.path.join(path, snapshot_folder, f'{snapshot_folder.replace("snapshot", "snap")}.{i}.hdf5')
            #logger.debug(file)
            with h5py.File(file, 'r') as f:
                tmp = f[f'PartType{itype}/{att}'][...]
                data.append(tmp)
                cgs = f[f'PartType{itype}/{att}'].attrs.get('CGSConversionFactor', 1)
                aexp = f[f'PartType{itype}/{att}'].attrs.get('aexp-scale-exponent')
                hexp = f[f'PartType{itype}/{att}'].attrs.get('h-scale-exponent')
                var = f[f'PartType{itype}/{att}'].attrs.get('VarDescription')
                # print(f'   file: {i}, cfg={cgs}, aexp={aexp}, hpex={hexp}')

                a = f['Header'].attrs.get('Time')
                h = f['Header'].attrs.get('HubbleParam')

        if len(tmp.shape) > 1:
            data = np.vstack(data)
        else:
            data = np.concatenate(data)
        '''
        if data.dtype not in (np.int32, np.int64):
            data = np.multiply(data, cgs * a ** aexp * h ** hexp, dtype='f8')
        '''
    except Exception as e:
        logger.error(e)
        error_message = traceback.format_exc()
        logger.error(error_message)
    return data


def angle(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the angle between two vectors using the cosine similarity.

    Parameters:
    -----------
    u : np.ndarray
        The first vector. Must be of the same dimension as `v`.
    v : np.ndarray
        The second vector. Must be of the same dimension as `u`.

    Returns:
    --------
    float
        The angle in radians between the vectors `u` and `v`.

    Note:
    -----
    The input vectors `u` and `v` should be normalized (unit vectors) for
    accurate angle calculations. If the vectors are not normalized, the returned
    angle may not accurately represent the angle between them.
    """
    return np.arccos(1 - cosine(u, v))


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle_: float) -> np.ndarray:
    """
    Rotate a vector around a specified axis by a given angle.

    Parameters:
    -----------
    v : np.ndarray
        The 3D vector to be rotated. Must be a 1D array of length 3.
    axis : np.ndarray
        The 3D vector representing the axis of rotation. Must also be a 1D array of length 3.
    angle_ : float
        The angle by which to rotate the vector, in degrees.

    Returns:
    --------
    np.ndarray
        The rotated 3D vector as a 1D array.

    Note:
    -----
    The function assumes that both the input vector `v` and the rotation axis
    `axis` are 3D vectors (1D arrays with length 3). The angle is converted
    from degrees to radians for the calculations.
    """
    # Normalize the rotation axis
    u = axis / np.linalg.norm(axis)

    # Convert angle to radians
    theta = np.radians(angle_)

    # Calculate components of Rodrigues' rotation formula
    v_rot = (v * np.cos(theta) +
             np.cross(u, v) * np.sin(theta) +
             u * (np.dot(u, v) * (1 - np.cos(theta))))

    return v_rot


def random_vector_at_angle(direction, angle_deg=45, length=1.0):
    """
    Generates a random vector with an angle of angle_deg to direction.
    The vector lies on a cone around the direction vector.
    """
    # Normalize direction vector
    direction = np.array(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    angle_rad = np.deg2rad(angle_deg)

    # Find two orthogonal vectors to the direction (form a basis for the plane perpendicular to direction)
    # Method: find a vector that's not parallel to direction, then use Gram-Schmidt
    if abs(direction[0]) < 0.9:  # direction is not too close to x-axis
        temp = np.array([1, 0, 0])
    else:  # direction is close to x-axis, use y-axis instead
        temp = np.array([0, 1, 0])

    # First orthogonal vector (in the plane perpendicular to direction)
    u1 = temp - np.dot(temp, direction) * direction
    u1 = u1 / np.linalg.norm(u1)

    # Second orthogonal vector (cross product gives another perpendicular vector)
    u2 = np.cross(direction, u1)
    u2 = u2 / np.linalg.norm(u2)  # Should already be normalized, but just in case

    # Generate random angle in the perpendicular plane
    theta = np.random.uniform(0, 2 * np.pi)

    # The vector components:
    # - Component along direction: length * cos(angle_rad)
    # - Component in perpendicular plane: length * sin(angle_rad) * (random direction in plane)
    parallel_component = length * np.cos(angle_rad) * direction
    perpendicular_component = length * np.sin(angle_rad) * (np.cos(theta) * u1 + np.sin(theta) * u2)

    return parallel_component + perpendicular_component


def get_transform_matrix_old(n: np.ndarray, axis: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
    OLD TRANSFORM MATRIX. NOT ROBUST ENOUGH, USE NEW VERSION
    Calculate the rotation matrix that aligns a given vector with the new z-axis.

    Parameters:
    -----------
    n : np.ndarray
        A vector representing the new z-axis direction. Should be a 1D array of length 3.
    axis : np.ndarray
        The original z-axis vector (should be a standard reference vector).
    normalized : bool, optional
        Whether the vector `n` is normalized. Defaults to True.
    debug : bool, optional
        If True, enables debug logging for internal calculations. Defaults to False.

    Returns:
    --------
    np.ndarray
        A 3x3 rotation matrix that transforms the original coordinate system
        to the new coordinate system aligned with the vector `n`.

    Note:
    -----
    If the vector `n` is not normalized and `normalized` is set to False,
    the function will normalize it internally before computation.
    """
    if not normalized:
        # unit vector
        n /= np.linalg.norm(n)

    b = np.cross(axis, n)
    theta = angle(axis, n)
    angle_degree = theta * (180 / np.pi)

    # Quaternion components
    q0 = np.cos(theta / 2)
    q1, q2, q3 = np.sin(theta / 2) * b

    '''
    Q_wrong = np.array([
        [q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
        [2 * (q2 * q1 - q0 * q3), q0 ** 2 - q1 ** 2 - q3 ** 2 + q2 ** 2, 2 * (q3 * q2 + q0 * q1)],
        [2 * (q3 * q1 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0 ** 2 - q1 ** 2 + q3 ** 2 - q2 ** 2]
    ])
    '''

    Q = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])

    return Q


def get_transform_matrix(n: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Calculate the rotation matrix that aligns a given vector with the new z-axis.

    Parameters:
    -----------
    n : np.ndarray
        A vector representing the new z-axis direction. Should be a 1D array of length 3.
    axis : np.ndarray
        The original z-axis vector (should be a standard reference vector).

    Returns:
    --------
    np.ndarray
        A 3x3 rotation matrix that transforms the original coordinate system
        to the new coordinate system aligned with the vector `n`.
    """
    # Normalize inputs
    if np.linalg.norm(n) != 1.0:
        n = n / np.linalg.norm(n)
    axis = axis / np.linalg.norm(axis)  # Ensure axis is unit vector

    # Handle colinear vectors
    cos_theta = np.dot(axis, n)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    b = np.cross(axis, n)
    b_norm = np.linalg.norm(b)
    tol = 1e-12

    if b_norm < tol:
        # Colinear case
        if cos_theta > 0:  # Vectors are parallel
            return np.eye(3)
        else:  # Antiparallel: 180° rotation
            # Find an arbitrary perpendicular axis
            candidates = np.eye(3)
            for candidate in candidates:
                perp = candidate - np.dot(candidate, axis) * axis
                if np.linalg.norm(perp) > tol:
                    perp = perp / np.linalg.norm(perp)
                    break
            q0 = 0.0  # cos(90°)
            q1, q2, q3 = perp  # sin(90°) * perp
    else:
        # General case
        b = b / b_norm  # Normalize rotation axis
        q0 = np.cos(theta / 2)
        sin_half = np.sin(theta / 2)
        q1, q2, q3 = sin_half * b

    # Construct rotation matrix
    Q = np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])
    return Q


def rotate_3dvec(Q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Apply a rotation matrix to a 3-dimensional vector.

    Parameters:
    -----------
    Q : np.ndarray
        A 3x3 rotation matrix.
    vec : np.ndarray
        A 1D array or list of length 3 representing the vector to be rotated.

    Returns:
    --------
    np.ndarray
        The rotated 3-dimensional vector as a 1D array.

    Note:
    -----
    Ensure that the input vector `vec` is a 1D array or list of length 3.
    The rotation matrix `Q` should be properly constructed and validated
    before applying this function to ensure correct results.
    """
    return (Q @ vec.T).T


def circle(x: float, y: float, h: float, k: float, r: float) -> float:
    """
    Check if a point (x, y) lies within a circle defined by its center (h, k)
    and radius r.

    Parameters:
    -----------
    x : float
        The x-coordinate of the point to check.
    y : float
        The y-coordinate of the point to check.
    h : float
        The x-coordinate of the center of the circle.
    k : float
        The y-coordinate of the center of the circle.
    r : float
        The radius of the circle.

    Returns:
    --------
    float
        A value representing the square of the distance from the point (x, y)
        to the center of the circle (h, k) divided by the square of the radius
        (r). If this value is less than or equal to 1, the point lies within
        or on the circle.
    """
    return ((x - h) ** 2 + (y - k) ** 2) / r ** 2


def muse_bins_r(fov: float, spatial_resolution: float, bin_spatial_resolution_scale: float) -> tuple[int, float]:
    """
    Calculate the number of spatial bins and the radius for MUSE observations.

    Parameters:
    -----------
    fov : float
        The field of view (FoV) in kiloparsecs (kpc).
    spatial_resolution : float
        The base spatial resolution in kiloparsecs (kpc), which defines the
        unscaled size of each bin.
    bin_spatial_resolution_scale : float
        A scaling factor applied to the spatial resolution to determine the
        final bin size.

    Returns:
    --------
    tuple[int, float]
        A tuple containing:
        - bins : int
            The number of bins along one axis, calculated as the quotient of
            the field of view divided by the scaled spatial resolution.
        - r : float
            The radius of the field of view, computed as half of the FoV.

    Notes:
    ------
    - The function assumes `fov` and `spatial_resolution` are positive values.
    - The spatial resolution is scaled by `bin_spatial_resolution_scale` before computing bins.
    - The number of bins is rounded down to the nearest integer.
    """
    bin_size = spatial_resolution * bin_spatial_resolution_scale
    bins = int(fov / bin_size) # Number of bins along one axis
    return bins


def plot_transition_radius(cfg, transition_radius_list, log_normalization=True):
    """
    Plots and saves a histogram of transition radius values, optionally using a log10 scale.

    The function filters out invalid values (e.g. None or non-positive) when log-normalization is applied,
    then plots the histogram and saves it as a PNG file in the logs directory.

    Parameters:
    ----------
    cfg : object
        Configuration object containing path attributes:
        - cfg.paths.datasets_dir : str, base path to datasets
        - cfg.catalog.path : str, full path to the dataset catalog file

    transition_radius_list : list of float
        List of transition radii (typically normalized by R200) to be plotted.

    log_normalization : bool, optional (default=True)
        If True, applies a log10 transformation to the data before plotting.

    Returns:
    -------
    None
        The function saves the histogram plot as a PNG file and does not return any value.
    """
    plt.figure(figsize=(8, 6))
    dataset_dir = cfg.paths.datasets_dir
    name = cfg.catalog.path.replace(dataset_dir, '')
    name = name.replace('/', '_')

    # Take the log10 of transition radius values
    if log_normalization:
        # Remove None or non-positive values before taking the log
        transition_radius_list = [r for r in transition_radius_list if r and r > 0]
        log_transition_radius = np.log10(transition_radius_list)
        plt.hist(log_transition_radius, bins=50, color='skyblue', edgecolor='black')

        plt.xlabel(r'$\log_{10}(\mathrm{Transition\ Radius\ r/R200})$')
        plt.ylabel('Counts')
        plt.title('Histogram of Log Transition Radii')
        path = cfg.paths.datasets_dir + f'/../logs/transition_radius_histogram_log_{name}.png'
    else:
        plt.hist(transition_radius_list, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Transition Radius r')
        plt.ylabel('Counts')
        plt.title('Histogram of Transition Radii')
        path = cfg.paths.datasets_dir + f'/../logs/transition_radius_histogram_{name}.png'

    plt.savefig(path, dpi=300)


def plot_hist(df, attribute, title, bins, log=False):
    """
       Plots a histogram of the 'm200' column from a DataFrame.

       Parameters
       ----------
       df : pandas.DataFrame
           DataFrame containing the halos attributes.
       bins : int or sequence, optional
           Number of bins or specific bin edges (default: 30).
       log : bool, optional
           If True, use a logarithmic scale on the x-axis (default: False).
       title : str, optional
           Title of the plot.
       """
    if attribute not in df:
        raise Exception()

    dataset_dir = '/gpfs/scratch/bsc70/hpai/storage/projects/space/datasets'
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.hist(df[attribute].dropna(), bins=bins, color='skyblue', edgecolor='black', alpha=0.7)

    if log:
        plt.xscale('log')

    plt.xlabel(attribute)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    #plt.show()

    name = 'tng_cluster'
    path = dataset_dir + f'/../logs/{name}_{attribute}_histogram.png'
    plt.savefig(path, dpi=300)


def plot_icl_bcg_profiles(cfg, icl_cumulative, bcg_cumulative, r_by_r200, t_idx, halo_id):
    """
    Plots and saves the cumulative mass profiles of ICL and BCG components for a given halo.

    The function visualizes the cumulative mass as a function of normalized radius (r/r200),
    highlighting the transition radius with a vertical line and a scatter point.

    Parameters:
    ----------
    cfg : object
        Configuration object containing the path information, particularly:
        - cfg.paths.datasets_dir : str, base directory path to datasets.

    icl_cumulative : array-like
        Cumulative mass profile of the intracluster light (ICL) component.

    bcg_cumulative : array-like
        Cumulative mass profile of the brightest cluster galaxy (BCG) component.

    r_by_r200 : array-like
        Radii normalized by r200, corresponding to the mass profiles.

    t_idx : int
        Index of the transition radius in the `r_by_r200` array.

    halo_id : int or str
        Identifier of the halo being analyzed (used in the plot title and filename).

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the logs directory and displays it,
        but does not return any value.
    """
    dataset_dir = cfg.paths.datasets_dir
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(r_by_r200, icl_cumulative, label='ICL Cumulative Mass', color='orange')
    plt.plot(r_by_r200, bcg_cumulative, label='BCG Cumulative Mass', color='blue')
    plt.axvline(r_by_r200[t_idx], color='red', linestyle='--', label='Transition radius')
    plt.scatter(r_by_r200[t_idx], icl_cumulative[t_idx], color='red', zorder=5)
    plt.xlabel('r')
    plt.ylabel('Cumulative Mass (M☉)')
    plt.title(f'Cumulative Mass Profiles for Halo {halo_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    name = cfg.catalog.path.replace(dataset_dir, '')
    name = name.replace('/', '_')
    path = dataset_dir + f'/../logs/{name}_transition_radius_cumsum_{halo_id}.png'
    plt.savefig(path, dpi=300)
    #plt.show()


def plot_icl_bcg_profiles_conflictive(cfg, icl_cumulative, bcg_cumulative, r_by_r200, halo_id):
    """
    Plots and saves the cumulative mass profiles of ICL and BCG components for a given halo.

    Parameters:
    ----------
    cfg : object
        Configuration object containing the path information, particularly:
        - cfg.paths.datasets_dir : str, base directory path to datasets.

    icl_cumulative : array-like
        Cumulative mass profile of the intracluster light (ICL) component.

    bcg_cumulative : array-like
        Cumulative mass profile of the brightest cluster galaxy (BCG) component.

    r_by_r200 : array-like
        Radii normalized by r200, corresponding to the mass profiles.

    halo_id : int or str
        Identifier of the halo being analyzed (used in the plot title and filename).

    Returns:
    -------
    None
        The function saves the plot as a PNG file in the logs directory and displays it.
    """
    dataset_dir = cfg.paths.datasets_dir

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(r_by_r200, icl_cumulative, label='ICL Cumulative Mass', color='orange')
    plt.plot(r_by_r200, bcg_cumulative, label='BCG Cumulative Mass', color='blue')
    plt.xlabel('r / r200')
    plt.ylabel('Cumulative Mass (M☉)')
    plt.title(f'Cumulative Mass Profiles for Halo {halo_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    name = cfg.catalog.path.replace(dataset_dir, '').replace('/', '_')
    path = dataset_dir + f'/../logs/{name}_cumsum_{halo_id}_error.png'
    plt.savefig(path, dpi=300)
    plt.show()


def setup_output_directories(catalog_path: str, rotation_angle: float, bin_spatial_resolution_scale: float,
                             input_feature_name: str, output_feature_name: str):
    """
    Creates necessary output directories for storing feature data, optionally organizing by rotation angle.

    Parameters:
    -----------
    catalog_path : str
        The base path where the catalog directories are located.
    rotation_angle : optional
        The rotation angle to use for naming subdirectories. If None, directories are created directly under the
        catalog path.
    input_feature_name : str
        Name of the input feature for which the directory will be created.
    output_feature_name : str
        Name of the output feature for which the directory will be created.

    Notes:
    ------
    - If directories already exist, they will not be recreated.
    - This function ensures that directories for both input and output features are properly organized under the
      specified catalog path, with an optional subdirectory for rotation angles.
    """
    if os.path.exists(catalog_path):
        if rotation_angle is not None:
            dataset_path = os.path.join(catalog_path, f'rotation_{rotation_angle}')
            dataset_path = f'{dataset_path}_s{bin_spatial_resolution_scale}'
            if not os.path.exists(dataset_path):
                os.mkdir(dataset_path)
        else:
            dataset_path = os.path.join(catalog_path, f'no_rotation')
            dataset_path = f'{dataset_path}_s{bin_spatial_resolution_scale}'
            if not os.path.exists(dataset_path):
                os.mkdir(dataset_path)

        if not os.path.exists(os.path.join(dataset_path, output_feature_name)):
            os.mkdir(os.path.join(dataset_path, output_feature_name))
        if not os.path.exists(os.path.join(dataset_path, input_feature_name)):
            os.mkdir(os.path.join(dataset_path, input_feature_name))


def to_list(d, column):
    """
    Converts data from a dictionary or NumPy array into a list.

    Parameters:
    ----------
    d : dict or np.ndarray
        - If dict: Assumes `d[column]` is an indexable object (e.g., NumPy array or list)
          and extracts elements from it.
        - If np.ndarray: Returns a list of elements directly from the array.

    column : str
        The key to access the desired data if `d` is a dictionary. Ignored if `d` is a NumPy array.

    Returns:
    -------
    list
        A list of elements extracted from the input data.

    Raises:
    ------
    TypeError:
        If `d` is neither a dict nor a NumPy array.
    """
    if isinstance(d, dict):
        return [d[column][i] for i in range(d[column].shape[0])]
    elif isinstance(d, np.ndarray):
        return [d[i] for i in range(d.shape[0])]


def to_list_in_batches(d, column, batch_size=1000000):
    """
    Converts the values of a specific column in a DataFrame or similar structure to a list,
    processing the data in batches to handle large datasets efficiently.

    Parameters:
    ----------
    d : pandas.DataFrame or similar
        The input data structure containing the column to convert.

    column : str
        The name of the column to extract and convert to a list.

    batch_size : int, optional (default=1000000)
        The number of rows to process in each batch.

    Returns:
    -------
    result : list
        A flattened list containing all the elements from the specified column,
        aggregated batch by batch.

    Notes:
    -----
    Assumes that a function `to_list(batch, column)` exists and returns a list of items
    from the given batch.
    """
    result = []
    # Loop over the data in batches
    for start in range(0, d[column].shape[0], batch_size):
        end = min(start + batch_size, d[column].shape[0])
        batch = d[column][start:end]
        result.extend(to_list(batch, column))  # Add batch to result list
    return result


def apply_conversion(values, scaling_formula, scalefactor, h, **kwargs):
    """Applies the correct unit transformation, including indexing if necessary."""
    # Prepare a safe eval context
    context = {"x": values, "a": scalefactor, "h": h, "np": np, "math": math}
    context.update(kwargs)

    return eval(scaling_formula, context)


def dict_depth(d, level=0):
    if (not isinstance(d, DictConfig) and not isinstance(d, dict)) or not d:  # Base case: not a dict or empty dict
        return level
    return max(dict_depth(v, level + 1) for v in d.values())


def rotate_old(z, axis, degree, X):
    """
    THIS WAS THE OLD VERSION, WHICH ONLY ROTATED Z TO AXIS = [X,Y,Z] AND THEN 45 DEG ON THE XY PLANE; DO NOT USE IT.
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
    new_z = rotate_vector(v=z, axis=axis, angle_=degree)
    Q = get_transform_matrix(new_z, z, normalized=False)
    X_rot = np.float32(rotate_3dvec(Q, X))
    return X_rot


def rotate(z, degree, X):
    """
    NEW ROTATE. GENERATES A RANDOM VECTOR NEW_Z WITH ANGLE(Z, NEW_Z) = DEGREE. THEN ROTATES X BY NEW_Z.
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
    if degree == 0:
        return X
    new_z = random_vector_at_angle(z, angle_deg=degree)
    Q = get_transform_matrix(new_z, z)
    X_rot = np.float32(rotate_3dvec(Q, X))
    return X_rot


def r500_to_r200(r500):
    """Convert R500c to R200c using scaling relation"""
    return r500 / 0.65  # R200c ≈ R500c / 0.65


def m500_to_m200(m500):
    """Convert M500c to M200c assuming NFW scaling"""
    return m500 * (1.54) ** 3  # M200c ≈ M500c * (R200c / R500c)^3


def store_overlapping_matrix_as_df(cfg, overlapping_matrix):
    """
    Stores an overlapping matrix as a CSV file using a pandas DataFrame.

    Each row of the DataFrame represents a halo and its list of overlapping (halo_id, subhalo_id) tuples.
    The resulting CSV is saved under the `logs` directory relative to the dataset path defined in the configuration.

    Parameters:
    ----------
    cfg : object
        Configuration object with at least the following attributes:
        - cfg.paths.datasets_dir : str, path to the datasets directory
        - cfg.catalog.path : str, full path to the dataset catalog file

    overlapping_matrix : dict
        Dictionary where keys are halo IDs (int or str), and values are lists of tuples
        representing overlapping halo-subhalo pairs in the format (halo_id, subhalo_id).

    Returns:
    -------
    None
        The function writes the DataFrame to a CSV file but does not return anything.
    """
    dataset_dir = cfg.paths.datasets_dir
    df = pd.DataFrame(data=[], columns=['halo id', 'overlapping (halo id, subhalo id)'])

    for halo_id, tuple_list in overlapping_matrix.items():
        temp_df = pd.DataFrame([[halo_id, tuple_list]], columns=['halo id', 'overlapping (halo id, subhalo id)'])
        df = pd.concat([df, temp_df], ignore_index=True)

    name = cfg.catalog.path.replace(dataset_dir, '')
    name = name.replace('/', '_')

    path = cfg.paths.datasets_dir + f'/../logs/overlapping_halos_{name}.csv'
    df.to_csv(path, index=False)


def store_obj(cfg, halo, subhalo_ctx, cosmos_data):
    sim_code = cfg.catalog.path.split('/')[-1]
    datasets_dir = cfg.paths.datasets_dir
    name = cfg.catalog.name
    path = datasets_dir + f'/../logs/halo_{name}_{sim_code}_{halo.halo_id}.pkl'
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(halo, f)

    path = datasets_dir + f'/../logs/subhalo_ctx_{name}_{sim_code}_{halo.halo_id}.pkl'
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(subhalo_ctx, f)

    path = datasets_dir + f'/../logs/cosmos_{name}_{sim_code}_{halo.halo_id}.pkl'
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(cosmos_data, f)


def store_tr_obj(cfg, halo, subhalo_particles, icl_mask):
    print(f'Storing oject for transition radius debugging for: {halo.halo_id}')
    sim_code = cfg.catalog.path.split('/')[-1]
    datasets_dir = cfg.paths.datasets_dir
    name = cfg.catalog.name
    path = datasets_dir + f'/../logs/halo_{name}_{sim_code}_{halo.halo_id}.pkl'
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(halo, f)

    path = datasets_dir + f'/../logs/subhalo_particles_{name}_{sim_code}_{halo.halo_id}.pkl'
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(subhalo_particles, f)

    path = datasets_dir + f'/../logs/icl_mask_{name}_{sim_code}_{halo.halo_id}.npy'
    np.save(path, icl_mask)


def merge_json_files_to_csv(input_dir, output_file, chunk_size=1000, length=None):
    """
    Efficiently merges many JSON files from a directory into one CSV.

    Args:
        input_dir (str): Path containing JSON files.
        output_file (str): Path to the output CSV file.
        chunk_size (int): Number of JSONs to load per chunk before writing to disk.
    """
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    if length is not None:
        json_files = json_files[:length]

    total = len(json_files)
    if total == 0:
        print("No JSON files found.")
        return

    print(f"Found {total} JSON files. Merging in chunks of {chunk_size}...")

    first_write = True
    buffer = []

    if os.path.exists(output_file):
        os.remove(output_file)

    for i, filename in enumerate(json_files, start=1):
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    buffer.append(data)
                else:
                    print(f"Skipping {filename} (not a dict)")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Write to CSV in chunks to save memory
        if len(buffer) >= chunk_size or i == total:
            df = pd.DataFrame(buffer)
            df.to_csv(output_file, index=False, mode="a", header=first_write)
            first_write = False
            buffer.clear()

            print(f"Processed {i}/{total} files ({(i/total)*100:.1f}%)")

    print(f"Merge complete. Output saved to {output_file}")


def gaussian_kernel_1d(ksize, sigma, normalize=True):
    x = np.arange(ksize) - ksize//2
    K = np.exp(-(x**2)/(2*sigma**2))
    if normalize:
        K /= K.sum()
    return K

def rebin_1d(arr, factor, agg="mean"):
    """
    Rebin a 1D array by integer factor.
    """
    n = len(arr)
    m = n // factor
    arr = arr[:m*factor].reshape(m, factor)
    if agg == "mean":
        return arr.mean(axis=1)
    elif agg == "sum":
        return arr.sum(axis=1)
    else:
        raise ValueError("agg must be 'mean' or 'sum'")


def avg_pool_2x2(arr):
    # truncate to even shape to make divisible by 2
    rows, cols = arr.shape
    arr = arr[:rows - rows % 2, :cols - cols % 2]  # truncate if needed

    # reshape into blocks of 2x2
    arr_reshaped = arr.reshape(rows // 2, 2, cols // 2, 2)

    # take mean over the 2x2 blocks
    pooled = arr_reshaped.mean(axis=(1, 3))
    return pooled


def convolve_and_pixelize_from_bins_1d(
    F_intr,
    edges,
    stat_mode="std",
    FWHM=2.5,
    ksize=None,
    bin_factor=1,
    agg_pixel="mean"
):
    print('aa')
    n = F_intr.shape[0]

    # FWHM → sigma
    sigma = FWHM / (2 * np.sqrt(2 * np.log(2)))
    bin_width = edges[1] - edges[0]
    sigma_bins = sigma_vel / bin_width

    print('sigma', sigma)

    # Gaussian kernel length
    if ksize is None:
        ksize = int(np.ceil(6 * sigma))
        if ksize % 2 == 0:
            ksize += 1

    # Build kernel -- apply it in bin units
    K = gaussian_kernel_1d(ksize, sigma_bins, normalize=True)

    # Convolution
    F_conv = fftconvolve(F_intr, K, mode="same")

    # Pixelization
    if bin_factor == 1:
        F_pix = F_conv.copy()
    else:
        F_pix = rebin_1d(F_conv, bin_factor, agg=agg_pixel)

    return F_intr, F_conv, F_pix
