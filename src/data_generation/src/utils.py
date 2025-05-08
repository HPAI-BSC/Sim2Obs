import logging
import os
import re
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

    # Ensure no previous handlers are attached
    if logger.hasHandlers():
        logger.handlers.clear()

    # Prevent log propagation to avoid duplicate logs
    # logger.propagate = False

    console_handler = logging.StreamHandler()

    # Updated format string to ensure line number is not included
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


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

    # Update nfiles
    snapdir = os.path.join(path, snapshot_folder)
    nfiles = sum(1 for entry in Path(snapdir).iterdir() if entry.is_file())

    for i in range(nfiles):
        file = os.path.join(path, snapshot_folder, f'{snapshot_folder.replace("snapshot", "snap")}.{i}.hdf5')
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


def get_transform_matrix(n: np.ndarray, axis: np.ndarray, normalized: bool = True) -> np.ndarray:
    """
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
    plt.figure(figsize=(8, 6))

    dataset_dir = '/gpfs/scratch/bsc70/hpai/storage/projects/space/datasets/'
    name = cfg.catalog.path.replace(dataset_dir, '')
    name = name.replace('/', '_')

    print(name)

    # Take the log10 of transition radius values
    if log_normalization:
        # Remove None or non-positive values before taking the log
        transition_radius_list = [r for r in transition_radius_list if r and r > 0]
        log_transition_radius = np.log10(transition_radius_list)
        plt.hist(log_transition_radius, bins=50, color='skyblue', edgecolor='black')

        plt.xlabel(r'$\log_{10}(\mathrm{Transition\ Radius\ [kpc]})$')
        plt.ylabel('Counts')
        plt.title('Histogram of Log Transition Radii')
        path = cfg.paths.datasets_dir + f'/../logs/transition_radius_histogram_log_{name}.png'
    else:
        plt.hist(transition_radius_list, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel(r'(\mathrm{Transition\ Radius\ [kpc]})$')
        plt.ylabel('Counts')
        plt.title('Histogram of Transition Radii')
        path = cfg.paths.datasets_dir + f'/../logs/transition_radius_histogram_{name}.png'

    plt.savefig(path, dpi=300)


def plot_icl_bcg_profiles(icl_cumulative, bcg_cumulative, r_by_r200, t_idx, halo_id):
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(r_by_r200, icl_cumulative, label='ICL Cumulative Mass', color='orange')
    plt.plot(r_by_r200, bcg_cumulative, label='BCG Cumulative Mass', color='blue')
    plt.axvline(r_by_r200[t_idx], color='red', linestyle='--', label='Transition radius')
    plt.scatter(r_by_r200[t_idx], icl_cumulative[t_idx], color='red', zorder=5)
    plt.xlabel('log(r / r200)')
    plt.ylabel('Cumulative Mass (M☉)')
    plt.title(f'Cumulative Mass Profiles for Halo {halo_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = f'/gpfs/scratch/bsc70/hpai/storage/projects/space/logs/transition_radius_cumsum_{halo_id}.png'
    plt.savefig(path, dpi=300)
    #plt.show()


def plot_icl_bcg_mass_pdf(icl_pdf, bcg_pdf, bin_centers, halo_id):
    plt.figure(figsize=(8, 5))
    plt.plot(bin_centers, icl_pdf, label='ICL Cumulative Mass', color='orange')
    plt.plot(bin_centers, bcg_pdf, label='BCG Cumulative Mass', color='blue')
    plt.xlabel('r / r200')
    plt.ylabel('Mass-weighted PDF')
    plt.title(f'log/r/R200) Halo {halo_id}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = f'/gpfs/scratch/bsc70/hpai/storage/projects/space/logs/mass_weighted_pdf_{halo_id}.png'
    plt.savefig(path, dpi=300)


def plot_icl_bcg_density_profile(r, density_total, density_icl, density_bcg, halo_id):
    plt.figure(figsize=(8, 5))
    plt.plot(r, density_total, label="Total", color="black")
    if density_icl is not None:
        plt.plot(r, density_icl, label="ICL", color="orange")
        plt.plot(r, density_bcg, label="BCG", color="blue")

    plt.yscale("log")
    plt.xlabel("r / R200")
    plt.ylabel("Density (M☉ / kpc³)")
    plt.title(f"Stellar Mass Density Profile - Halo {halo_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    path = f'/gpfs/scratch/bsc70/hpai/storage/projects/space/logs/density_profile_{halo_id}.png'
    plt.savefig(path, dpi=300)


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
    if isinstance(d, dict):
        return [d[column][i] for i in range(d[column].shape[0])]
    elif isinstance(d, np.ndarray):
        return [d[i] for i in range(d.shape[0])]


def to_list_in_batches(d, column, batch_size=1000000):
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


def rotate(z, axis, degree, X):
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
    new_z = rotate_vector(v=z, axis=axis, angle_=degree)
    Q = get_transform_matrix(new_z, z, normalized=False)
    X_rot = np.float32(rotate_3dvec(Q, X))
    return X_rot


def r500_to_r200(r500):
    """Convert R500c to R200c using scaling relation"""
    return r500 / 0.65  # R200c ≈ R500c / 0.65


def m500_to_m200(m500):
    """Convert M500c to M200c assuming NFW scaling"""
    return m500 * (1.54) ** 3  # M200c ≈ M500c * (R200c / R500c)^3


def store_overlapping_matrix_as_df(cfg, overlapping_matrix):
    df_list = []

    for halo_id, tuple_list in overlapping_matrix.items():
        temp_df = pd.DataFrame(halo_id, tuple_list, columns=['halo', 'overlapping halo - subhalo'])
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    path = cfg.paths.datasets_dir + f'/../logs/overlapping_halos_{cfg.catalog.name}.csv'
    df.to_csv(path, index=False)