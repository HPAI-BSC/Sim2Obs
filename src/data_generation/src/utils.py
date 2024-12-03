import hashlib
import logging
import os
import re
from typing import List, Tuple
import eagleSqlTools as sql
import h5py

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloParticles

# TODO
load_dotenv('../.env')


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f"{name}.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def get_connection():
    username = os.getenv("USERNAME")
    password = os.getenv("PASSWORD")
    con = sql.connect(username, password)
    return con


def read_dataset(itype, att, path, nfiles=16):
    data = []
    conversion_att = {
        'Mass': {
            'aexp': 0,
            'hexp': -1
        },
        'Coordinates': {
            'aexp': 1,
            'hexp': -1
        },
        'Velocity': {
            'aexp': 1/2,
            'hexp': 0
        }
    }

    def find_folders_with_snapshot(parent_folder):
        # Create a regex pattern to match folders containing 'snapshot'
        pattern = re.compile(r'snapshot_*', re.IGNORECASE)

        found_folder = [item for item in os.listdir(parent_folder) if pattern.search(item)]
        if len(found_folder) == 0:
            return None
        return found_folder[0]

    snapshot_folder = find_folders_with_snapshot(path)

    for i in range(nfiles):
        file = os.path.join(path, snapshot_folder, f'{snapshot_folder.replace("snapshot", "snap")}.{i}.hdf5')
        f = h5py.File(file, 'r')
        tmp = f[f'PartType{itype}/{att}'][...]
        data.append(tmp)
        cgs = f[f'PartType{itype}/{att}'].attrs.get('CGSConversionFactor')
        if cgs is None:
            cgs = 1
        # Not working
        # https://arxiv.org/pdf/1706.09899 page 5
        keys = list(f[f'PartType{itype}/{att}'].attrs.keys())
        print(att, keys)
        aexp = f[f'PartType{itype}/{att}'].attrs.get('aexp−scale−exponent')
        hexp = f[f'PartType{itype}/{att}'].attrs.get('h−scale−exponent')
        print(f'   file: {i}, cfg={cgs}, aexp={aexp}, hpex={hexp}')

        a = f['Header'].attrs.get('Time')
        h = f['Header'].attrs.get('HubbleParam')
        f.close()

    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np. concatenate(data)

    print(att, len(data))

    if data.dtype != np.int32 and data.dtype != np.int64:
        aexp = conversion_att.get(att, None)['aexp']
        hexp = conversion_att.get(att, None)['hexp']
        data = np.multiply(data, cgs * a**aexp * h**hexp, dtype='f8')
    return data


def label_particles(subhalo_particles: SubhaloParticles, halo_context: HaloContext, id_halo: int, G: float) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify particles as belonging to the Brightest Cluster Galaxy (BCG) or Intracluster Light (ICL).

    This function utilizes a pre-trained machine learning model to label particles based on their
    distances from the center of the halo, their velocities, and the halo mass. The particles are
    classified into two categories: BCG (label 1) and ICL (label 0).

    Parameters:
    -----------
    r : numpy.ndarray
        An array of distances of particles from the center of the halo, in kpc.
    modulo_v : numpy.ndarray
        An array of particle velocities, in km/s.
    R200 : float
        The radius (R200) of the halo, in kpc, used for scaling distances.
    M200 : float
        The mass (M200) of the halo, in solar masses (Msun), used for calculating the velocity scale.

    Returns:
    --------
    tuple
        A tuple containing two pandas DataFrames:
        - df_bcg: DataFrame with indices of particles classified as BCG.
        - df_icl: DataFrame with indices of particles classified as ICL.

    Note:
    -----
    Ensure that the pre-trained model 'Best_Model-Random-3par.pkl' is available in the 'model/'
    directory before calling this function. The model expects specific input features as outlined
    in the function.
    """
    v_cm = np.average(subhalo_particles.v, axis=0)  # calculate the center-of-mass velocity (avg velocity)
    # Computing the Euclidean distance from the new origin to each particle
    distance = np.sqrt((subhalo_particles.p ** 2).sum(axis=1))
    # Computing the delta vel. from each particle in comparison to the mean-vel.
    v_delta = np.sqrt(((subhalo_particles.v - v_cm) ** 2).sum(axis=1))

    final_model = joblib.load('../model/Best_Model-Random-3par.pkl')
    # Define simulation parameters
    R200, M200, _, _ = halo_context.get_item(id_halo)
    V200 = np.sqrt(G * M200 / R200)

    # Create a DataFrame with features required by the model
    # Features are: log-scaled distance (r/R200), scaled velocity (modulo_v/V200), and log-scaled halo mass (M200)
    Data = pd.DataFrame({
        'distance': np.log10(distance / R200),  # Log-scaled distance
        'velocity': v_delta / V200,  # Scaled velocity
        'm200': [np.log10(M200 * 1e-14)] * len(v_delta)  # Log-scaled halo mass (m200, constant for all particles)
    })

    # Replace any infinite values in the DataFrame with NaN, then drop all rows with NaN values
    Data = Data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Create an array of particle indices (0 to len(r)-1)
    IDS = np.linspace(0, len(distance) - 1, len(distance), dtype=int)

    # Use the trained model to make predictions on the data
    # The model predicts whether each particle belongs to BCG (label 1) or ICL (label 0)
    # log.info("Labeling particles ...")
    final_predictions = final_model.predict(Data.to_numpy())

    # Extract indices of particles predicted to be part of the BCG
    ids_BCG = IDS[np.where(final_predictions == 1)[0]]

    # Extract indices of particles predicted to be part of the ICL
    ids_ICL = IDS[np.where(final_predictions == 0)[0]]
    df_bcg = pd.DataFrame(ids_BCG)
    df_icl = pd.DataFrame(ids_ICL)
    return df_bcg, df_icl


def get_icl_mask(df_icl: pd.DataFrame, m: np.ndarray) -> np.ndarray:
    """
    Creates a mask array for ICL particles based on the provided DataFrame of ICL indices.

    Parameters:
    -----------
    df_icl : pd.DataFrame
        DataFrame containing indices of ICL particles.
    m : np.ndarray
        Array representing masses of the particles.

    Returns:
    --------
    np.ndarray
        An array of the same shape as `m` where ICL particle positions are marked with 1 and others with 0.
    """
    type4 = np.zeros_like(m)
    if df_icl.size > 0:  # Check if df_icl is not empty
        try:
            type4[df_icl[0]] = 1
        except IndexError as e:
            print(
                "IndexError: df_icl[0] is out of bounds. Ensure df_icl has valid indices.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    else:
        print("Warning: df_icl is empty; no masking applied.")
    return type4


def hash_image(image_path: str) -> str:
    """
    Generate a hash for an image.

    Args:
        image_path (str): The file path to the image.

    Returns:
        str: The MD5 hash of the image.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")  # Ensure the image is in RGB mode
        img = img.resize((100, 100))  # Resize to reduce hash collision
        img_data = img.tobytes()
        return hashlib.md5(img_data).hexdigest()


def find_duplicates_in_halo(image_dir: str, idhalo: str) -> List:
    """
    Finds duplicated images within a specified directory that match a given halo identifier.

    Parameters:
    -----------
    image_dir : str
        The directory containing the images to check for duplicates.
    idhalo : str
        The halo identifier used to filter images for duplication checks.

    Returns:
    --------
    list
        A list of tuples, each containing the filenames of duplicated images.
    """
    hashes = {}
    duplicates = []

    for filename in os.listdir(image_dir):
        if f"halo-{idhalo}" in filename:
            file_path = os.path.join(image_dir, filename)
            # log.debug(file_path)
            if os.path.isfile(file_path):
                image_hash = hash_image(file_path)

                if image_hash in hashes:
                    duplicates.append((hashes[image_hash], filename))  # Found a duplicate
                else:
                    hashes[image_hash] = filename  # Store the original filename
    return duplicates


def compute_intrahalo_variance(input_dir: str, feature_1_name: str, feature_2_name: str) -> pd.DataFrame:
    """
    Computes the intrahalo variance for particle classifications and velocity dispersion maps
    based on multiple zooms per subhalo within a specified directory.

    Parameters:
    -----------
    input_dir : str
        Directory path where the zoom data (.npz files) for each subhalo is stored.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the intrahalo variances for labels and velocity dispersion per halo.
    """
    path_feature_1 = os.path.join(input_dir, feature_1_name)
    path_feature_2 = os.path.join(input_dir, feature_2_name)

    if not os.path.exists(path_feature_1) or not os.path.exists(path_feature_2):
        raise Exception(f'Following paths do not exists: {path_feature_1}, {path_feature_2}')

    pattern = fr"halo-(\d+)_.*\.npz$"
    maps = {}

    for file in os.listdir(path_feature_1):
        filepath_1 = os.path.join(path_feature_1, file)
        filepath_2 = filepath_1.replace(path_feature_1, path_feature_2)
        match = re.search(pattern, file)
        halo_id = int(match.group(1))  # Extract halo ID

        if os.path.exists(filepath_1) and os.path.exists(filepath_2):
            if filepath_1.endswith('npz') and filepath_2.endswith('npz'):
                with np.load(filepath_2) as label_data, np.load(filepath_1) as vel_disp_data:
                    if halo_id not in maps:
                        maps[halo_id] = {feature_1_name: [], feature_2_name: []}
                    maps[halo_id][feature_2_name].append(label_data['arr_0'][:, :, :2])
                    maps[halo_id][feature_1_name].append(vel_disp_data['arr_0'][:, :, :2])

    df = pd.DataFrame(columns=["halo_id", f"{feature_1_name}_variance", f"{feature_2_name}_variance"])

    for halo_id, values in maps.items():
        # Stack zooms and calculate variance
        label_zooms_stacked = np.stack(values[feature_2_name], axis=0)  # Shape: (n_zooms, H, W, 2)
        vel_disp_zooms_stacked = np.stack(values[feature_1_name], axis=0)  # Shape: (n_zooms, H, W, 2)

        # Compute variance across the zoom dimension (axis=0)
        label_variance = np.var(label_zooms_stacked, axis=0).mean()  # Single variance value for labels
        vel_disp_variance = np.var(vel_disp_zooms_stacked, axis=0).mean()  # Single variance value for velocity dispersion

        # Append the row with halo ID and variances
        df = pd.concat([df, pd.DataFrame({"halo_id": [halo_id], f"{feature_1_name}_variance": [label_variance],
                                          f"{feature_2_name}_variance": [vel_disp_variance]})], ignore_index=True)

    df = df.sort_values(by="halo_id")
    return df


def compute_intrahalo_variance_ilaria(input_dir: str) -> pd.DataFrame:
    pattern = r"^D(\d+)_.*\.npz$"
    matched_files = []

    # Iterate over all files in the directory
    for filename in os.listdir(input_dir):
        if re.match(pattern, filename):
            matched_files.append(filename)

    maps = {}

    for file in matched_files:
        match = re.search(pattern, file)
        halo_id = int(match.group(1))  # Extract halo ID
        full_path_icl = os.path.join(input_dir, file)
        full_path_vel = full_path_icl.replace('Labels', 'Velocity_Dispersion')

        if os.path.exists(full_path_icl) and os.path.exists(full_path_vel):
            with np.load(full_path_icl) as label_data, np.load(full_path_vel) as vel_disp_data:
                if halo_id not in maps:
                    maps[halo_id] = {"label_zooms": [], "vel_disp_zooms": []}
                maps[halo_id]["label_zooms"].append(label_data['arr_0'])
                maps[halo_id]["vel_disp_zooms"].append(vel_disp_data['arr_0'])

    df = pd.DataFrame(columns=["halo_id", "icl_variance", "vel_variance"])

    for halo_id, values in maps.items():
        # Stack zooms and calculate variance
        label_zooms_stacked = np.stack(values["label_zooms"], axis=0)  # Shape: (n_zooms, H, W, 2)
        vel_disp_zooms_stacked = np.stack(values["vel_disp_zooms"], axis=0)  # Shape: (n_zooms, H, W, 2)

        # Compute variance across the zoom dimension (axis=0)
        label_variance = np.var(label_zooms_stacked, axis=0).mean()  # Single variance value for labels
        vel_disp_variance = np.var(vel_disp_zooms_stacked,
                                   axis=0).mean()  # Single variance value for velocity dispersion

        # Append the row with halo ID and variances
        df = pd.concat([df, pd.DataFrame(
            {"halo_id": [halo_id], "icl_variance": [label_variance], "vel_variance": [vel_disp_variance]})],
                       ignore_index=True)

    df = df.sort_values(by="halo_id")
    return df


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

    #log.debug(f'--> Rotation angle: {angle_degree}')

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


def muse_bins_r(fov: float, spatial_resolution: float) -> tuple[int, float]:
    """
    Calculate the number of bins and the radius for MUSE observations.

    Parameters:
    -----------
    fov : float
        The field of view in kiloparsecs (kpc).
    spatial_resolution : float
        The spatial resolution in kiloparsecs (kpc) that defines the size of each bin.

    Returns:
    --------
    tuple
        A tuple containing:
        - bins : int
            The number of bins along one axis calculated as the quotient of
            the field of view divided by the spatial resolution.
        - r : float
            The radius of the field of view, computed as half of the fov.

    Note:
    -----
    The function assumes that both `fov` and `spatial_resolution` are positive
    values and that the spatial resolution is less than or equal to the field of view.
    """
    bins = int(fov / spatial_resolution)  # Number of bins along one axis
    r = fov / 2  # Radius of the field of view
    return bins, r


def setup_output_directories(catalog_path: str, rotation_angle, input_feature_name: str, output_feature_name: str):
    if os.path.exists(catalog_path):
        if rotation_angle is not None:
            dataset_path = os.path.join(catalog_path, f'rotation_{rotation_angle}')
            if not os.path.exists(dataset_path):
                # TODO can we executed this inside mn5 ?
                os.mkdir(dataset_path)
        else:
            dataset_path = catalog_path
        if not os.path.exists(os.path.join(dataset_path, output_feature_name)):
            os.mkdir(os.path.join(dataset_path, output_feature_name))
        if not os.path.exists(os.path.join(dataset_path, input_feature_name)):
            os.mkdir(os.path.join(dataset_path, input_feature_name))

