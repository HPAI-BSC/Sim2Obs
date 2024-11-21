import hashlib
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

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloParticles

# TODO change this
load_dotenv('../.env')


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
