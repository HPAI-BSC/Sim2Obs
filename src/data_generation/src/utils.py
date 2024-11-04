import hashlib
import logging
import os
import re
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from src.data_generation.src.constants import G
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloParticles


def setup_logger():
    """
    Configures and returns a logger to capture debugging and information logs.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,  # Set the logging level to DEBUG to capture all levels
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Custom log format
        handlers=[
            logging.FileHandler("../data_generation.log"),  # Log to a file named app.log
            logging.StreamHandler()  # Also log to the console
        ]
    )
    logger = logging.getLogger("Logger")
    return logger


logger = setup_logger()


def label_particles(subhalo_particles: SubhaloParticles, halo_context: HaloContext, id_halo: int) \
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

    R200 = halo_context.r200[id_halo]
    M200 = halo_context.m200[id_halo]
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
    logger.info("Labeling particles ...")
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
            logger.debug(file_path)
            if os.path.isfile(file_path):
                image_hash = hash_image(file_path)

                if image_hash in hashes:
                    duplicates.append((hashes[image_hash], filename))  # Found a duplicate
                else:
                    hashes[image_hash] = filename  # Store the original filename
    return duplicates


def compute_intrahalo_variance(input_dir: str) -> pd.DataFrame:
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
    pattern = r"^halo-(\d+)_.*\.npz$"
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
        full_path_vel = full_path_icl.replace('labels', 'velocity_dispersion')

        if os.path.exists(full_path_icl) and os.path.exists(full_path_vel):
            with np.load(full_path_icl) as label_data, np.load(full_path_vel) as vel_disp_data:
                if halo_id not in maps:
                    maps[halo_id] = {"label_zooms": [], "vel_disp_zooms": []}
                maps[halo_id]["label_zooms"].append(label_data['arr_0'][:, :, :2])
                maps[halo_id]["vel_disp_zooms"].append(vel_disp_data['arr_0'][:, :, :2])

    df = pd.DataFrame(columns=["halo_id", "icl_variance", "vel_variance"])

    for halo_id, values in maps.items():
        # Stack zooms and calculate variance
        label_zooms_stacked = np.stack(values["label_zooms"], axis=0)  # Shape: (n_zooms, H, W, 2)
        vel_disp_zooms_stacked = np.stack(values["vel_disp_zooms"], axis=0)  # Shape: (n_zooms, H, W, 2)

        # Compute variance across the zoom dimension (axis=0)
        label_variance = np.var(label_zooms_stacked, axis=0).mean()  # Single variance value for labels
        vel_disp_variance = np.var(vel_disp_zooms_stacked, axis=0).mean()  # Single variance value for velocity dispersion

        # Append the row with halo ID and variances
        df = pd.concat([df, pd.DataFrame({"halo_id": [halo_id], "icl_variance": [label_variance], "vel_variance": [vel_disp_variance]})], ignore_index=True)

    return df

