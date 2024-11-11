import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from src.data_generation.src.constants import CATALOG
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.utils import logger


def create_maps(r200_halo: float, spos_rot: np.ndarray, pos_rot: np.ndarray, vel_rot: np.array, type4: np.array) -> dict:
    """
    Create 2D maps for particle properties in a halo based on their positions and velocities.

    Parameters:
    -----------
    r200_halo : float
        The R200 radius of the halo.
    spos_rot : np.ndarray
        An array of shape (N, 3) representing the rotated positions of subhalos in 3D space.
    pos_rot : np.ndarray
        An array of shape (M, 3) representing the rotated positions of particles.
    vel_rot : np.ndarray
        An array of shape (M, 3) representing the rotated velocities of particles.
    type4 : np.ndarray
        An array indicating the type of particles.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'maps': A list of tuples, each representing a map for the projections.
        - 'interlopers_id': An array containing the indices of interloper subhalos.
    """
    # r200_halo, spos_rot (posicion of subhalos), pos_rot and vel_rot position and velocity of particles in the
    # primary subhalo, type4 particles of primary subhalo that are ICL

    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
    ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

    # Considering as interloper any subhalo contained in a box defined by [-r, r] in the x and y directions and
    # inside [-R_200, R_200] along the z axis

    interlopers_id = np.where((np.abs(spos_rot[:, 0]) < r)
                              & (np.abs(spos_rot[:, 1]) < r)
                              & (np.abs(spos_rot[:, 2]) < r200_halo))[0]

    # removing first galaxy (central/BCG), since it's usually the central galaxy or BCG
    # TODO is it possible to have no interlopers ?
    interlopers_id = interlopers_id[1:]

    if len(interlopers_id) == 0:
        logger.warning('There is no interlopers.')

    # Create three different maps based on projections onto different coordinate planes (X-Y, Y-Z, X-Z).
    # Break up the 2D spatial plane into small regions (bins), compute the mean of the squared velocities
    # in each region, and return the resulting grid of values.
    projections = [(0, 1), (1, 2), (0, 2)]
    projection_vel = [2, 0, 1]

    maps = []

    for i, projection in enumerate(projections):
        j, k = projection
        v = projection_vel[i]

        # Velocity dispersion for j-k projection
        vel_disp_jk, binx, biny, _ = stats.binned_statistic_2d(pos_rot[:, j], pos_rot[:, k], (vel_rot[:, v]) ** 2,
                                                                     statistic="mean", bins=bins, range=ranging)
        vel_disp_jk = np.sqrt(vel_disp_jk / 3)

        # ICL fraction for j-k projection
        label_jk, _, _, _ = stats.binned_statistic_2d(pos_rot[:, j], pos_rot[:, k], type4,
                                                      statistic="sum", bins=bins, range=ranging)
        label_count_jk, _, _, _ = stats.binned_statistic_2d(pos_rot[:, j], pos_rot[:, k], type4,
                                                            statistic="count", bins=bins, range=ranging)
        label_jk = label_jk / label_count_jk
        label_jk[np.isnan(label_jk)] = 0
        vel_disp_jk[np.isnan(vel_disp_jk)] = 0
        maps.append((label_jk, vel_disp_jk, biny))

    return {
        'maps': maps,
        'interlopers_id': interlopers_id
    }


def create_random_zoom(label_masked: np.ndarray, vel_disp_masked: np.ndarray, mask: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Creates a random zoom of the masked labels and velocity dispersion maps.

    Parameters:
    ----------
    label_masked : np.ndarray
        A 2D array representing the masked labels.

    vel_disp_masked : np.ndarray
        A 2D array representing the masked velocity dispersion.

    Returns:
    -------
    tuple
        A tuple containing:
        - label_masked1 : np.ndarray
            A smaller 2D array of masked labels.
        - vel_disp_masked1 : np.ndarray
            A smaller 2D array of masked velocity dispersion.
    """
    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
    xo, yo = (int(bins / 2), int(bins / 2))
    xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)

    vel_disp_1 = vel_disp_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                       yoprime - int(bins / 4): yoprime + int(bins / 4)]

    label_1 = label_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                    yoprime - int(bins / 4): yoprime + int(bins / 4)]

    mask1 = mask[xoprime - int(bins / 4):xoprime + int(bins / 4),
            yoprime - int(bins / 4):yoprime + int(bins / 4)]
    return label_1, vel_disp_1, mask1


def create_n_zooms(label_masked: np.ndarray, vel_disp_masked: np.ndarray, **kwargs) -> None:
    """
    Creates and saves three random zooms of the masked labels and velocity dispersion maps.

    Parameters:
    ----------
    label_masked : np.ndarray
        A 2D array representing the masked labels.

    vel_disp_masked : np.ndarray
        A 2D array representing the masked velocity dispersion.

    **kwargs : keyword arguments
        Additional parameters required for file naming:
        - idhalo : int
        - idsubhalo : int
        - projection : str
        - n: int
        - mask: np.ndarray
        - rotation: str

    Returns:
    -------
    None
    """
    idhalo = kwargs["idhalo"]
    idsubhalo = kwargs["idsubhalo"]
    projection = kwargs["projection"]
    n = kwargs["n"]
    mask = kwargs["mask"]
    rotation = kwargs.get("rotation", None)

    for i in range(n):
        label_i, vel_disp_i, mask_i = create_random_zoom(label_masked, vel_disp_masked, mask)

        # Expand the dimensions of the mask to match the shape of the other arrays
        mask_expanded = np.expand_dims(mask_i, axis=-1)  # Shape: (H, W, 1)

        # Expand the dimensions of vel_disp_i and label_i to make them 3D
        vel_disp_i_expanded = np.expand_dims(vel_disp_i, axis=-1)  # Shape: (H, W, 1)
        label_i_expanded = np.expand_dims(label_i, axis=-1)  # Shape: (H, W, 1)

        # Concatenate the mask as an additional channel to the zoomed arrays
        vel_disp_with_mask = np.concatenate((vel_disp_i_expanded, mask_expanded), axis=-1)  # Shape: (H, W, 2)
        label_with_mask = np.concatenate((label_i_expanded, mask_expanded), axis=-1)  # Shape: (H, W, 2)

        if rotation is not None:
            path_vel = f"../data/npz/velocity_dispersion/halo-{idhalo}_subhalo-{idsubhalo}_{CATALOG}_rotation_{rotation}_{projection}_zoom_{i}"
            path_icl = f"../data/npz/labels/halo-{idhalo}_subhalo-{idsubhalo}_{CATALOG}_rotation_{rotation}_{projection}_zoom_{i}"

        else:
            path_vel = f"../data/npz/velocity_dispersion/halo-{idhalo}_subhalo-{idsubhalo}_{CATALOG}_{projection}_zoom_{i}"
            path_icl = f"../data/npz/labels/halo-{idhalo}_subhalo-{idsubhalo}_{CATALOG}_{projection}_zoom_{i}"

        np.savez(path_vel, vel_disp_with_mask)
        np.savez(path_icl, label_with_mask)


def get_interlopers_mask(label: np.ndarray, vel_disp: np.ndarray, spos_rot: np.ndarray, shmr: pd.DataFrame,
                         interlopers_id: np.ndarray, biny: np.ndarray) -> np.ndarray:
    """
    Create a mask to identify and exclude interloper subhalos from calculations.

    Parameters:
    ----------
    label : np.ndarray
        A 2D array representing the labels of the particles.

    vel_disp : np.ndarray
        A 2D array representing the velocity dispersion.

    spos_rot : np.ndarray
        A 2D array containing the spatial positions of each subhalo.

    shmr : pd.DataFrame
        A DataFrame containing the SubhaloHalfmassRadType of subhalos.

    interlopers_id : np.ndarray
        An array of identifiers for the interloper subhalos.

    biny : np.ndarray
        An array representing the y-coordinates for the masking grid.

    Returns:
    -------
    np.ndarray
        A boolean mask identifying interloper regions.
    """
    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius for the grid
    maschera = np.zeros_like(label)

    # Generate X and Y based on maschera's shape for consistency
    rows, cols = maschera.shape
    X, Y = np.meshgrid(
        np.linspace(-r, r, cols),  # X coordinates matching columns in maschera
        np.linspace(biny[0], biny[-1], rows)  # Y coordinates matching rows in maschera
    )

    for i in interlopers_id:
        # For each interloper get its center position (h, k) and its radius
        h = spos_rot[i, 0]
        k = spos_rot[i, 1]
        rradius = shmr.iloc[i, 0]
        for cc in range(rows):
            for dd in range(cols):
                radius_circle = circle(X[cc, dd], Y[cc, dd], h, k, rradius)
                if radius_circle <= 1:
                    maschera[cc, dd] = 1

    # Check if the mask is just zero
    if np.all(maschera == 0):
        logger.warning('Mask is full of zeros.')

    assert maschera.shape == label.shape == vel_disp.shape

    maschera = maschera.astype(bool)
    # Masking interlopers
    # vel_disp_masked = np.ma.masked_array(vel_disp, mask=maschera).filled(fill_value=0)
    # label_masked = np.ma.masked_array(label, mask=maschera).filled(fill_value=0)
    return maschera


def create_maps_and_zooms(halo_context: HaloContext, subhalo_particles: SubhaloParticles, id_halo: int,
                          id_subhalo: int, spos_rot: np.ndarray, shmr_h: pd.DataFrame, type4: np.ndarray, rotation: str
                          ) -> None:
    """
    Create maps for projections xy, yz, and xz, and save random zooms for each projection.

    Parameters:
    -----------
    halo_context : HaloContext
        The context containing halo information, including r200.
    subhalo_particles : SubhaloParticles
        The particles associated with the subhalo.
    id_halo : int
        The ID of the halo.
    id_subhalo : int
        The ID of the subhalo.
    spos_rot : np.ndarray
        The positions of particles
    shmr_h : pd.DataFrame
        Stellar halo mass ratio for each particle.
    type4 : np.ndarray
        Masking of the particles belonging to icl
    rotation : str
        String with the rotation degree and the iteration until completing 360º

    Returns:
    --------
    None
    """
    logger.info("--> Creating maps for projections xy, yz and xz...")

    # Check if spos_rot is a NumPy array, if not, convert it
    if not isinstance(spos_rot, np.ndarray):
        spos_rot = spos_rot.to_numpy()  # Convert DataFrame to a NumPy array

    result = create_maps(halo_context.r200[id_halo], spos_rot, subhalo_particles.p, subhalo_particles.v, type4)
    label_vel_biny_projections = result["maps"]
    interlopers_id = result["interlopers_id"]
    projections = ["xy", "yz", "xz"]

    for j, (label_i, vel_disp_i, biny_i) in enumerate(label_vel_biny_projections):
        projection = projections[j]
        logger.info(f"---> Masking interlopers for projection {projection}...")
        mask = get_interlopers_mask(label_i, vel_disp_i, spos_rot, shmr_h, interlopers_id, biny_i)
        logger.info("---> Saving three random zooms for icl fraction and velocity "
                    f"dispersion for proyection {projection}...")
        create_n_zooms(label_i, vel_disp_i, idhalo=id_halo, mask=mask,
                       idsubhalo=id_subhalo, projection=projection, n=3, rotation=rotation)


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


def generate_img_dataset(input_dirs: list[str], output_dirs: list[str]) -> None:
    """
    Generate a dataset of images from .npz files in the specified input directories
    and save them as PNG files in the corresponding output directories.

    Parameters:
    -----------
    input_dirs : list[str]
        A list of directories containing the .npz files.
    output_dirs : list[str]
        A list of directories where the generated PNG files will be saved.

    Returns:
    --------
    None
    """
    assert len(input_dirs) == len(output_dirs)

    for i in range(len(input_dirs)):
        input_dir = input_dirs[i]
        output_dir = output_dirs[i]

        # Iterate through all .npz files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.npz'):
                file_path = os.path.join(input_dir, filename)

                # Load the npz file
                with np.load(file_path) as data:
                    for array_name in data.files:
                        array = data[array_name]

                        if array.ndim == 3 and array.shape[2] == 2:  # Ensure the array has 3 dimensions and 2 channels
                            # Select the first channel (or second, based on your needs)
                            image_to_plot = array[:, :, 0]  # Ignore the third dimension
                            # Plot and save as PNG
                            plt.imshow(image_to_plot)
                            plt.axis('off')  # Hide axis
                            png_filename = os.path.splitext(filename)[0] + f'_{array_name}.png'
                            plt.savefig(os.path.join(output_dir, png_filename), bbox_inches='tight', pad_inches=0)
                            plt.close()  # Close the plot to free memory
    print("Conversion complete!")