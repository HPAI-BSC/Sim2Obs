import numpy as np
from scipy import stats

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.utils import logger


def create_maps(r200_halo: float, spos_rot: np.ndarray, pos_rot: np.ndarray, vel_rot: np.array, type4: np.array):
    """
    Create 2D maps for particle properties in a halo based on their positions and velocities.

    This function generates three different maps (X-Y, Y-Z, and X-Z projections) that display the
    velocity dispersion of particles and the fraction of intracluster light (ICL) in each region of
    the 2D space. The function also identifies interloper subhalos based on their positions within
    a defined range.

    Parameters:
    -----------
    r200_halo : float
        The R200 radius of the halo, which is used to define the vertical extent of the interloper
        identification.
    spos_rot : numpy.ndarray
        An array of shape (N, 3) representing the rotated positions of subhalos in 3D space, where N
        is the number of subhalos.
    pos_rot : numpy.ndarray
        An array of shape (M, 3) representing the rotated positions of particles in the primary subhalo,
        where M is the number of particles.
    vel_rot : numpy.ndarray
        An array of shape (M, 3) representing the rotated velocities of particles in the primary
        subhalo, corresponding to the positions in pos_rot.
    type4 : numpy.ndarray
        An array indicating the type of particles in the primary subhalo, where particles classified
        as ICL are identified.

    Returns:
    --------
    list
        A list containing three tuples, each representing a map for the projections:
        - Each tuple consists of:
            - label_map: The fraction of ICL particles in the corresponding projection.
            - vel_disp: The velocity dispersion of particles in the corresponding projection.
            - bins: The bin edges used for creating the maps.
    numpy.ndarray
        An array containing the indices of interloper subhalos found within the defined spatial range.
    """
    # r200_halo, spos_rot (posicion of subhalos), pos_rot and vel_rot position and velocity of particles in the
    # primary subhalo, type4 particles of primary subhalo that are ICL

    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
    ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

    # Considering as interloper any subhalo contained in a box defined by [-r, r] in the x and y directions and
    # inside [-R_200, R_200] along the z axis

    interlopers_id = \
    np.where((np.abs(spos_rot[:, 0]) < r) & (np.abs(spos_rot[:, 1]) < r) & (np.abs(spos_rot[:, 2]) < r200_halo))[0]

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


def create_random_zoom(label_masked, vel_disp_masked):
    """
    Creates a random zoom of the masked labels and velocity dispersion maps.

    This function selects a random central point in the 2D masked maps of labels
    and velocity dispersion, and returns a smaller sub-region around that point
    defined by a fixed size based on the number of bins calculated from the MUSE
    observations.

    Parameters:
    ----------
    label_masked : ndarray
        A 2D array representing the masked labels of the particles or regions of interest.

    vel_disp_masked : ndarray
        A 2D array representing the masked velocity dispersion of the particles or regions of interest.

    Returns:
    -------
    tuple
        A tuple containing:
        - label_masked1 : ndarray
            A smaller 2D array of masked labels centered around a random point.
        - vel_disp_masked1 : ndarray
            A smaller 2D array of masked velocity dispersion centered around the same random point.
    """
    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
    xo, yo = (int(bins / 2), int(bins / 2))
    xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)
    vel_disp_masked1 = vel_disp_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                       yoprime - int(bins / 4): yoprime + int(bins / 4)]
    label_masked1 = label_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                    yoprime - int(bins / 4): yoprime + int(bins / 4)]
    return label_masked1, vel_disp_masked1


def create_three_zoom(label_masked, vel_disp_masked, **kwargs):
    """
    Creates and saves three random zooms of the masked labels and velocity dispersion maps.

    This function generates three random sub-regions of the masked label and velocity
    dispersion maps, saving each zoomed-in region to separate files. The file names
    are constructed based on halo and subhalo identifiers, snapshot number, and projection.

    Parameters:
    ----------
    label_masked : ndarray
        A 2D array representing the masked labels of the particles or regions of interest.

    vel_disp_masked : ndarray
        A 2D array representing the masked velocity dispersion of the particles or regions of interest.

    **kwargs : keyword arguments
        Additional parameters required for file naming:
        - idhalo : int
            Identifier for the halo.
        - idsubhalo : int
            Identifier for the subhalo.
        - projection : str
            Projection type used in the analysis.
        - snapnum : int
            Snapshot number for the simulation.

    Returns:
    -------
    None
        The function saves the zoomed maps as `.npz` files and does not return any value.
    """
    idhalo = kwargs["idhalo"]
    idsubhalo = kwargs["idsubhalo"]
    projection = kwargs["projection"]
    snapnum = kwargs["snapnum"]
    mask = kwargs["mask"]
    label_masked1, vel_disp_masked1 = create_random_zoom(label_masked, vel_disp_masked)
    label_masked2, vel_disp_masked2 = create_random_zoom(label_masked, vel_disp_masked)
    label_masked3, vel_disp_masked3 = create_random_zoom(label_masked, vel_disp_masked)

    NAME_TO_SAVE = f"../data/npz/velocity_dispersion/halo-{idhalo}_subhalo-{idsubhalo}_snap{snapnum}_{projection}_"
    np.savez(NAME_TO_SAVE + f"xyz-zoom1", vel_disp_masked1)
    np.savez(NAME_TO_SAVE + f"xyz-zoom2", vel_disp_masked2)
    np.savez(NAME_TO_SAVE + f"xyz-zoom3", vel_disp_masked3)

    NAME_TO_SAVE = f"../data/npz/labels/halo-{idhalo}_subhalo-{idsubhalo}_snap{snapnum}_{projection}_"
    np.savez(NAME_TO_SAVE + f"xyz-zoom1", label_masked1)
    np.savez(NAME_TO_SAVE + f"xyz-zoom2", label_masked2)
    np.savez(NAME_TO_SAVE + f"xyz-zoom3", label_masked3)


def get_interlopers_mask(label, vel_disp, spos_rot, shmr, interlopers_id, biny):
    """
    Create a mask to identify and exclude interloper subhalos from calculations.

    Interlopers are subhalos that appear close to a primary subhalo but are not physically associated
    with it. This function generates a mask that excludes these interloper subhalos from the provided
    label and velocity dispersion arrays based on their spatial positions and radii.

    Parameters:
    ----------
    label : ndarray
        A 2D array representing the labels of the particles or regions of interest.

    vel_disp : ndarray
        A 2D array representing the velocity dispersion of the particles in the primary subhalo.

    spos_rot : ndarray
        A 2D array containing the spatial positions of each subhalo.

    shmr : DataFrame
        A DataFrame containing the SubhaloHalfmassRadType of subhalos within the current halo.

    interlopers_id : list or ndarray
        An array or list of identifiers for the interloper subhalos.

    biny : ndarray
        An array representing the y-coordinates for the masking grid.

    Returns:
    -------
    tuple
        A tuple containing:
        - label_masked : ndarray
            A masked array of the labels, with interloper regions excluded.
        - vel_disp_masked : ndarray
            A masked array of the velocity dispersions, with interloper regions excluded.
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


def create_maps_and_zooms(halo_context: HaloContext, subhalo_particles: SubhaloParticles, id_halo, id_subhalo, spos_rot,
                          shmr_h, type4, snapnum):
    logger.info("--> Creating maps for projections xy, yz and xz...")
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
        create_three_zoom(label_i, vel_disp_i, idhalo=id_halo, mask=mask,
                          idsubhalo=id_subhalo, projection=projection, snapnum=snapnum)


def circle(x, y, h, k, r):
    """
    Check if a point (x, y) lies within a circle defined by its center (h, k)
    and radius r.

    This function determines the relative position of a point to a circle.
    If the point lies within or on the boundary of the circle, it will return a value
    indicating that condition. Angles (if used) should be in radians.

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


def muse_bins_r(fov, spatial_resolution):
    """
    Calculate the number of bins and the radius for MUSE observations.

    This function computes the number of bins required to cover a specified
    field of view (fov) based on a given spatial resolution. It also calculates
    the radius of the field of view, which is half of the fov.

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
