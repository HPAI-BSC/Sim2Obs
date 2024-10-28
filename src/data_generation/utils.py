import illustris_python as il
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine

# Retrieve particle data for a specific halo (coordinates, velocities, masses)
def get_p_halo(basePath, snapnum, ihalo, fields=None):
    '''GET PARTICLE DATA IN HALO'''
    '''isubhalo needs to be the first subhalo'''
    if fields is None:

        fields = ["Coordinates", "Velocities", "Masses"]
        star_p = il.snapshot.loadHalo(basePath, snapnum, ihalo, "star", fields=fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"]
        m = star_p["Masses"]
        return m, p, v
    else:
        print("Non usato field")
        star_p = il.groupcat.loadHalo(basePath, snapnum, ihalo, fields=fields)
        print(star_p.shape)
        return star_p


# Retrieve particle data for a specific subhalo (coordinates, velocities, masses)
def get_p_subhalo(basePath, snapnum, isubhalo, fields=None):
    ''' GET PARTICLE DATA IN SUBHALO isubhalo needs  to be the first subhalo '''
    if fields is None:

        fields = ["Coordinates", "Velocities", "Masses"]
        # TODO filter them by mass
        star_p = il.snapshot.loadSubhalo(basePath, snapnum, isubhalo,  "star", fields=fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"]
        m = star_p["Masses"]
        return m, p, v
    else:
        print ("Non usato field")
        star_p = il.groupcat.loadSubhalos(basePath, snapnum, isubhalo, fields=fields)
        print (star_p.shape)
        return star_p


# Retrieve global halo properties like mass (m200), radius (r200) and position (gpos) and index
# of the first subhalo (fsub).
# Load all halo information from the entire group catalog for one snapshot
def get_halos(basePath, snapnum, fields=None):
    '''GET HALO DATA'''
    if fields is None:

        fields = ["Group_M_Crit200","Group_R_Crit200","GroupPos", "GroupFirstSub"]
        halos = il.groupcat.loadHalos(basePath, snapnum, fields= fields)
        r200 = halos["Group_R_Crit200"]
        m200 = halos["Group_M_Crit200"]
        gpos = halos["GroupPos"]
        fsub = halos["GroupFirstSub"]

        return r200,m200,gpos,fsub
    else:
        halos = il.groupcat.loadHalos(basePath, snapnum, fields=fields)
        return halos


# Retrieve subhalo properties like position (spos), group number the subhalo belongs to (grnr) and
# stellar half-mass radius (shmr)
def get_subhalos(basePath, snapnum, fields=None):
    '''GET SUBHALO DATA'''
    if fields is None:

        fields = ["SubhaloPos","SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"]
        halos = il.groupcat.loadSubhalos(basePath, snapnum, fields=fields)
        spos = halos["SubhaloPos"]
        grnr = halos["SubhaloGrNr"]
        shmr = halos["SubhaloHalfmassRadType"][:,4] #stars
        mstr = halos["SubhaloMassType"][:,4]
        return spos,grnr,shmr,mstr
    else:
        halos = il.groupcat.loadSubhalos(basePath, snapnum, fields=fields)
        return halos


# create_maps

def circle(x, y, h, k, r):
    '''
    This function checks if a point (x, y) lies within a circle of radius r
    centered at (h, k). Note: Angles (A) need to be in radians.
    '''
    tmp = ((x - h) ** 2 + (y - k) ** 2) / r ** 2
    return tmp


def find_bin(x, r, Nbin):
    '''
    Maps a 1D coordinate in physical space to its corresponding bin number.
    This is useful for creating histograms or 2D maps.
    '''
    delta_r = 2 * r / Nbin  # Width of each bin
    x_prime = x + r  # Shift x to the range [0, 2r]
    return (x_prime / delta_r).astype(int)  # Return bin index


# Function to calculate the number of bins and radius for MUSE observations
def muse_bins_r(fov, spatial_resolution):
    '''
    Calculates the number of bins and the radius for MUSE observations based on
    the field of view (fov) and spatial resolution (both in kpc).
    '''
    bins = int(fov / spatial_resolution)  # Number of bins along one axis
    r = fov / 2  # Radius of the field of view
    return bins, r


def angle(u,v):
    return np.arccos(1 - cosine(u,v))


def get_transform_matrix(orig_system, n, normalized=True):
    '''
    Function that gives the rotation matrix of a given system of reference provided the vector we want the new z-axis to point.
    See https://stackoverflow.com/questions/70271242/3d-vectors-align-a-vectors-coordinate-system-to-a-normal for reference.

    input:
    : orig_system : original system of reference, should be np.eye(3)
    : n : normalized vector representing the new z-axis
    : normalized: (bool) whether the n vector is normalized or not.
    returns:
    : Q : rotation matrix

    '''

    if not normalized:
        # unit vector
        n /= np.linalg.norm(n)

    j = orig_system[:, -1]

    # Orthogonal vector to original z-axis and the new z-axis
    b = np.cross(j, n)
    theta = angle(j, n) / 2  # 45 degrees
    print(f'  rotation angle: {theta * (180 / np.pi)}')

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

    # Sanity checks
    is_orthogonal = np.allclose(np.dot(Q.T, Q), np.eye(3))
    print(f"Is Q orthogonal? {is_orthogonal}")

    det_Q = np.linalg.det(Q)
    print(f"Determinant of Q: {det_Q}")
    is_det_one = np.isclose(det_Q, 1.0)
    print(f"Is the determinant of Q equal to 1? {is_det_one}")
    q_norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)

    return Q


def rotate_3dvec(Q, vec):
    '''
    This function applies the rotation to a 3dimensional vector, such as POS from Gadget.
    input:
    : Q : rotational matrix
    : vec : vector to apply
    returns
    : rotated vector :
    '''
    return (Q @ vec.T).T


def label_particles(r, modulo_v, R200, M200):
    final_model = joblib.load('model/Best_Model-Random-3par.pkl')
    # Define simulation parameters
    G = 4.30091e-6  # Gravitational constant in units of kpc Msun^-1 (km/s)^2

    V200 = np.sqrt(G * M200 / R200)

    # Create a DataFrame with features required by the model
    # Features are: log-scaled distance (r/R200), scaled velocity (modulo_v/V200), and log-scaled halo mass (M200)
    tmp = r.size - np.count_nonzero(r)
    Data = pd.DataFrame({
        'distance': np.log10(r / R200),  # Log-scaled distance
        'velocity': modulo_v / V200,  # Scaled velocity
        'm200': [np.log10(M200 * 1e-14)] * len(modulo_v)  # Log-scaled halo mass (m200, constant for all particles)
    })

    # Replace any infinite values in the DataFrame with NaN, then drop all rows with NaN values
    Data = Data.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    # Create an array of particle indices (0 to len(r)-1)
    IDS = np.linspace(0, len(r) - 1, len(r), dtype=int)

    # Use the trained model to make predictions on the data
    # The model predicts whether each particle belongs to BCG (label 1) or ICL (label 0)
    final_predictions = final_model.predict(Data.to_numpy())

    # Extract indices of particles predicted to be part of the BCG
    ids_BCG = IDS[np.where(final_predictions == 1)[0]]

    # Extract indices of particles predicted to be part of the ICL
    ids_ICL = IDS[np.where(final_predictions == 0)[0]]
    df_bcg = pd.DataFrame(ids_BCG)
    df_icl = pd.DataFrame(ids_ICL)
    return df_bcg, df_icl


def rotate_particles(p, v, spos, rotation_vector):
    orig_system = np.eye(3)  # Original coordinate system (identity matrix)
    # Vectors for rotating the system (axes of rotation)
    # vectors = [(0, 0, 1), (0, 1, 0)]  # Rotation vectors (e.g., z-axis and y-axis)

    n = np.array(rotation_vector, dtype=np.float32)
    Q = get_transform_matrix(orig_system, n, normalized=False)
    pos_rot = np.float32(rotate_3dvec(Q, p))
    vel_rot = np.float32(rotate_3dvec(Q, v))
    spos_rot = np.float32(rotate_3dvec(Q, spos))
    return pos_rot, vel_rot, spos_rot


def create_maps(r200_halo, spos_rot, pos_rot, vel_rot, type4):
    # r200_halo, spos_rot (posicion of subhalos), pos_rot and vel_rot position and velocity of particles in the
    # primary subhalo, type4 particles of primary subhalo that are ICL

    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
    ranging = ((-r, r), (-r, r))  # Defines the range for the 2D maps

    # Considering as interloper any subhalo contained in a box defined by [-r, r] in the x and y directions and
    # inside [-R_200, R_200] along the z axis

    interlopers_id = \
    np.where((np.abs(spos_rot[0]) < r) & (np.abs(spos_rot[1]) < r) & (np.abs(spos_rot[2]) < r200_halo))[0]

    # removing first galaxy (central/BCG), since it's usually the central galaxy or BCG
    # TODO is it possible to have no interlopers ?
    interlopers_id = interlopers_id[1:]

    if len(interlopers_id) == 0:
        print(' There is no interlopers.')

    # Create three different maps based on projections onto different coordinate planes (X-Y, Y-Z, X-Z).
    # Break up the 2D spatial plane into small regions (bins), compute the mean of the squared velocities
    # in each region, and return the resulting grid of values.

    # Velocity dispersion for X-Y projection
    vel_disp_xy, binx_xy, biny_xy, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 1], (vel_rot[:, 2]) ** 2,
                                                     statistic="mean", bins=bins, range=ranging)
    vel_disp_xy = np.sqrt(vel_disp_xy / 3)

    # Velocity dispersion for Y-Z projection
    vel_disp_yz, binx_yz, biny_yz, _ = stats.binned_statistic_2d(pos_rot[:, 1], pos_rot[:, 2], (vel_rot[:, 0]) ** 2,
                                                     statistic="mean", bins=bins, range=ranging)
    vel_disp_yz = np.sqrt(vel_disp_yz / 3)

    # Velocity dispersion for X-Z projection
    vel_disp_xz, binx_xz, biny_xz, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 2], (vel_rot[:, 1]) ** 2,
                                                     statistic="mean", bins=bins, range=ranging)
    vel_disp_xz = np.sqrt(vel_disp_xz / 3)

    # ICL fraction for X-Y projection
    label_xy, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 1], type4,
                                                  statistic="sum", bins=bins, range=ranging)
    label_xy_count, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 1], type4,
                                                        statistic="count", bins=bins, range=ranging)
    label_xy = label_xy / label_xy_count

    # ICL fraction for Y-Z projection
    label_yz, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 1], pos_rot[:, 2], type4,
                                                  statistic="sum", bins=bins, range=ranging)
    label_yz_count, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 1], pos_rot[:, 2], type4,
                                                        statistic="count", bins=bins, range=ranging)
    label_yz = label_yz / label_yz_count

    # ICL fraction for X-Z projection
    label_xz, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 2], type4,
                                                  statistic="sum", bins=bins, range=ranging)
    label_xz_count, _, _, _ = stats.binned_statistic_2d(pos_rot[:, 0], pos_rot[:, 2], type4,
                                                        statistic="count", bins=bins, range=ranging)
    label_xz = label_xz / label_xz_count

    # Handling NaN values in the maps
    label_xy[np.isnan(label_xy)] = 0
    vel_disp_xy[np.isnan(vel_disp_xy)] = 0

    label_yz[np.isnan(label_yz)] = 0
    vel_disp_yz[np.isnan(vel_disp_yz)] = 0

    label_xz[np.isnan(label_xz)] = 0
    vel_disp_xz[np.isnan(vel_disp_xz)] = 0
    return [(label_xy, vel_disp_xy, binx_xy),
            (label_yz, vel_disp_yz, binx_yz),
            (label_xz, vel_disp_xz, biny_xz)], interlopers_id


def create_random_zoom(label_masked, vel_disp_masked):
    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius
    xo, yo = (int(bins / 2), int(bins / 2))
    xoprime, yoprime = xo + np.random.randint(-20, 21), yo + np.random.randint(-20, 21)
    vel_disp_masked1 = vel_disp_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                       yoprime - int(bins / 4): yoprime + int(bins / 4)]
    label_masked1 = label_masked[xoprime - int(bins / 4):xoprime + int(bins / 4),
                    yoprime - int(bins / 4): yoprime + int(bins / 4)]
    return label_masked1, vel_disp_masked1


def mask_interlopers(label, vel_disp, spos_rot, shmr, interlopers_id, biny):
    '''
     create a mask to identify and exclude "interloper" subhalos from calculations. Interlopers are subhalos that
      appear close but aren’t physically associated with a primary subhalo. This function aims to mask out interloper
       subhalos from label and vel_disp arrays using spatial position and radius.
    :param label:
    :param vel_disp: velocity dispersion of the particles in the primary subhalo
    :param spos_rot: position of each subhalo
    :param shmr: SubhaloHalfmassRadType of subhalos inside current halo
    :param interlopers_id: id of interlopers in the context of subhalos
    :param biny:
    :return:
    '''
    bins, r = muse_bins_r(2 * 263.34, 2.02)  # Computes bins and radius for the grid
    maschera = np.zeros_like(label)

    X, Y = np.meshgrid(np.linspace(-r, r, bins), biny)  # Create a grid of X, Y coordinates

    '''
    for i in interlopers_id:
        h = spos_rot[i, 0]
        k = spos_rot[i, 1]
        rradius = shmr.iloc[i, 0]
        for cc, x in enumerate(np.linspace(-r, r, bins)):
            try:
                maschera[cc, np.where(circle(x, biny, h, k, rradius) <= 1)[0]] = 1
            except IndexError:  # catch the error
                pass
    '''
    for i in interlopers_id:
        # For each interloper get its center position (h, k) and its radius
        h = spos_rot[i, 0]
        k = spos_rot[i, 1]
        rradius = shmr.iloc[i, 0]
        for cc in range(len(X)):
            for dd in range(Y.shape[1]):
                # Check if the points in the grid falls within the interloper's radius
                if circle(X[cc, dd], Y[cc, dd], h, k, rradius) <= 1:
                    if cc < maschera.shape[0] and dd < maschera.shape[1]:
                        # if the point is within the radius, mask is updated to mark this region as part of the
                        # interloper
                        maschera[cc, dd] = 1
                    else:
                        print(f"Warning: Index out of bounds - cc: {cc}, dd: {dd}")

    # Check if the mask is just zero
    if np.all(maschera == 0):
        print(' Mask is full of zeros.')

    # Masking interlopers
    vel_disp_masked = np.ma.masked_array(vel_disp, mask=maschera).filled(fill_value=0)
    label_masked = np.ma.masked_array(label, mask=maschera).filled(fill_value=0)
    return label_masked, vel_disp_masked
