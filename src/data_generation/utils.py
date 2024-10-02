import illustris_python as il
import numpy as np
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
        star_p = il.snapshot.loadSubhalo(basePath, snapnum, isubhalo,  "star", fields= fields)
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
# of the first subhalo (fsub)
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

        fields = ["SubhaloPos","SubhaloGrNr", "SubhaloHalfmassRadType"]
        halos = il.groupcat.loadSubhalos(basePath, snapnum, fields= fields)
        spos = halos["SubhaloPos"]
        grnr = halos["SubhaloGrNr"]
        shmr = halos["SubhaloHalfmassRadType"][:,4] #stars
        return spos,grnr,shmr
    else:
        halos = il.groupcat.loadSubhalos(basePath, snapnum, fields=fields)
        return halos


# create_maps

def circle(x, y, h, k, r):
    '''
    This function checks if a point (x, y) lies within a circle of radius r
    centered at (h, k). Note: Angles (A) need to be in radians.
    '''
    return ((x - h) ** 2 + (y - k) ** 2) / r ** 2


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
        n /= np.linalg.norm(n)

    j = orig_system[:, -1]

    b = np.cross(j, n)

    theta = angle(j, n)

    q0 = np.cos(theta / 2)

    q1, q2, q3 = np.sin(theta / 2) * b

    Q = np.array([

        [q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],

        [2 * (q2 * q1 - q0 * q3), q0 ** 2 - q1 ** 2 - q3 ** 2 + q2 ** 2, 2 * (q3 * q2 + q0 * q1)],

        [2 * (q3 * q1 + q0 * q2), 2 * (q2 * q3 - q0 * q1), q0 ** 2 - q1 ** 2 + q3 ** 2 - q2 ** 2]

    ])

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
