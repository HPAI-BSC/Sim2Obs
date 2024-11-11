from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.map import create_maps_and_zooms
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.utils import get_icl_mask, logger


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

    logger.debug(f'--> Rotation angle: {angle_degree}')

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


def rotate_particles(subhalo_particles: SubhaloParticles, spos: pd.DataFrame, rotation_degree: float,
                     z: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate particle positions and velocities in 3D space around a specified axis.

    Parameters:
    -----------
    subhalo_particles : SubhaloParticles
        An instance of SubhaloParticles containing particle data.
    spos : pd.DataFrame
        DataFrame of shape (M, 3) representing the positions of M subhalos or particles in 3D space.
    rotation_degree : float
        The angle of rotation around the specified axis, in degrees.
    z : np.ndarray
        A 3-element array representing the original z-axis direction vector before rotation.
    axis : np.ndarray
        A 3-element array representing the axis of rotation in 3D space.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - pos_rot: The rotated positions of the particles, of shape (N, 3).
        - vel_rot: The rotated velocities of the particles, of shape (N, 3).
        - spos_rot: The rotated positions of the subhalos or particles, of shape (M, 3).
    """

    new_z = rotate_vector(z, axis=axis, angle_=rotation_degree)
    Q = get_transform_matrix(new_z, z, normalized=False)
    pos_rot = np.float32(rotate_3dvec(Q, subhalo_particles.p))
    vel_rot = np.float32(rotate_3dvec(Q, subhalo_particles.v))
    spos_rot = np.float32(rotate_3dvec(Q, spos))
    return pos_rot, vel_rot, spos_rot


def apply_rotation(rotation_groups: List, subhalo_particles: SubhaloParticles,
                   halo_context: HaloContext, df_icl: pd.DataFrame, spos_h: pd.DataFrame,
                   shmr_h: pd.DataFrame, id_halo: int, id_subhalo: int) -> None:
    """
    Rotate particle positions and velocities based on specified rotation groups.

    Parameters:
    -----------
    rotation_groups : np.ndarray
        An array containing rotation parameters, such as degrees and rotation axes.
    subhalo_particles : SubhaloParticles
        An instance of SubhaloParticles containing particle data.
    halo_context : HaloContext
        An instance of HaloContext for context-related operations.
    df_icl : pd.DataFrame
        DataFrame containing information about intracluster light.
    spos_h : pd.DataFrame
        Array representing the positions of subhalos or particles.
    shmr_h : pd.DataFrame
        Array representing the subhalo mass ratios.
    id_halo : int
        ID of the halo being processed.
    id_subhalo : int
        ID of the subhalo being processed.

    Returns:
    --------
    None
    """
    logger.info("Rotating positions and velocities ...")
    for row in rotation_groups:
        degree_0, z, rotation_directions = row
        iterations = 360 // degree_0
        logger.debug(f"{iterations} rotations of {degree_0} degrees of axis: {z} over axis:"
                     f" {rotation_directions}")

        for rotation_axis in rotation_directions:
            degree = 0
            for i in range(iterations):
                logger.debug(f'--> Rotation: {i}, degrees: {degree}')
                z = np.array(z)
                rotation_axis = np.array(rotation_axis)
                pos_rot, vel_rot, spos_rot = rotate_particles(subhalo_particles, spos_h, degree, z,
                                                              rotation_axis)
                subhalo_particles_rotated = SubhaloParticles(None, pos_rot, vel_rot, init=False)
                degree += degree_0

                # Create a masking array for BCG and ICL (BCG==0, ICL==1)
                type4 = get_icl_mask(df_icl, subhalo_particles.m)
                create_maps_and_zooms(halo_context, subhalo_particles_rotated, id_halo, id_subhalo,
                                      spos_rot, shmr_h, type4, rotation=f"{degree_0}_{i}")
