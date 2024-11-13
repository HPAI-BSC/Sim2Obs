import numpy as np

from src.data_generation.src.constants import SOLAR_MASS_CONVERSION


class SubhaloContext:
    def __init__(self, spos: np.ndarray, grnr: np.ndarray, shmr: np.ndarray,
                 mstr: np.ndarray, scale_factor: float, subhalo_id: np.ndarray) -> None:
        """
        Initialize the SubhaloContext with properties of subhalos.

        Parameters:
        -----------
        spos : np.ndarray
            The positions of subhalos. Array should be of shape (N, 3) where N is the number of subhalos.
        grnr : np.ndarray
            The group numbers corresponding to each subhalo.
        shmr : np.ndarray
            The stellar half-mass radius for each subhalo.
        mstr : np.ndarray
            The mass of stars in each subhalo.
        scale_factor : float
            The scale factor to apply to subhalo positions and other properties.
        subhalo_id: np.ndarray
            List with the subhalo identifiers
        """
        self.spos = spos * scale_factor           # Apply scaling factor to subhalo positions
        self.shmr = shmr * scale_factor           # Apply scaling factor to shmr
        self.mstr = mstr                          # Solar masses
        self.id_parents = grnr
        self.subhalo_id = subhalo_id

    def get_item(self, subhalo_id):
        position = np.where(self.subhalo_id == subhalo_id)
        if len(position) - 1 > 0:
            raise Exception('There are repeated subhalo ids')
        else:
            position = position[0]
            if position < len(self.spos):
                return self.spos[position], self.shmr[position], self.mstr[position]
            else:
                raise Exception('Error, the subhalo_id could not be found.')

    def get_items_by_halo(self, halo_id):
        positions = np.where(self.id_parents == halo_id)
        if len(positions) > 0:
            return self.spos[positions], self.shmr[positions], self.mstr[positions]
        else:
            return [], [], []


class SubhaloParticles:
    def __init__(self, m: np.ndarray, p: np.ndarray, v: np.ndarray,
                 scale_factor: float = None, init: bool = True) -> None:
        """
        Initialize the SubhaloParticles with mass, position, and velocity data.

        Parameters:
        -----------
        m : np.ndarray
            The mass of particles.
        p : np.ndarray
            The positions of particles. Should be of shape (N, 3) where N is the number of particles.
        v : np.ndarray
            The velocities of particles. Should also be of shape (N, 3).
        scale_factor : float, optional
            The scale factor to apply to positions and velocities. Defaults to None.
        init : bool, optional
            If True, initializes the attributes with conversion. Defaults to True.
        """
        if init:
            self.m = m
            self.p = p * scale_factor
            self.v = v
        else:
            self.m = m
            self.p = p
            self.v = v




