import numpy as np

from src.data_generation.src.constants import SOLAR_MASS_CONVERSION


class SubhaloContext:
    def __init__(self, spos: np.ndarray, grnr: np.ndarray, shmr: np.ndarray,
                 mstr: np.ndarray, scale_factor: float, hubble_constant: float) -> None:
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
        hubble_constant : float
            The Hubble constant used for mass conversion.
        """
        self.spos = spos * scale_factor          # Apply scaling factor to subhalo positions
        self.shmr = shmr * scale_factor           # Apply scaling factor to shmr
        self.mstr = mstr * 1e10 / hubble_constant  # Convert to solar masses
        self.id_parents = grnr
        self.subhalo_min_mass = None              # To be set later


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
            self.m = m * SOLAR_MASS_CONVERSION
            self.p = p * scale_factor
            self.v = v * np.sqrt(scale_factor)
        else:
            self.m = m
            self.p = p
            self.v = v




