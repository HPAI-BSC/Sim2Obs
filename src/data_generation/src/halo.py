import numpy as np

from src.data_generation.src.constants import SOLAR_MASS_CONVERSION


class HaloContext:
    def __init__(self, r200: np.ndarray, m200: np.ndarray, gpos: np.ndarray, fsub: np.ndarray, scale_factor: float):
        """
        Initialize a HaloContext object with scaled halo properties.

        Parameters:
        -----------
        r200 : np.ndarray
            Radii of the halos, scaled by the scale factor.
        m200 : np.ndarray
            Masses of the halos, converted to solar masses.
        gpos : np.ndarray
            Positions of the halos, scaled by the scale factor.
        fsub : np.ndarray
            Indices of the first subhalo in each halo.
        scale_factor : float
            Scaling factor to apply to r200 and gpos.
        """
        self.r200 = r200 * scale_factor  # Apply scaling factor
        self.m200 = m200 * SOLAR_MASS_CONVERSION          # Convert halo masses to solar masses
        self.gpos = gpos * scale_factor   # Apply scaling factor to halo positions
        self.fsub = fsub
        self.halo_mass_min = None         # To be set later

