import numpy as np


class HaloContext:
    def __init__(self, r200: np.ndarray, m200: np.ndarray, gpos: np.ndarray, fsub: np.ndarray, scale_factor: float,
                 halo_id: np.ndarray):
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
        halo_id: np.ndarray
            Halo identifiers
        """
        self.r200 = r200 * scale_factor   # Apply scaling factor
        self.m200 = m200                  # Solar masses
        self.gpos = gpos * scale_factor   # Apply scaling factor to halo positions
        self.fsub = fsub.astype(int)
        self.halo_id = halo_id

    def get_item(self, halo_id):
        position = np.where(self.halo_id == halo_id)
        if len(position) - 1 > 0:
            raise Exception('Error, there are repeated halo ids')
        else:
            position = position[0]
            if position < len(self.r200):
                return self.r200[position], self.m200[position], self.gpos[position], self.fsub[position][0]
            else:
                raise Exception(f'Error, the halo_id={halo_id} could not be found.')




