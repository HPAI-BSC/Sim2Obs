from typing import Optional, List, Union, Tuple

import illustris_python as il
import numpy as np

from src.data_generation.src.constants import SOLAR_MASS_CONVERSION, SNAPNUM, SCALF


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


def get_halos(base_path: str, fields: Optional[List[str]] = None) -> HaloContext:
    """
    Retrieve global halo properties such as mass (M200), radius (R200), position (GroupPos),
    and index of the first subhalo (GroupFirstSub).

    This function loads halo information from the entire group catalog for a specified snapshot.
    If no fields are specified, it defaults to retrieving M200, R200, GroupPos, and GroupFirstSub.

    Parameters:
    -----------
    base_path : str
        The base directory path to the simulation data.
    fields : list of str, optional
        A list of field names to retrieve. If not specified, the function defaults to:
        ["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"].

    Returns:
    --------
    HaloContext
        If `fields` is not provided, returns a `HaloContext` object containing the following attributes:
            - r200 : np.ndarray
                The radii of the halos (R200).
            - m200 : np.ndarray
                The masses of the halos (M200).
            - gpos : np.ndarray
                The positions of the halos.
            - fsub : np.ndarray
                The index of the first subhalo in each halo.
        If `fields` is provided, returns an array of halo data corresponding to the specified fields.
    """
    if fields is None:

        fields = ["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"]
        halos = il.groupcat.loadHalos(base_path, SNAPNUM, fields=fields)
        r200 = halos["Group_R_Crit200"]
        m200 = halos["Group_M_Crit200"]
        gpos = halos["GroupPos"]
        fsub = halos["GroupFirstSub"]

        return HaloContext(r200, m200, gpos, fsub, SCALF)
    else:
        halos = il.groupcat.loadHalos(base_path, SNAPNUM, fields=fields)
        return halos


def get_p_halo(base_path: str, snapnum: int, ihalo: int, fields: Optional[List[str]] = None) \
        -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Retrieve particle data for a specific halo, such as coordinates, velocities, and masses.

    Parameters:
    -----------
    base_path : str
       The base directory path to the simulation data.
    snapnum : int
       The snapshot number in the simulation from which to load halo data.
    ihalo : int
       The halo ID number (or index) from which to retrieve particle data.
    fields : list of str, optional
       A list of field names to retrieve. By default, ["Coordinates", "Velocities", "Masses"] is used.

    Returns:
    --------
    Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
       If `fields` is not provided, returns a tuple `(m, p, v)`:
           - m : np.ndarray
               Masses of the particles in the halo.
           - p : np.ndarray
               Coordinates of the particles in the halo.
           - v : np.ndarray
               Velocities of the particles in the halo.
       If `fields` is provided, returns an array of particle data corresponding to the specified fields.
   """

    if fields is None:

        fields = ["Coordinates", "Velocities", "Masses"]
        star_p = il.snapshot.loadHalo(base_path, snapnum, ihalo, "star", fields=fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"]
        m = star_p["Masses"]
        return m, p, v
    else:
        star_p = il.groupcat.loadHalo(base_path, snapnum, ihalo, fields=fields)
        return star_p
