import illustris_python as il
import numpy as np

from src.data_generation.src.constants import SOLAR_MASS_CONVERSION, SNAPNUM, SCALF, H


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


def get_subhalos(base_path: str, fields: list[str] = None) -> SubhaloContext:
    """
    Retrieve subhalo properties such as position, group number, and stellar half-mass radius.

    This function loads subhalo information from the specified snapshot of a simulation.
    If no fields are specified, it defaults to retrieving SubhaloPos, SubhaloGrNr,
    SubhaloHalfmassRadType, and SubhaloMassType.

    Parameters:
    -----------
    base_path : str
        The base directory path to the simulation data.
    fields : list[str], optional
        A list of field names to retrieve. If not specified, defaults to:
        ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"].

    Returns:
    --------
    SubhaloContext | dict
        If `fields` is not provided, returns a SubhaloContext object containing:
        - spos : np.ndarray
            The positions of the subhalos.
        - grnr : np.ndarray
            The group numbers that each subhalo belongs to.
        - shmr : np.ndarray
            The stellar half-mass radius for each subhalo.
        - mstr : np.ndarray
            The mass of stars in each subhalo.
        If `fields` is provided, returns a dictionary of subhalo data corresponding to the specified fields.
    """
    if fields is None:

        fields = ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"]
        halos = il.groupcat.loadSubhalos(base_path, SNAPNUM, fields=fields)
        spos = halos["SubhaloPos"]
        grnr = halos["SubhaloGrNr"]
        shmr = halos["SubhaloHalfmassRadType"][:, 4] #stars
        mstr = halos["SubhaloMassType"][:, 4]
        return SubhaloContext(spos, grnr, shmr, mstr, SCALF, H)

    else:
        halos = il.groupcat.loadSubhalos(base_path, SNAPNUM, fields=fields)
        return halos


def get_p_subhalo(base_path: str, isubhalo: int, fields: list[str] = None) -> SubhaloParticles:
    """
    Retrieve particle data for a specific subhalo, including coordinates, velocities, and masses.

    This function retrieves particle data from a subhalo in a simulation snapshot.
    If no fields are specified, it defaults to retrieving coordinates, velocities,
    and masses of star particles. The subhalo specified by `isubhalo` must be the
    first subhalo.

    Parameters:
    -----------
    base_path : str
        The base directory path to the simulation data.
    isubhalo : int
        The index of the subhalo from which to retrieve particle data. This must be the first subhalo.
    fields : list[str], optional
        A list of field names to retrieve. If not specified, defaults to retrieving
        ["Coordinates", "Velocities", "Masses"].

    Returns:
    --------
    SubhaloParticles | dict
        If `fields` is not provided, returns a SubhaloParticles object containing:
        - m : np.ndarray
            The masses of the particles in the subhalo.
        - p : np.ndarray
            The coordinates of the particles in the subhalo.
        - v : np.ndarray
            The velocities of the particles in the subhalo.
        If `fields` is provided, returns a dictionary of particle data corresponding to the specified fields.
    """
    if fields is None:
        fields = ["Coordinates", "Velocities", "Masses"]
        star_p = il.snapshot.loadSubhalo(base_path, SNAPNUM, isubhalo,  "star", fields=fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"]
        m = star_p["Masses"]

        return SubhaloParticles(m, p, v, SCALF)
    else:
        star_p = il.groupcat.loadSubhalos(base_path, SNAPNUM, isubhalo, fields=fields)
        return star_p
