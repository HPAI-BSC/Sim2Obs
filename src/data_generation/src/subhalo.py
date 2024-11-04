import illustris_python as il
import numpy as np

from src.data_generation.src.constants import SOLAR_MASS_CONVERSION, SNAPNUM, SCALF, H


class SubhaloContext:
    def __init__(self, spos, grnr, shmr, mstr, scale_factor, hubble_constant):
        self.spos = spos * scale_factor          # Apply scaling factor to subhalo positions
        self.shmr = shmr * scale_factor           # Apply scaling factor to shmr
        self.mstr = mstr * 1e10 / hubble_constant  # Convert to solar masses
        self.id_parents = grnr
        self.subhalo_min_mass = None              # To be set later


class SubhaloParticles:
    def __init__(self, m, p, v, scale_factor=None, init=True):
        if init:
            self.m = m * SOLAR_MASS_CONVERSION
            self.p = p * scale_factor
            self.v = v * np.sqrt(scale_factor)
        else:
            self.m = m
            self.p = p
            self.v = v


def get_subhalos(base_path: str, fields=None):
    """
    Retrieve subhalo properties such as position (SubhaloPos), group number (SubhaloGrNr),
    and stellar half-mass radius (SubhaloHalfmassRadType).

    This function loads subhalo information from the specified snapshot of a simulation.
    If no fields are specified, it defaults to retrieving SubhaloPos, SubhaloGrNr,
    SubhaloHalfmassRadType, and SubhaloMassType.

    Parameters:
    -----------
    basePath : str
        The base directory path to the simulation data.
    snapnum : int
        The snapshot number in the simulation from which to load subhalo data.
    fields : list of str, optional
        A list of field names to retrieve. If not specified, the function defaults to:
        ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"].

    Returns:
    --------
    tuple
        If `fields` is not provided, returns a tuple containing:
        - spos : array
            The positions of the subhalos.
        - grnr : array
            The group numbers that each subhalo belongs to.
        - shmr : array
            The stellar half-mass radius for each subhalo.
        - mstr : array
            The mass of stars in each subhalo.
        If `fields` is provided, returns an array of subhalo data corresponding to the specified fields.
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


def get_p_subhalo(base_path: str, isubhalo: int, fields=None):
    """
    Retrieve particle data for a specific subhalo, including coordinates, velocities, and masses.

    This function retrieves particle data from a subhalo in a simulation snapshot.
    If no fields are specified, it defaults to retrieving coordinates, velocities,
    and masses of star particles. The subhalo specified by `isubhalo` must be the
    first subhalo.

    Parameters:
    -----------
    basePath : str
        The base directory path to the simulation data.
    snapnum : int
        The snapshot number in the simulation from which to load subhalo data.
    isubhalo : int
        The index of the subhalo from which to retrieve particle data.
        This must be the first subhalo.
    fields : list of str, optional
        A list of field names to retrieve. If not specified, the function defaults
        to retrieving ["Coordinates", "Velocities", "Masses"].

    Returns:
    --------
    tuple
        If `fields` is not provided, returns a tuple containing:
        - m : array
            The masses of the particles in the subhalo.
        - p : array
            The coordinates of the particles in the subhalo.
        - v : array
            The velocities of the particles in the subhalo.
        If `fields` is provided, returns an array of particle data corresponding to
        the specified fields.
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
