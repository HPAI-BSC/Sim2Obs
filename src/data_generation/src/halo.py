import illustris_python as il

from src.data_generation.src.constants import SOLAR_MASS_CONVERSION, SNAPNUM, SCALF


class HaloContext:
    def __init__(self, r200, m200, gpos, fsub, scale_factor):
        self.r200 = r200 * scale_factor  # Apply scaling factor
        self.m200 = m200 * SOLAR_MASS_CONVERSION          # Convert halo masses to solar masses
        self.gpos = gpos * scale_factor   # Apply scaling factor to halo positions
        self.fsub = fsub
        self.halo_mass_min = None         # To be set later


def get_halos(base_path: str, fields=None):
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
    tuple
        If `fields` is not provided, returns a tuple containing:
        - r200 : array
            The radii of the halos (R200).
        - m200 : array
            The masses of the halos (M200).
        - gpos : array
            The positions of the halos.
        - fsub : array
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


def get_p_halo(base_path, snapnum, ihalo, fields=None):
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
        A list of field names to retrieve. By default, ['Coordinates', 'Velocities', 'Masses'] is used.

    Returns:
    --------
    tuple
        Returns a tuple of arrays `(m, p, v)` with masses, coordinates, and velocities of particles in the halo.
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
