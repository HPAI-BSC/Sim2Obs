from abc import ABC, abstractmethod
from dotenv import load_dotenv
import eagleSqlTools as sql
import numpy as np
import pandas as pd
import illustris_python as il

from src.data_generation.src.constants import SCALF, H, HALO_MIN_MASS, SUBHALO_MIN_MASS, SNAPNUM_EAGLE, \
    SNAPNUM_ILLUSTRIS
from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection

load_dotenv('../.env')


class DataFetcher(ABC):
    @abstractmethod
    def get_halos(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_subhalos(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_p_subhalo(self, *args, **kwargs):
        pass


class DataFetcherFactory:
    @staticmethod
    def get_data_fetcher(catalog: str, arg: str) -> DataFetcher:
        if catalog == "illustris-300":
            return Illustris300Fetcher(arg)
        elif catalog == "eagle":
            return EagleFetcher(arg)
        else:
            raise ValueError("Unsupported catalog type")


class Illustris300Fetcher(DataFetcher):
    def __init__(self, path):
        self.path = path

    def get_halos(self, *args, **kwargs) -> HaloContext:
        return self._get_halos(self.path)

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        return self._get_subhalos(self.path)

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        idsubhalo = kwargs.get('idsubhalo', None)

        if idsubhalo is None:
            raise ValueError("idsubhalo must be provided in kwargs")

        return self._get_p_subhalo(self.path, idsubhalo)

    def _get_halos(self, base_path: str) -> HaloContext:
        """
        Retrieve global halo properties such as mass (M200), radius (R200), position (GroupPos),
        and index of the first subhalo (GroupFirstSub).

        This function loads halo information from the entire group catalog for a specified snapshot.
        If no fields are specified, it defaults to retrieving M200, R200, GroupPos, and GroupFirstSub.

        Parameters:
        -----------
        base_path : str
            The base directory path to the simulation data.
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
        fields = ["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"]
        halos = il.groupcat.loadHalos(base_path, SNAPNUM_ILLUSTRIS, fields=fields)
        r200 = halos["Group_R_Crit200"]
        m200 = halos["Group_M_Crit200"]
        gpos = halos["GroupPos"]
        fsub = halos["GroupFirstSub"]
        return HaloContext(r200, m200, gpos, fsub, SCALF)

    def _get_subhalos(self, base_path: str) -> SubhaloContext:
        """
        Retrieve subhalo properties such as position, group number, and stellar half-mass radius.

        This function loads subhalo information from the specified snapshot of a simulation.
        If no fields are specified, it defaults to retrieving SubhaloPos, SubhaloGrNr,
        SubhaloHalfmassRadType, and SubhaloMassType.

        Parameters:
        -----------
        base_path : str
            The base directory path to the simulation data.

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
        fields = ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"]
        halos = il.groupcat.loadSubhalos(base_path, SNAPNUM_ILLUSTRIS, fields=fields)
        spos = halos["SubhaloPos"]
        grnr = halos["SubhaloGrNr"]
        shmr = halos["SubhaloHalfmassRadType"][:, 4]  # stars
        mstr = halos["SubhaloMassType"][:, 4]
        return SubhaloContext(spos, grnr, shmr, mstr, SCALF, H)

    def _get_p_subhalo(self, base_path: str, isubhalo: int) -> SubhaloParticles:
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
        fields = ["Coordinates", "Velocities", "Masses"]
        star_p = il.snapshot.loadSubhalo(base_path, SNAPNUM_ILLUSTRIS, isubhalo, "star", fields=fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"]
        m = star_p["Masses"]

        return SubhaloParticles(m, p, v, SCALF)


class EagleFetcher(DataFetcher):
    def __init__(self, simulation_code: str):
        self.simulation_code = simulation_code

    def get_halos(self, *args, **kwargs) -> HaloContext:
        return self._get_halos(self.simulation_code, SCALF)

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        return self._get_subhalos(self.simulation_code, SCALF, H)

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        idsubhalo = kwargs.get('idsubhalo', None)

        if idsubhalo is None:
            raise ValueError("idsubhalo must be provided in kwargs")

        return self._get_p_subhalo(self.simulation_code, idsubhalo, SCALF)

    def _get_halos(self, simulation_code, scalf) -> HaloContext:
        conn = get_connection()
        query = f"""
                SELECT
                    FOF.GroupID AS halo_id, -- Unique identifier for each halo
                    FOF.Group_M_Crit200 AS m200, -- Mass within R200 (critical density)
                    FOF.Group_R_Crit200 AS r200, -- Radius within which density is 200 times the critical density
                    FOF.GroupCentreOfPotential_x AS pos_x, -- X coordinate of halo's center
                    FOF.GroupCentreOfPotential_y AS pos_y, -- Y coordinate of halo's center
                    FOF.GroupCentreOfPotential_z AS pos_z, -- Z coordinate of halo's center
                    FOF.NumOfSubhalos AS num_subhalos -- Number of subhalos in each halo
                FROM
                    {simulation_code}_FOF AS FOF -- Table name for the Ref-L0100N1504 simulation
                WHERE
                    FOF.Group_M_Crit200 > {HALO_MIN_MASS} -- Minimum halo mass threshold
                    AND FOF.SnapNum = {SNAPNUM_EAGLE} -- Snapshot for redshift ~0.3
                ORDER BY
                    FOF.Group_M_Crit200 DESC -- Order results by halo mass
                """
        # Run the query
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        gpos = df.apply(lambda row: np.array([row['pos_x'], row['pos_y'], row['pos_z']]), axis=1)
        gpos = np.vstack(gpos)
        r200 = df['r200'].values
        m200 = df['m200'].values
        halo_id = df['halo_id'].values
        halo_context = HaloContext(r200, m200, gpos, halo_id, scale_factor=scalf)
        return halo_context

    def _get_subhalos(self, simulation_code: str, scalf: float, H: float) -> SubhaloContext:
        conn = get_connection()
        query = f"""
            SELECT
                SH.SubGroupNumber AS subhalo_id, -- Subhalo identifier within the halo
                SUM(SH.MassType_Star) AS total_stellar_mass, -- Total stellar mass in the subhalo
                AVG(SH.HalfMassRad_Star) AS avg_stellar_half_mass_radius, -- Average stellar half-mass radius in the subhalo
                AVG(SH.CentreOfPotential_x) AS avg_subhalo_pos_x, -- Average X coordinate of subhalo's center
                AVG(SH.CentreOfPotential_y) AS avg_subhalo_pos_y, -- Average Y coordinate of subhalo's center
                AVG(SH.CentreOfPotential_z) AS avg_subhalo_pos_z -- Average Z coordinate of subhalo's center
            FROM
                {simulation_code}_SubHalo AS SH -- Subhalo properties
            WHERE
                SH.MassType_Star > {SUBHALO_MIN_MASS} -- Minimum subhalo stellar mass threshold
                AND SH.SnapNum = {SNAPNUM_EAGLE} -- Snapshot for redshift ~0.3
            GROUP BY
                SH.SubGroupNumber -- Group only by subhalo id
            ORDER BY
                SH.SubGroupNumber ASC -- Order results by subhalo id
            """
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        spos = df.apply(lambda row: np.array([row['avg_subhalo_pos_x'], row['avg_subhalo_pos_y'],
                                              row['avg_subhalo_pos_z']]), axis=1).values
        spos = np.vstack(spos)
        grnr = df['subhalo_id'].values
        shmr = df['avg_stellar_half_mass_radius'].values
        mstr = df['total_stellar_mass'].values
        subhalo_context = SubhaloContext(spos, grnr, shmr, mstr, scalf, H)
        return subhalo_context

    def _get_p_subhalo(self, simulation_code: str, subhalo_id: int, scalf: float):
        conn = get_connection()
        query = f"""
            SELECT
                SH.SubGroupNumber AS subhalo_id, -- Subhalo identifier within the halo
                SH.MassType_Star AS stellar_mass, -- Stellar mass in the subhalo
                SH.HalfMassRad_Star AS stellar_half_mass_radius, -- Stellar half-mass radius in the subhalo
                SH.CentreOfPotential_x AS subhalo_pos_x, -- X coordinate of subhalo's center
                SH.CentreOfPotential_y AS subhalo_pos_y, -- Y coordinate of subhalo's center
                SH.CentreOfPotential_z AS subhalo_pos_z, -- Z coordinate of subhalo's center
                SH.GroupID AS group_id, -- ID of halo parent
                SH.Velocity_x AS particle_vel_x,     -- X velocity of the particle
                SH.Velocity_y AS particle_vel_y,     -- Y velocity of the particle
                SH.Velocity_z AS particle_vel_z      -- Z velocity of the particle
            FROM
                {simulation_code}_SubHalo AS SH -- Subhalo properties
            WHERE
                SH.SubGroupNumber = {subhalo_id}  -- Filter particles by the subhalo ID
                AND SH.SnapNum = {SNAPNUM_EAGLE} -- Snapshot for redshift ~0.3
        """
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        m = df['stellar_mass'].values
        p = df.apply(lambda row: np.array([row['subhalo_pos_x'], row['subhalo_pos_y'],
                                           row['subhalo_pos_z']]), axis=1).values
        p = np.vstack(p)
        v = df.apply(lambda row: np.array([row['particle_vel_x'], row['particle_vel_y'],
                                           row['particle_vel_z']]), axis=1).values
        v = np.vstack(v)
        subhalo_particles = SubhaloParticles(m, p, v, scalf)
        return subhalo_particles
