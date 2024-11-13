from abc import ABC, abstractmethod

import h5py
from dotenv import load_dotenv
import eagleSqlTools as sql
import numpy as np
import pandas as pd
import illustris_python as il

from src.data_generation.src.constants import H, HALO_MIN_MASS, SUBHALO_MIN_MASS, SNAPNUM_EAGLE, \
    SNAPNUM_ILLUSTRIS, SOLAR_MASS_CONVERSION, SCALF_ILLUSTRIS, SCALF_EAGLE
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
        # Units of measure -> [10^10𝑀⊙/ℎ, 𝑐𝑘𝑝𝑐/ℎ, 𝑐𝑘𝑝𝑐/ℎ, int]
        fields = ["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"]
        halos = il.groupcat.loadHalos(base_path, SNAPNUM_ILLUSTRIS, fields=fields)
        halos['pos_x'], halos['pos_y'], halos['pos_z'] = np.split(halos["GroupPos"], 3, axis=1)
        halos['pos_x'] = halos['pos_x'].flatten()
        halos['pos_y'] = halos['pos_y'].flatten()
        halos['pos_z'] = halos['pos_z'].flatten()
        del halos["GroupPos"]
        halos = pd.DataFrame.from_dict(halos)
        halos['Group_M_Crit200'] = halos['Group_M_Crit200'] * SOLAR_MASS_CONVERSION # 𝑀⊙/ℎ

        # Filter halos by min mass
        halos = halos[halos['Group_M_Crit200'] > HALO_MIN_MASS * H]
        r200 = halos["Group_R_Crit200"]
        m200 = halos["Group_M_Crit200"]
        gpos = np.vstack(halos.apply(lambda row: np.array([row['pos_x'], row['pos_y'], row['pos_z']]), axis=1).values)
        fsub = halos["GroupFirstSub"]
        halo_id = np.arange(len(r200))
        return HaloContext(r200, m200, gpos, fsub, SCALF_ILLUSTRIS, halo_id)

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
        # Units of measure -> [𝑐𝑘𝑝𝑐/ℎ, int, 𝑐𝑘𝑝𝑐/ℎ, 10^10𝑀⊙/ℎ]
        fields = ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"]
        halos = il.groupcat.loadSubhalos(base_path, SNAPNUM_ILLUSTRIS, fields=fields)
        spos = halos["SubhaloPos"]
        grnr = halos["SubhaloGrNr"]
        shmr = halos["SubhaloHalfmassRadType"][:, 4]  # stars
        mstr = halos["SubhaloMassType"][:, 4] * SOLAR_MASS_CONVERSION / H # 𝑀⊙
        subhalo_id = np.arange(len(spos))
        return SubhaloContext(spos, grnr, shmr, mstr, SCALF_ILLUSTRIS, subhalo_id)

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
        # Units of measure [𝑐𝑘𝑝𝑐/ℎ, 𝑘𝑚𝑎‾√/𝑠, 10^10𝑀⊙/ℎ]
        fields = ["Coordinates", "Velocities", "Masses"]
        star_p = il.snapshot.loadSubhalo(base_path, SNAPNUM_ILLUSTRIS, isubhalo, "star", fields=fields)
        p = star_p["Coordinates"]
        v = star_p["Velocities"] * np.sqrt(SCALF_ILLUSTRIS) # 𝑘𝑚𝑎‾√/𝑠 -> kms-1
        m = star_p["Masses"] * SOLAR_MASS_CONVERSION

        return SubhaloParticles(m, p, v, SCALF_ILLUSTRIS)


class EagleFetcher(DataFetcher):
    def __init__(self, simulation_code: str):
        self.simulation_code = simulation_code

    def get_halos(self, *args, **kwargs) -> HaloContext:
        return self._get_halos(self.simulation_code, SCALF_EAGLE)

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        return self._get_subhalos(self.simulation_code, SCALF_EAGLE)

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        idsubhalo = kwargs.get('idsubhalo', None)

        if idsubhalo is None:
            raise ValueError("idsubhalo must be provided in kwargs")

        return self._get_p_subhalo(self.simulation_code, idsubhalo, SCALF_EAGLE)

    def _get_halos(self, simulation_code, scalf) -> HaloContext:
        conn = get_connection()
        query = f"""
                WITH RankedSubhalos AS (
                    SELECT 
                        SH.GroupID AS halo_id, -- Halo ID linked to the subhalo
                        SH.GalaxyID AS subhalo_id, -- Subhalo ID
                        SH.MassType_Star AS stellar_mass, -- Stellar mass of the subhalo
                        ROW_NUMBER() OVER(PARTITION BY SH.GroupID ORDER BY SH.MassType_Star DESC) AS rn -- Rank subhalos by mass
                    FROM
                        {simulation_code}_Subhalo AS SH
                    WHERE
                        SH.SnapNum = {SNAPNUM_EAGLE} -- Match the snapshot for consistency
                        AND SH.MassType_Star > {SUBHALO_MIN_MASS} -- Minimum subhalo stellar mass threshold
                )
                SELECT
                    FOF.GroupID AS halo_id, -- Unique identifier for each halo
                    FOF.Group_M_Crit200 AS m200, -- Mass within R200 (critical density)
                    FOF.Group_R_Crit200 AS r200, -- Radius within which density is 200 times the critical density
                    FOF.GroupCentreOfPotential_x AS pos_x, -- X coordinate of halo's center
                    FOF.GroupCentreOfPotential_y AS pos_y, -- Y coordinate of halo's center
                    FOF.GroupCentreOfPotential_z AS pos_z, -- Z coordinate of halo's center
                    FOF.NumOfSubhalos AS num_subhalos, -- Number of subhalos in each halo
                    RS.subhalo_id AS max_subhalo_id, -- ID of the most massive subhalo in the halo
                    RS.stellar_mass AS max_subhalo_mass -- Stellar mass of the most massive subhalo
                FROM
                    {simulation_code}_FOF AS FOF -- Table for halo properties
                LEFT JOIN 
                    RankedSubhalos AS RS ON FOF.GroupID = RS.halo_id AND RS.rn = 1 -- Only the most massive subhalo (rn = 1)
                WHERE
                    FOF.Group_M_Crit200 > {HALO_MIN_MASS} -- Minimum halo mass threshold
                    AND FOF.SnapNum = {SNAPNUM_EAGLE} -- Snapshot for redshift ~0.3
                ORDER BY
                    FOF.GroupID ASC -- Order results by halo ID
                """
        # Run the query
        array = sql.execute_query(conn, query)
        # Units of measure [m200=M⊙, r200=pkpc, gpos=cMpc]
        df = pd.DataFrame(array)

        # pkps -> ckps / h
        gpos = df.apply(lambda row: np.array([row['pos_x'] / H, row['pos_y'] / H, row['pos_z'] / H]), axis=1)
        gpos = np.vstack(gpos)
        r200 = df['r200'].values / H    # pkps -> ckps / h
        m200 = df['m200'].values / H    # 𝑀⊙/ℎ
        halo_id = df['halo_id'].values
        df['halo_id'] = df['halo_id'].astype(int)
        fsub = df['max_subhalo_id'].values

        halo_context = HaloContext(r200, m200, gpos, fsub, scale_factor=scalf, halo_id=halo_id)
        return halo_context

    def _get_subhalos(self, simulation_code: str, scalf: float) -> SubhaloContext:
        conn = get_connection()
        query = f"""
            SELECT
                FOF.GroupID AS halo_id, -- Halo identifier
                FOF.Group_M_Crit200 AS m200, -- Halo mass within R200
                SH.GalaxyID AS subhalo_id, -- Subhalo identifier
                SH.MassType_Star AS total_stellar_mass, -- Total stellar mass in the subhalo
                SH.HalfMassRad_Star AS avg_stellar_half_mass_radius, --  stellar half-mass radius in the subhalo
                SH.CentreOfPotential_x AS avg_subhalo_pos_x, --  X coordinate of subhalo's center
                SH.CentreOfPotential_y AS avg_subhalo_pos_y, --  Y coordinate of subhalo's center
                SH.CentreOfPotential_z AS avg_subhalo_pos_z --  Z coordinate of subhalo's center
            FROM
                {simulation_code}_FOF AS FOF -- Halo properties
            JOIN
                {simulation_code}_Subhalo AS SH -- Subhalo properties
            ON 
                FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
            WHERE
                SH.MassType_Star > {SUBHALO_MIN_MASS} -- Minimum subhalo stellar mass threshold
                AND FOF.Group_M_Crit200 > {HALO_MIN_MASS} -- Minimum halo mass threshold
                AND SH.SnapNum = {SNAPNUM_EAGLE} -- Snapshot for redshift ~0.2
            ORDER BY
                SH.GalaxyID ASC -- Order results by subhalo id
            """
        # Executing the join is important so we ensure that we are selecting subhalos that are child of valid halos
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        # Units of measure [avg_stellar_half_mass_radius=pkpc, total_stellar_mass=M⊙, spos=cMpc]
        df = df.sort_values(by='subhalo_id')

        # pkps -> ckps / h
        spos = df.apply(lambda row: np.array([row['avg_subhalo_pos_x'] / H, row['avg_subhalo_pos_y'] / H,
                                              row['avg_subhalo_pos_z'] / H]), axis=1).values
        spos = np.vstack(spos)
        grnr = df['halo_id'].values
        shmr = df['avg_stellar_half_mass_radius'].values / H  # pkps -> ckps / h
        mstr = df['total_stellar_mass'].values
        subhalo_ids = df['subhalo_id'].values
        subhalo_context = SubhaloContext(spos, grnr, shmr, mstr, scalf, subhalo_ids)
        return subhalo_context

    def _get_p_subhalo(self, simulation_code: str, subgroup_id: int, halo_id: int, scalf: float):
        pass
