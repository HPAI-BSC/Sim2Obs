import os.path
from abc import ABC, abstractmethod

from dotenv import load_dotenv
import eagleSqlTools as sql
import numpy as np
import pandas as pd
import illustris_python as il
from omegaconf import DictConfig

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection, read_dataset

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

# TODO what if results are none ?


class DataFetcherFactory:
    @staticmethod
    def get_data_fetcher(cfg: DictConfig) -> DataFetcher:
        if cfg.catalog.name == "illustris-300":
            return Illustris300Fetcher(cfg)
        elif cfg.catalog.name == "eagle":
            return EagleFetcher(cfg)
        else:
            raise ValueError(f"Unsupported catalog={cfg.catalog.name} type")


class Illustris300Fetcher(DataFetcher):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def get_halos(self, *args, **kwargs) -> HaloContext:
        return self._get_halos()

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        return self._get_subhalos()

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        idsubhalo = kwargs.get('idsubhalo', None)

        if idsubhalo is None:
            raise ValueError("idsubhalo must be provided in kwargs")

        return self._get_p_subhalo(idsubhalo)

    def _get_halos(self) -> HaloContext:
        """
        Retrieve global halo properties such as mass (M200), radius (R200), position (GroupPos),
        and index of the first subhalo (GroupFirstSub).

        This function loads halo information from the entire group catalog for a specified snapshot.
        If no fields are specified, it defaults to retrieving M200, R200, GroupPos, and GroupFirstSub.

        Parameters:
        -----------
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
        path = self.cfg.catalog.path + '/output'

        if not os.path.exists(path):
            raise Exception(f'The path {path} does not exist.')

        halos = il.groupcat.loadHalos(path, self.cfg.catalog.snapnum, fields=fields)
        if len(halos) > 0:
            halos['pos_x'], halos['pos_y'], halos['pos_z'] = np.split(halos["GroupPos"], 3, axis=1)
            halos['pos_x'] = halos['pos_x'].flatten()
            halos['pos_y'] = halos['pos_y'].flatten()
            halos['pos_z'] = halos['pos_z'].flatten()
            del halos["GroupPos"]
            halos = pd.DataFrame.from_dict(halos)
            halos['Group_M_Crit200'] = halos['Group_M_Crit200'] * self.cfg.solar_mass_convention # 𝑀⊙/ℎ

            # Filter halos by min mass
            halos = halos[halos['Group_M_Crit200'] > self.cfg.halo_min_mass * self.cfg.H]
            r200 = halos["Group_R_Crit200"].values
            m200 = halos["Group_M_Crit200"].values
            gpos = np.vstack(halos.apply(lambda row: np.array([row['pos_x'], row['pos_y'], row['pos_z']]), axis=1).values)
            fsub = halos["GroupFirstSub"].values
            halo_id = np.arange(len(r200))
        else:
            r200 = m200 = gpos = fsub = halo_id = np.ndarray([])
        return HaloContext(r200, m200, gpos, fsub, self.cfg.catalog.scalf, halo_id)

    def _get_subhalos(self) -> SubhaloContext:
        """
        Retrieve subhalo properties such as position, group number, and stellar half-mass radius.

        This function loads subhalo information from the specified snapshot of a simulation.
        If no fields are specified, it defaults to retrieving SubhaloPos, SubhaloGrNr,
        SubhaloHalfmassRadType, and SubhaloMassType.

        Parameters:
        -----------
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
        path = self.cfg.catalog.path + '/output'

        if not os.path.exists(path):
            raise Exception(f'The path {path} does not exist.')

        halos = il.groupcat.loadSubhalos(path, self.cfg.catalog.snapnum, fields=fields)
        if len(halos) > 0:
            spos = halos["SubhaloPos"]
            grnr = halos["SubhaloGrNr"]
            shmr = halos["SubhaloHalfmassRadType"][:, 4]  # stars
            mstr = halos["SubhaloMassType"][:, 4] * self.cfg.solar_mass_convention / self.cfg.H # 𝑀⊙
            subhalo_id = np.arange(len(spos))
        else:
            spos = grnr = shmr = mstr = subhalo_id = np.ndarray([])
        return SubhaloContext(spos, grnr, shmr, mstr, self.cfg.catalog.scalf, subhalo_id)

    def _get_p_subhalo(self, isubhalo: int) -> SubhaloParticles:
        """
        Retrieve particle data for a specific subhalo, including coordinates, velocities, and masses.

        This function retrieves particle data from a subhalo in a simulation snapshot.
        If no fields are specified, it defaults to retrieving coordinates, velocities,
        and masses of star particles. The subhalo specified by `isubhalo` must be the
        first subhalo.

        Parameters:
        -----------
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
        path = self.cfg.catalog.path + '/output'

        if not os.path.exists(path):
            raise Exception(f'The path {path} does not exist.')

        star_p = il.snapshot.loadSubhalo(path, self.cfg.catalog.snapnum, isubhalo, "star", fields=fields)
        if len(star_p) > 0:
            p = star_p["Coordinates"]
            v = star_p["Velocities"] * np.sqrt(self.cfg.catalog.scalf) # 𝑘𝑚𝑎‾√/𝑠 -> kms-1
            m = star_p["Masses"] * self.cfg.solar_mass_convention
        else:
            p = v = m = np.ndarray([])
        return SubhaloParticles(m, p, v, self.cfg.catalog.scalf)


class EagleFetcher(DataFetcher):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def get_halos(self, *args, **kwargs) -> HaloContext:
        return self._get_halos()

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        return self._get_subhalos()

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        subgroup_id = kwargs.get('subgroup_id', None)
        halo_id = kwargs.get('halo_id')

        if subgroup_id is None:
            raise ValueError("subgroup_id must be provided in kwargs")
        if halo_id is None:
            raise ValueError("halo_id must be provided in kwargs")
        return self._get_p_subhalo(subgroup_id, halo_id)

    def _get_halos(self) -> HaloContext:
        conn = get_connection()
        query = f"""
                SELECT 
                    FOF.GroupID AS halo_id,                          -- Unique identifier for each halo
                    FOF.Group_M_Crit200 AS m200,                     -- Mass within R200 (critical density)
                    FOF.Group_R_Crit200 AS r200,                     -- Radius within which density is 200 times the critical density
                    FOF.GroupCentreOfPotential_x AS pos_x,          -- X coordinate of halo's center
                    FOF.GroupCentreOfPotential_y AS pos_y,          -- Y coordinate of halo's center
                    FOF.GroupCentreOfPotential_z AS pos_z,          -- Z coordinate of halo's center
                    SH.SubGroupNumber AS subhalo_id,                 -- ID of the subhalo
                    SH.MassType_Star AS subhalo_mass                -- Mass of the subhalo
                FROM 
                    {self.cfg.catalog.simulation_code}_FOF AS FOF -- Halo properties
                LEFT JOIN 
                    (
                        SELECT 
                            SH.GroupID,
                            SH.SubGroupNumber,
                            SH.MassType_Star
                        FROM 
                            {self.cfg.catalog.simulation_code}_Subhalo AS SH
                        WHERE 
                            SH.MassType_Star = (
                                SELECT MAX(SH2.MassType_Star) -- Ensure it's the largest subhalo by mass
                                FROM {self.cfg.catalog.simulation_code}_Subhalo AS SH2
                                WHERE SH2.GroupID = SH.GroupID AND SH.MassType_Star > {self.cfg.subhalo_min_mass}
                                AND SH.SnapNum = {self.cfg.catalog.snapnum} 
                            )
                    ) AS SH ON FOF.GroupID = SH.GroupID
                WHERE 
                    FOF.Group_M_Crit200 > {self.cfg.halo_min_mass}  -- Minimum halo mass threshold
                    AND FOF.SnapNum = {self.cfg.catalog.snapnum} -- Snapshot for redshift ~0.2
                """
        # Run the query
        array = sql.execute_query(conn, query)
        # Units of measure [m200=M⊙, r200=pkpc, gpos=cMpc]
        df = pd.DataFrame(array)
        if len(df) > 0:
            df = df.sort_values(by='halo_id')
            # pkps -> ckps / h
            gpos = df.apply(lambda row: np.array([row['pos_x'] / self.cfg.H, row['pos_y'] / self.cfg.H, row['pos_z'] / self.cfg.H]), axis=1)
            gpos = np.vstack(gpos)
            r200 = df['r200'].values / self.cfg.H    # pkps -> ckps / h
            m200 = df['m200'].values / self.cfg.H    # 𝑀⊙/ℎ
            halo_id = df['halo_id'].values
            df['halo_id'] = df['halo_id'].astype(int)
            fsub = df['subhalo_id'].values

            halo_context = HaloContext(r200, m200, gpos, fsub, scale_factor=self.cfg.catalog.scalf, halo_id=halo_id)
        else:
            halo_context = HaloContext(np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]),
                                       scale_factor=self.cfg.catalog.scalf, halo_id=np.ndarray([]))
        return halo_context

    def _get_subhalos(self) -> SubhaloContext:
        conn = get_connection()
        query = f"""
            SELECT
                FOF.GroupID AS halo_id, -- Halo identifier
                FOF.Group_M_Crit200 AS m200, -- Halo mass within R200
                SH.SubGroupNumber AS subhalo_id, -- Subgroup
                SH.MassType_Star AS total_stellar_mass, -- Total stellar mass in the subhalo
                SH.HalfMassRad_Star AS stellar_half_mass_radius, --  stellar half-mass radius in the subhalo
                SH.CentreOfPotential_x AS subhalo_pos_x, --  X coordinate of subhalo's center
                SH.CentreOfPotential_y AS subhalo_pos_y, --  Y coordinate of subhalo's center
                SH.CentreOfPotential_z AS subhalo_pos_z --  Z coordinate of subhalo's center
            FROM
                {self.cfg.catalog.simulation_code}_FOF AS FOF -- Halo properties
            JOIN
                {self.cfg.catalog.simulation_code}_Subhalo AS SH -- Subhalo properties
            ON 
                FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
            WHERE
                SH.MassType_Star > {self.cfg.subhalo_min_mass} -- Minimum subhalo stellar mass threshold
                AND FOF.Group_M_Crit200 > {self.cfg.halo_min_mass} -- Minimum halo mass threshold
                AND SH.SnapNum = {self.cfg.catalog.snapnum} -- Snapshot for redshift ~0.2
            ORDER BY
                SH.SubGroupNumber ASC -- Order results by subhalo id
            """

        '''
        query = f"""
            SELECT
                FOF.GroupID AS halo_id, -- Halo identifier
                FOF.Group_M_Crit200 AS m200, -- Halo mass within R200
                SH.SubGroupNumber AS subhalo_id, -- Subgroup
                AVG(SH.MassType_Star) AS avg_total_stellar_mass, -- Average total stellar mass in the subhalo
                AVG(SH.HalfMassRad_Star) AS avg_stellar_half_mass_radius, -- Average stellar half-mass radius in the subhalo
                AVG(SH.CentreOfPotential_x) AS avg_subhalo_pos_x, -- Average X coordinate of subhalo's center
                AVG(SH.CentreOfPotential_y) AS avg_subhalo_pos_y, -- Average Y coordinate of subhalo's center
                AVG(SH.CentreOfPotential_z) AS avg_subhalo_pos_z -- Average Z coordinate of subhalo's center
            FROM
                {self.cfg.catalog.simulation_code}_FOF AS FOF -- Halo properties
            JOIN
                {self.cfg.catalog.simulation_code}_Subhalo AS SH -- Subhalo properties
            ON 
                FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
            WHERE
                SH.MassType_Star > {self.cfg.subhalo_min_mass} -- Minimum subhalo stellar mass threshold
                AND FOF.Group_M_Crit200 > {self.cfg.halo_min_mass} -- Minimum halo mass threshold
                AND SH.SnapNum = {self.cfg.catalog.snapnum} -- Snapshot for redshift ~0.2
            GROUP BY
                FOF.GroupID, SH.SubGroupNumber, FOF.Group_M_Crit200 -- Group by halo id and subhalo id
            ORDER BY
                SH.SubGroupNumber ASC -- Order results by subhalo id
            """
        '''
        # Executing the join is important so we ensure that we are selecting subhalos that are child of valid halos
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        # Units of measure [avg_stellar_half_mass_radius=pkpc, total_stellar_mass=M⊙, spos=cMpc]
        if len(df) > 0:
            df = df.sort_values(by='subhalo_id')

            # pkps -> ckps / h
            spos = df.apply(lambda row: np.array([row['subhalo_pos_x'] / self.cfg.H, row['subhalo_pos_y'] / self.cfg.H,
                                                  row['subhalo_pos_z'] / self.cfg.H]), axis=1).values
            spos = np.vstack(spos)
            grnr = df['halo_id'].values
            shmr = df['stellar_half_mass_radius'].values / self.cfg.H  # pkps -> ckps / h
            mstr = df['total_stellar_mass'].values
            subhalo_ids = df['subhalo_id'].values
            subhalo_context = SubhaloContext(spos, grnr, shmr, mstr, self.cfg.catalog.scalf, subhalo_ids)
        else:
            subhalo_context = SubhaloContext(np.ndarray([]), np.ndarray([]), np.ndarray([]),
                                             np.ndarray([]), self.cfg.catalog.scalf, np.ndarray([]))
        return subhalo_context

    def _get_p_subhalo(self, subgroup_id: int, halo_id: int):
        itype = 4  # PartType
        path = self.cfg.catalog.path

        gns = read_dataset(itype, 'GroupNumber', path)
        sgns = read_dataset(itype, 'SubGroupNumber', path)
        mask = np.logical_and(gns == halo_id, sgns == subgroup_id)

        # TODO automate this in the future
        # TODO check conversion required
        data = {
            'mass': read_dataset(itype, 'Mass', path)[mask],
            'coords': read_dataset(itype, 'Coordinates', path)[mask],
            'velocity': read_dataset(itype, 'Velocity', path)[mask]
        }

        subhalo_particles = SubhaloParticles(data['mass'], data['coords'], data['velocity'], self.cfg.catalog.scalf)
        return subhalo_particles

