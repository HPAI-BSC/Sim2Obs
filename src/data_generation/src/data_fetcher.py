import os.path
from abc import ABC, abstractmethod
import rootutils

from dotenv import load_dotenv
import eagleSqlTools as sql
import numpy as np
import pandas as pd
import illustris_python as il
from omegaconf import DictConfig

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection, read_dataset
#from src import utils

load_dotenv('../.env')


#rootutils.setup_root(__file__, pythonpath=True)
#log = utils.get_pylogger(__name__)


class DataFetcher(ABC):
    """
    Abstract base class for fetching data from various catalogs.
    """
    @abstractmethod
    def get_halos(self, *args, **kwargs):
        """
        Abstract method to fetch halo data.
        """
        pass

    @abstractmethod
    def get_subhalos(self, *args, **kwargs):
        """
        Abstract method to fetch subhalo data.
        """
        pass

    @abstractmethod
    def get_p_subhalo(self, *args, **kwargs):
        """
        Abstract method to fetch particle data for a subhalo.
        """
        pass


class DataFetcherFactory:
    """
    Factory class to instantiate the correct DataFetcher based on the configuration.
    """
    @staticmethod
    def get_data_fetcher(cfg: DictConfig) -> DataFetcher:
        """
        Returns the appropriate data fetcher based on the catalog name in the configuration.

        Parameters:
        -----------
        cfg : DictConfig
            Configuration object containing catalog settings.

        Returns:
        --------
        DataFetcher
            The appropriate data fetcher instance for the specified catalog.
        """
        #log.debug(f"Fetching data fetcher for catalog: {cfg.catalog.name}")
        if cfg.catalog.name == "illustris-300":
            return Illustris300Fetcher(cfg)
        elif cfg.catalog.name == "eagle":
            return EagleFetcher(cfg)
        else:
            raise ValueError(f"Unsupported catalog={cfg.catalog.name} type")


class Illustris300Fetcher(DataFetcher):
    """
    Data fetcher for the Illustris-300 catalog.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initializes the fetcher with the given configuration.

        Parameters:
        -----------
        cfg : DictConfig
            Configuration object containing catalog settings.
        """
        self.cfg = cfg
        #log.info(f"Illustris300Fetcher initialized with config: {cfg}")

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """
        Fetches the halo data.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
        #log.info("Fetching halo data...")
        return self._get_halos()

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        """
        Fetches the subhalo data.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        #log.info("Fetching subhalo data...")
        return self._get_subhalos()

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        """
        Fetches particle data for a given subhalo.

        Parameters:
        -----------
        subhalo_id : int
            ID of the subhalo for which particle data is needed.

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data.
        """
        subhalo_id = kwargs.get('subhalo_id', None)

        if subhalo_id is None:
            #log.error("subhalo_id must be provided in kwargs")
            raise ValueError("subhalo_id must be provided in kwargs")

        #log.info(f"Fetching particle data for subhalo: {subhalo_id}")
        return self._get_p_subhalo(subhalo_id)

    def _get_halos(self) -> HaloContext:
        """
        Fetches halo data from the Illustris-300 simulation.

        Returns:
        --------
        HaloContext
            A context object containing halo data such as mass, radius, position, and subhalo index.
        """
        #log.debug("Fetching halos from Illustris-300...")

        # Units of measure -> [10^10𝑀⊙/ℎ, 𝑐𝑘𝑝𝑐/ℎ, 𝑐𝑘𝑝𝑐/ℎ, int]
        fields = ["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"]
        path = self.cfg.catalog.path + '/output'

        if not os.path.exists(path):
            #log.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        halos = il.groupcat.loadHalos(path, self.cfg.catalog.snapnum, fields=fields)
        if len(halos) > 0:
            halos['GroupPos'] = [halos['GroupPos'][i] for i in range(halos['GroupPos'].shape[0])]
            halos = pd.DataFrame.from_dict(halos)
            halos['halo_id'] = np.arange(len(halos['Group_M_Crit200']))
            halos['Group_M_Crit200'] = halos['Group_M_Crit200'] * self.cfg.solar_mass_convention # 𝑀⊙/ℎ
            # Filter halos by min mass
            halos = halos[halos['Group_M_Crit200'] > self.cfg.halo_min_mass * self.cfg.H]

            column_map = {'halo_id': 'halo_id', 'r200': 'Group_R_Crit200', 'm200': 'Group_M_Crit200',
                          'gpos': 'GroupPos', 'fsub': 'GroupFirstSub'}
            return HaloContext(halos, self.cfg.catalog.scalf, column_map)
            #log.debug("Halo data fetched successfully.")
        else:
            #log.warning("No halos found.")
            return HaloContext(pd.DataFrame(data=[]), self.cfg.catalog.scalf, {})

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetches subhalo data from the Illustris-300 simulation.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        #log.debug("Fetching subhalos from Illustris-300...")

        # Units of measure -> [𝑐𝑘𝑝𝑐/ℎ, int, 𝑐𝑘𝑝𝑐/ℎ, 10^10𝑀⊙/ℎ]
        fields = ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"]
        path = self.cfg.catalog.path + '/output'

        if not os.path.exists(path):
            #log.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        subhalos = il.groupcat.loadSubhalos(path, self.cfg.catalog.snapnum, fields=fields)
        if len(subhalos) > 0:
            subhalos['spos'] = subhalos['SubhaloPos']
            subhalos['SubhaloHalfmassRadType'] = subhalos["SubhaloHalfmassRadType"][:, 4]  # stars
            subhalos['SubhaloMassType'] = subhalos["SubhaloMassType"][:, 4] * self.cfg.solar_mass_convention / self.cfg.H # 𝑀⊙
            subhalos['subhalo_id'] = np.arange(len(subhalos['spos']))
            column_map = {"subhalo_id": "subhalo_id", "spos": "spos", "id_parents": "SubhaloGrNr",
                          "shmr": "SubhaloHalfmassRadType", "mstr": "SubhaloMassType"}
            return SubhaloContext(subhalos, self.cfg.catalog.scalf, column_map)
            #log.debug("Subhalo data fetched successfully.")
        else:
            #log.warning("No subhalos found.")
            return SubhaloContext(pd.DataFrame(data=[]), self.cfg.catalog.scalf, {})

    def _get_p_subhalo(self, subhalo_id: int) -> SubhaloParticles:
        """
        Fetches particle data for a specific subhalo.

        Parameters:
        -----------
        isubhalo : int
            The index of the subhalo from which to retrieve particle data.

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data such as mass, coordinates, and velocities.
        """
        #log.debug(f"Fetching particle data for subhalo: {subhalo_id}")

        # Units of measure [𝑐𝑘𝑝𝑐/ℎ, 𝑘𝑚𝑎‾√/𝑠, 10^10𝑀⊙/ℎ]
        fields = ["Coordinates", "Velocities", "Masses"]
        path = self.cfg.catalog.path + '/output'

        if not os.path.exists(path):
            #log.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        star_p = il.snapshot.loadSubhalo(path, self.cfg.catalog.snapnum, subhalo_id, "star", fields=fields)
        if len(star_p) > 0:
            p = star_p["Coordinates"]
            v = star_p["Velocities"] * np.sqrt(self.cfg.catalog.scalf) # 𝑘𝑚𝑎‾√/𝑠 -> kms-1
            m = star_p["Masses"] * self.cfg.solar_mass_convention
            #log.debug("Particle data fetched successfully.")
        else:
            #log.warning(f"No particle data found for subhalo {subhalo_id}.")
            p = v = m = np.ndarray([])
        return SubhaloParticles(m, p, v, self.cfg.catalog.scalf)


class EagleFetcher(DataFetcher):
    """
    Data fetcher for the Eagle catalog.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initializes the fetcher with the given configuration.

        Parameters:
        -----------
        cfg : DictConfig
            Configuration object containing catalog settings.
        """
        self.cfg = cfg
        #log.info(f"EagleFetcher initialized with config: {cfg}")

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """
        Fetches the halo data for Eagle catalog.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
        #log.info("Fetching halo data for Eagle catalog...")
        return self._get_halos()

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        """
        Fetches the subhalo data for Eagle catalog.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        #log.info("Fetching subhalo data for Eagle catalog...")
        return self._get_subhalos()

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        """
        Fetches particle data for a given subhalo in the Eagle catalog.

        Parameters:
        -----------
        subhalo_id : int
            ID of the subhalo for which particle data is needed.

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data.
        """
        subgroup_id = kwargs.get('subhalo_id', None)
        halo_id = kwargs.get('group_number')

        if subgroup_id is None:
            #log.error("subhalo_id must be provided in kwargs")
            raise ValueError("subgroup_id must be provided in kwargs")
        if halo_id is None:
            #log.error("halo_id must be provided in kwargs")
            raise ValueError("group_number must be provided in kwargs")

        #log.info(f"Fetching particle data for subhalo: {subgroup_id}")
        return self._get_p_subhalo(subgroup_id, halo_id)

    def _get_halos(self) -> HaloContext:
        """
        Fetches halo data from the Eagle simulation.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
        #log.debug("Fetching halos from Eagle...")
        conn = get_connection()
        query = f"""
                SELECT 
                    FOF.GroupID AS halo_id,                          -- Unique identifier for each halo
                    FOF.Group_M_Crit200 AS m200,                     -- Mass within R200 (critical density)
                    FOF.Group_R_Crit200 AS r200,                     -- Radius within which density is 200 times the critical density
                    FOF.GroupCentreOfPotential_x AS pos_x,          -- X coordinate of halo's center
                    FOF.GroupCentreOfPotential_y AS pos_y,          -- Y coordinate of halo's center
                    FOF.GroupCentreOfPotential_z AS pos_z,          -- Z coordinate of halo's center
                    SH.GroupNumber AS group_number,                 -- ID FoF, for linking particles
                    SH.SubGroupNumber AS subhalo_id,                 -- ID of the subhalo
                    SH.MassType_Star AS subhalo_mass                -- Mass of the subhalo
                FROM 
                    {self.cfg.catalog.simulation_code}_FOF AS FOF -- Halo properties
                LEFT JOIN 
                    (
                        SELECT 
                            SH.GroupID,
                            SH.SubGroupNumber,
                            SH.GroupNumber, 
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
            df['gpos'] = df.apply(lambda row: np.array([row['pos_x'] / self.cfg.H, row['pos_y'] / self.cfg.H, row['pos_z'] / self.cfg.H]), axis=1)
            df['r200'] = df['r200'].values / self.cfg.H    # pkps -> ckps / h
            df['m200'] = df['m200'].values / self.cfg.H    # 𝑀⊙/ℎ
            df['halo_id'] = df['halo_id'].astype(int)
            column_map = {'halo_id': 'halo_id', 'r200': 'r200', 'm200': 'm200', 'gpos': 'gpos', 'fsub': 'subhalo_id',
                          'group_number': 'group_number'}
            halo_context = HaloContext(df, scale_factor=self.cfg.catalog.scalf, column_map=column_map)
            #log.debug("Halo data fetched successfully.")
        else:
            #log.warning("No halo data found in Eagle database.")
            halo_context = HaloContext(pd.DataFrame(data=[]), scale_factor=self.cfg.catalog.scalf, column_map={})
        return halo_context

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetches subhalo data from the Eagle simulation.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        #log.debug("Fetching subhalos from Eagle...")
        conn = get_connection()
        query = f"""
            SELECT
                FOF.GroupID AS halo_id, -- Halo identifier
                FOF.Group_M_Crit200 AS m200, -- Halo mass within R200
                SH.GroupNumber AS group_number, -- ID FoF, for linking particles
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
        # Executing the join is important so we ensure that we are selecting subhalos that are child of valid halos
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)
        # Units of measure [avg_stellar_half_mass_radius=pkpc, total_stellar_mass=M⊙, spos=cMpc]
        if len(df) > 0:
            df = df.sort_values(by='subhalo_id')
            # pkps -> ckps / h
            df['spos'] = df.apply(lambda row: np.array([row['subhalo_pos_x'] / self.cfg.H,
                                                        row['subhalo_pos_y'] / self.cfg.H,
                                                        row['subhalo_pos_z'] / self.cfg.H]), axis=1)
            df['stellar_half_mass_radius'] = df['stellar_half_mass_radius'].values / self.cfg.H  # pkps -> ckps / h
            column_map = {"subhalo_id": "subhalo_id", "spos": "spos", "id_parents": "halo_id",
                          "shmr": "stellar_half_mass_radius", "mstr": "total_stellar_mass",
                          "group_number": "group_number"}
            subhalo_context = SubhaloContext(df, self.cfg.catalog.scalf, column_map)
            #log.debug("Subhalo data fetched successfully.")
        else:
            #log.warning("No subhalo data found in Eagle database.")
            subhalo_context = SubhaloContext(pd.DataFrame(data=[]), self.cfg.catalog.scalf, {})
        return subhalo_context

    def _get_p_subhalo(self, subgroup_id: int, group_number: int):
        """
        Fetches particle data for a specific subhalo from the Eagle simulation.

        Parameters:
        -----------
        subgroup_id : int
            The index of the subhalo from which to retrieve particle data.
        halo_id: int
            The index of the halo where the subhalo belongs from

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data such as mass, coordinates, and velocities.
        """
        #log.debug(f"Fetching particle data for subhalo: {subgroup_id}")
        itype = 4  # PartType
        path = self.cfg.catalog.path

        if not os.path.exists(path):
            #log.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')
        data = {
            'mass': read_dataset(itype, 'Mass', path),
            'coords': read_dataset(itype, 'Coordinates', path),
            'velocity': read_dataset(itype, 'Velocity', path),
            'GroupNumber': read_dataset(itype, 'GroupNumber', path),
            'SubGroupNumber': read_dataset(itype, 'SubGroupNumber', path)
        }

        data['coords'] = [data['coords'][i] for i in range(data['coords'].shape[0])]
        data['velocity'] = [data['velocity'][i] for i in range(data['velocity'].shape[0])]
        tmp = pd.DataFrame(data)
        data_filtered = tmp[(tmp['GroupNumber'] == group_number) & (tmp['SubGroupNumber'] == subgroup_id)]
        subhalo_particles = SubhaloParticles(data_filtered['mass'], data_filtered['coords'], data_filtered['velocity'],
                                             self.cfg.catalog.scalf)
        #log.debug("Particle data fetched successfully.")
        return subhalo_particles

