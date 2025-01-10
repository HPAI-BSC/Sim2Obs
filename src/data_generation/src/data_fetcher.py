import operator
import os.path
from abc import ABC, abstractmethod

import eagleSqlTools as sql
import numpy as np
import pandas as pd
import illustris_python as il
from omegaconf import DictConfig

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection, read_dataset, setup_logger


logger = setup_logger("DataFetcher")

operators = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne
}


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

    @staticmethod
    def filter_halos(halo_context: HaloContext, conditions) -> HaloContext:
        mask = pd.Series(True, index=halo_context.data.index)
        for feature, symbol, value in conditions:
            if feature not in halo_context.data:
                raise Exception(f'Error, feature "{feature}" is not present in the halo dataframe with columns:'
                                f' {halo_context.data.columns}')
            mask &= operators[symbol](halo_context.data[feature], value)

        halo_context.data = halo_context.data[mask]
        return halo_context
    @staticmethod
    def filter_subhalos(subhalo_context: SubhaloContext, conditions) -> SubhaloContext:
        """
        Abstract method to filter subhalos by a set of conditions
        """
        mask = pd.Series(True, index=subhalo_context.data.index)
        for feature, symbol, value in conditions:
            if feature not in subhalo_context.data:
                raise Exception(f'Error, feature "{feature}" is not present in the subhalo dataframe with columns:'
                                f' {subhalo_context.data.columns}')
            mask &= operators[symbol](subhalo_context.data[feature], value)

        subhalo_context.data = subhalo_context.data[mask]
        return subhalo_context


class DataFetcherFactory:
    """
    Factory class to instantiate the correct DataFetcher based on the configuration.
    """
    @staticmethod
    def get_data_fetcher(catalog: DictConfig, **kwargs) -> DataFetcher:
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
        logger.debug(f"Fetching data fetcher for catalog: {catalog.name}")
        if catalog.name == "illustris-300":
            return Illustris300Fetcher(catalog, **kwargs)
        elif catalog.name == "eagle":
            return EagleFetcher(catalog, **kwargs)
        else:
            raise ValueError(f"Unsupported catalog={catalog.name} type")


class Illustris300Fetcher(DataFetcher):
    """
    Data fetcher for the Illustris-300 catalog.
    """
    def __init__(self, catalog: DictConfig, **kwargs):
        """
        Initializes the fetcher with the given configuration.

        Parameters:
        -----------
        cfg : DictConfig
            Configuration object containing catalog settings.
        """
        self.catalog = catalog
        for key, value in kwargs.items():
            setattr(self, key, value)

        logger.info(f"Illustris300Fetcher initialized with config: {catalog}")

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """
        Fetches the halo data.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
        logger.info("Fetching halo data...")
        halo_context = self._get_halos()
        return halo_context

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        """
        Fetches the subhalo data.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        logger.info("Fetching subhalo data...")
        subhalo_context = self._get_subhalos()
        return subhalo_context

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
            logger.error("subhalo_id must be provided in kwargs")
            raise ValueError("subhalo_id must be provided in kwargs")

        logger.info(f"Fetching particle data for subhalo: {subhalo_id}")
        return self._get_p_subhalo(subhalo_id)

    def _get_halos(self) -> HaloContext:
        """
        Fetches halo data from the Illustris-300 simulation.

        Returns:
        --------
        HaloContext
            A context object containing halo data such as mass, radius, position, and subhalo index.
        """
        logger.debug("Fetching halos from Illustris-300...")

        # Units of measure -> [10^10𝑀⊙/ℎ, 𝑐𝑘𝑝𝑐/ℎ, 𝑐𝑘𝑝𝑐/ℎ, int]
        fields = ["Group_M_Crit200", "Group_R_Crit200", "GroupPos", "GroupFirstSub"]
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        halos = il.groupcat.loadHalos(path, self.catalog.snapnum, fields=fields)
        if len(halos) > 0:
            scalf = self.catalog.scalf
            halos['GroupPos'] = [halos['GroupPos'][i] * scalf for i in range(halos['GroupPos'].shape[0])]
            halos['Group_R_Crit200'] = halos['Group_R_Crit200'] * scalf
            halos = pd.DataFrame.from_dict(halos)
            halos['halo_id'] = np.arange(len(halos['Group_M_Crit200']))
            halos['Group_M_Crit200'] = halos['Group_M_Crit200'] * self.solar_mass_convention # 𝑀⊙/ℎ
            column_map = {'halo_id': 'halo_id', 'r200': 'Group_R_Crit200', 'm200': 'Group_M_Crit200',
                          'gpos': 'GroupPos', 'fsub': 'GroupFirstSub'}
            logger.debug("Halo data fetched successfully.")
            return HaloContext(halos, column_map)
        else:
            logger.warning("No halos found.")
            return HaloContext(pd.DataFrame(data=[]), {})

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetches subhalo data from the Illustris-300 simulation.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        logger.debug("Fetching subhalos from Illustris-300...")

        # Units of measure -> [𝑐𝑘𝑝𝑐/ℎ, int, 𝑐𝑘𝑝𝑐/ℎ, 10^10𝑀⊙/ℎ]
        fields = ["SubhaloPos", "SubhaloGrNr", "SubhaloHalfmassRadType", "SubhaloMassType"]
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        subhalos = il.groupcat.loadSubhalos(path, self.catalog.snapnum, fields=fields)
        if len(subhalos) > 0:
            subhalos['spos'] = subhalos['SubhaloPos']
            subhalos['SubhaloHalfmassRadType'] = subhalos["SubhaloHalfmassRadType"][:, 4]  # stars
            subhalos['SubhaloMassType'] = subhalos["SubhaloMassType"][:, 4] * self.solar_mass_convention / self.H # 𝑀⊙
            subhalos['subhalo_id'] = np.arange(len(subhalos['spos']))
            column_map = {"subhalo_id": "subhalo_id", "spos": "spos", "id_parents": "SubhaloGrNr",
                          "shmr": "SubhaloHalfmassRadType", "mstr": "SubhaloMassType"}
            logger.debug("Subhalo data fetched successfully.")
            subhalo_context = SubhaloContext(subhalos, self.catalog.scalf, column_map)
            return subhalo_context
        else:
            logger.warning("No subhalos found.")
            return SubhaloContext(pd.DataFrame(data=[]), self.catalog.scalf, {})

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
        logger.debug(f"Fetching particle data for subhalo: {subhalo_id}")

        # Units of measure [𝑐𝑘𝑝𝑐/ℎ, 𝑘𝑚𝑎‾√/𝑠, 10^10𝑀⊙/ℎ]
        fields = ["Coordinates", "Velocities", "Masses"]
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        star_p = il.snapshot.loadSubhalo(path, self.catalog.snapnum, subhalo_id, "star", fields=fields)
        if len(star_p) > 0:
            p = star_p["Coordinates"]
            v = star_p["Velocities"] * np.sqrt(self.catalog.scalf) # 𝑘𝑚𝑎‾√/𝑠 -> kms-1
            m = star_p["Masses"] * self.solar_mass_convention
            logger.debug(f"Particle data fetched successfully, {len(p)} particles retrieved")
        else:
            logger.warning(f"No particle data found for subhalo {subhalo_id}.")
            p = v = m = np.ndarray([])
        return SubhaloParticles(m, p, v, self.catalog.scalf)


class EagleFetcher(DataFetcher):
    """
    Data fetcher for the Eagle catalog.
    """
    def __init__(self, catalog: DictConfig, **kwargs):
        """
        Initializes the fetcher with the given configuration.

        Parameters:
        -----------
        cfg : DictConfig
            Configuration object containing catalog settings.
        """
        self.catalog = catalog
        for key, value in kwargs.items():
            setattr(self, key, value)
        logger.info(f"EagleFetcher initialized with config: {catalog}")

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """
        Fetches the halo data for Eagle catalog.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
        logger.info("Fetching halo data for Eagle catalog...")
        return self._get_halos()

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        """
        Fetches the subhalo data for Eagle catalog.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        logger.info("Fetching subhalo data for Eagle catalog...")
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
        group_number = kwargs.get('group_number')

        if subgroup_id is None:
            logger.error("subhalo_id must be provided in kwargs")
            raise ValueError("subgroup_id must be provided in kwargs")
        if group_number is None:
            logger.error("halo_id must be provided in kwargs")
            raise ValueError("group_number must be provided in kwargs")

        logger.info(f"Fetching particle data for subhalo: {subgroup_id}, Group_number: {group_number}")
        return self._get_p_subhalo(subgroup_id, group_number)

    def _get_halos(self) -> HaloContext:
        """
        Fetches halo data from the Eagle simulation.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
        logger.debug("Fetching halos from Eagle...")
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
                    {self.catalog.simulation_code}_FOF AS FOF -- Halo properties
                LEFT JOIN 
                    (
                        SELECT 
                            SH.GroupID,
                            SH.SubGroupNumber,
                            SH.GroupNumber, 
                            SH.MassType_Star
                        FROM 
                            {self.catalog.simulation_code}_Subhalo AS SH
                        WHERE 
                            SH.MassType_Star = (
                                SELECT MAX(SH2.MassType_Star) -- Ensure it's the largest subhalo by mass
                                FROM {self.catalog.simulation_code}_Subhalo AS SH2
                                WHERE SH2.GroupID = SH.GroupID AND SH.MassType_Star > {self.subhalo_min_mass}
                                AND SH.SnapNum = {self.catalog.snapnum} 
                            )
                    ) AS SH ON FOF.GroupID = SH.GroupID
                WHERE 
                    FOF.Group_M_Crit200 > {self.halo_min_mass}  -- Minimum halo mass threshold
                    AND FOF.SnapNum = {self.catalog.snapnum} -- Snapshot for redshift ~0.2
                """
        # Run the query
        array = sql.execute_query(conn, query)
        # Units of measure [m200=M⊙, r200=kpc/h, gpos=cMpc]
        df = pd.DataFrame(array)
        if len(df) > 0:
            df = df.sort_values(by='halo_id')
            # pkps -> ckps / h
            scalf = self.catalog.scalf
            df['gpos'] = df.apply(lambda row: np.array([(row['pos_x'] * scalf) / self.H,
                                                        (row['pos_y'] * scalf) / self.H,
                                                        (row['pos_z'] * scalf) / self.H]), axis=1)
            df['r200'] = df['r200'].values / (self.H * self.catalog.scalf)    # pkps -> kps / h
            df['m200'] = df['m200'].values / self.H    # 𝑀⊙/ℎ
            df['halo_id'] = df['halo_id'].astype(int)
            column_map = {'halo_id': 'halo_id', 'r200': 'r200', 'm200': 'm200', 'gpos': 'gpos', 'fsub': 'subhalo_id',
                          'group_number': 'group_number'}
            halo_context = HaloContext(df, column_map=column_map)
            logger.debug("Halo data fetched successfully.")
        else:
            logger.warning("No halo data found in Eagle database.")
            halo_context = HaloContext(pd.DataFrame(data=[]), column_map={})
        return halo_context

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetches subhalo data from the Eagle simulation.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        logger.debug("Fetching subhalos from Eagle...")
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
                {self.catalog.simulation_code}_FOF AS FOF -- Halo properties
            JOIN
                {self.catalog.simulation_code}_Subhalo AS SH -- Subhalo properties
            ON 
                FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
            WHERE
                SH.MassType_Star > {self.subhalo_min_mass} -- Minimum subhalo stellar mass threshold
                AND FOF.Group_M_Crit200 > {self.halo_min_mass} -- Minimum halo mass threshold
                AND SH.SnapNum = {self.catalog.snapnum} -- Snapshot for redshift ~0.2
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
            df['spos'] = df.apply(lambda row: np.array([row['subhalo_pos_x'] / self.H,
                                                        row['subhalo_pos_y'] / self.H,
                                                        row['subhalo_pos_z'] / self.H]), axis=1)
            df['stellar_half_mass_radius'] = df['stellar_half_mass_radius'].values / self.H  # pkps -> ckps / h
            column_map = {"subhalo_id": "subhalo_id", "spos": "spos", "id_parents": "halo_id",
                          "shmr": "stellar_half_mass_radius", "mstr": "total_stellar_mass",
                          "group_number": "group_number"}
            subhalo_context = SubhaloContext(df, self.catalog.scalf, column_map)
            logger.debug("Subhalo data fetched successfully.")
        else:
            logger.warning("No subhalo data found in Eagle database.")
            subhalo_context = SubhaloContext(pd.DataFrame(data=[]), self.catalog.scalf, {})
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
        itype = 4  # PartType
        path = self.catalog.path
        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
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

        if len(data_filtered) == 0:
            logger.error(f'Error, there is not particle data for GroupNumber={group_number} and '
                         f'SubGroupNumber={subgroup_id}')
            raise ValueError(f'Error, there is not particle data for GroupNumber={group_number} and'
                             f' SubGroupNumber={subgroup_id}')
        else:
            logger.debug(f"Particle data fetched successfully, {len(data_filtered)} particles retrieved.")

        velocity = np.vstack(data_filtered['velocity'].values)
        positions = np.vstack(data_filtered['coords'].values)
        subhalo_particles = SubhaloParticles(data_filtered['mass'].values, positions, velocity, self.catalog.scalf)
        return subhalo_particles

