import copy
import operator
import os.path
from abc import ABC, abstractmethod
from typing import Any
import swiftsimio as sw

import eagleSqlTools as sql
import numpy as np
import pandas as pd
import illustris_python as il
from omegaconf import DictConfig, ListConfig, OmegaConf
from swiftsimio import SWIFTDataset

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection, read_dataset, setup_logger, to_list, apply_conversion, \
    dict_depth, to_list_in_batches

logger = setup_logger("DataFetcher")

operators = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne
}


def pipeline_step(func):
    """Decorator to apply a transformation in a pipeline sequence."""
    def wrapper(data, *args, **kwargs):
        if data is None:
            return None  # Stop processing if data is None
        return func(data, *args, **kwargs)
    return wrapper


@pipeline_step
def assign_id(d: dict, id_column: str):
    """Assign an ID to each subhalo/halo."""
    if len(d) == 0:
        return None
    d[id_column] = np.arange(d['count'])
    return d


@pipeline_step
def validate_and_process_fields(d: dict, fields: list, table: str, cfg: DictConfig):
    """Validates fields, applies indexing, and converts multi-dimensional arrays."""

    for key in list(d.keys()):
        if key in fields:
            field_info = OmegaConf.select(cfg, f"fields.{table}.{key}")
            # Halo / subhalos case
            if dict_depth(field_info) == 1:
                if not getattr(field_info, 'name', None):
                    raise ValueError(f"Error: The field '{key}' must have a name.")
                index = field_info.get("index", None)
                if index is not None:
                    if isinstance(index, (int, list, ListConfig)):
                        d[key] = d[key][:, index]

                if isinstance(d[key], np.ndarray) and d[key].ndim > 1:
                    agg_func = field_info.get("agg", None)
                    if agg_func is None:
                        d[key] = to_list(d, key)
                    else:
                        result = getattr(np, agg_func)(d[key], axis=1)
                        d[key] = result

            # Particle case
            elif dict_depth(field_info) == 2:
                for key_ii in OmegaConf.select(cfg, f"fields.{table}.{key}"):
                    field_info_ii = OmegaConf.select(cfg, f"fields.{table}.{key}.{key_ii}")
                    h = dict_depth(field_info_ii)
                    if h == 1:
                        if not getattr(field_info_ii, 'name', None):
                            raise ValueError(f"Error: The field '{key}' must have a name.")
                        index = field_info_ii.get("index", None)
                        if index is not None:
                            d[key][key_ii] = d[key][key_ii][:, index]
                        if isinstance(d[key][key_ii], np.ndarray) and d[key][key_ii].ndim > 1:
                            agg_func = field_info.get("agg", None)
                            if agg_func is None:
                                d[key][key_ii] = to_list(d[key], key_ii)
                            else:
                                result = getattr(np, agg_func)(d[key], axis=1)
                                d[key][key_ii] = result
    return d


@pipeline_step
def convert_to_dataframe(d: dict):
    """Convert dictionary to Pandas DataFrame."""
    if dict_depth(d) == 1:
        return pd.DataFrame.from_dict(d)
    elif dict_depth(d) == 2:
        for key in d:
            d[key] = pd.DataFrame.from_dict(d[key])
        return d
    else:
        raise ValueError(f'The format of the data is not correct, {dict_depth(d)} levels of data detected.')


def apply_filter_df(d, conditions, config):
    if isinstance(d, pd.DataFrame):
        mask = pd.Series(True, index=d.index)
        for key, condition in conditions.items():
            if key not in d.columns:
                raise Exception(f'Error, feature "{key}" is not present in the halo dataframe with columns:'
                                f' {d.columns}')

            condition = OmegaConf.select(config, key)

            if isinstance(condition.filter, ListConfig):
                for expr in condition.filter:
                    if isinstance(expr, str):
                        symbol = expr.split(' ')[0]
                        right_expr = ' '.join(expr.split(' ')[1:])
                        value = float(eval(right_expr))
                        mask &= operators[symbol](d[key], value)
            else:
                if isinstance(condition.filter, str):
                    symbol = condition.filter.split(' ')[0]
                    right_expr = ' '.join(condition.filter.split(' ')[1:])
                    value = float(eval(right_expr))
                    mask &= operators[symbol](d[key], value)
        return d[mask]


@pipeline_step
def apply_filters(d: Any, conditions: dict, config: DictConfig):
    if isinstance(d, pd.DataFrame):
        return apply_filter_df(d, conditions, config)
    elif isinstance(d, dict):
        for key in d:
            d[key] = apply_filter_df(d[key], conditions.get(key, {}), config)
        return d


@pipeline_step
def apply_unit_conversion(df: Any, fields: list, table: str, cfg: DictConfig):
    """Apply unit conversions based on the catalog scaling factors."""
    for key in fields:
        if key in df:
            field_info = OmegaConf.select(cfg, f"fields.{table}.{key}")
            height = dict_depth(field_info)
            if height == 2:
                for key_ii in field_info:
                    field_info_ii = OmegaConf.select(cfg, f"fields.{table}.{key}.{key_ii}")
                    if field_info_ii.scaling is not None:
                        df[key][key_ii] = apply_conversion(df[key][key_ii], field_info_ii.scaling, cfg.a, cfg.H)
            else:
                if field_info.scaling is not None:
                    df[key] = apply_conversion(df[key], field_info.scaling, cfg.a, cfg.H)
    return df


def run_pipeline(data, steps):
    """Executes a list of transformation steps sequentially."""
    for step in steps:
        data = step(data)  # Each function modifies `data`
        if data is None:
            break  # Stop early if a step returns None
    return data


class DataFetcher(ABC):
    """
    Abstract base class for fetching data from various catalogs.
    """
    def __init__(self, catalog: DictConfig, **kwargs):
        self.catalog = catalog
        self.__build_config(**kwargs)

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

    def _get_fields(self):
        pass

    def __get_attr_recursive(self, config, attr_name, parent_key="", branch={}):
        for key, value in config.items():
            full_key = f"{parent_key}{key}" if parent_key else key  # Construct the full key
            if dict_depth(value) > 1:
                self.__get_attr_recursive(value, attr_name, parent_key=f"{full_key}.", branch=branch)
            else:
                branch[full_key] = value.get(attr_name)
        return branch

    def _get_attr(self, config, attr_name, parent_key=""):
        dict_branchs = {}
        for key, value in config.items():
            full_key = f"{parent_key}{key}" if parent_key else key  # Construct the full key
            if dict_depth(value) > 1:
                branch = {}
                self.__get_attr_recursive(value, parent_key=f"{full_key}.", attr_name=attr_name, branch=branch)
                dict_branchs.update(branch)
            else:
                # non-recursive case
                attr_value = value.get(attr_name)
                dict_branchs[full_key] = attr_value
        return dict_branchs

    def __explore_branch(self, config, parent_key=""):
        """Recursively get all keys up to the first level of fields."""
        keys = []
        for key, value in config.items():
            full_key = f"{parent_key}{key}" if parent_key else key  # Construct the full key
            if dict_depth(value) > 1:
                keys.append(full_key)
                self._get_nested_keys(value, parent_key=f"{full_key}.")
            else:
                keys.append(full_key)
        return keys

    def _get_nested_keys(self, config, parent_key=""):
        """Recursively get all keys up to the first level of fields."""
        keys_branchs = []
        for key, value in config.items():
            full_key = f"{parent_key}{key}" if parent_key else key  # Construct the full key
            if dict_depth(value) > 1:
                keys = self.__explore_branch(value, parent_key=f"{full_key}.")
                keys_branchs.extend(keys)
            else:
                # Non recursive case
                keys_branchs.append(full_key)
        return keys_branchs

    def __build_config(self, **kwargs):
        self.fields = self._get_fields()

        fields = getattr(self.catalog, "fields", None)
        if not fields:
            raise ValueError("The catalog must contain a 'fields' attribute.")

        self.halo_fields = self._get_nested_keys(self.catalog.fields.halo)
        self.subhalo_fields = self._get_nested_keys(self.catalog.fields.subhalo)
        particle_fields = self._get_nested_keys(self.catalog.fields.particles)
        self.particle_fields = set([p.split('.')[0] for p in particle_fields])

        kv_halo = self._get_attr(self.catalog.fields.halo, attr_name='name')
        kv_subhalo = self._get_attr(self.catalog.fields.halo, attr_name='name')
        kv_particles = self._get_attr(self.catalog.fields.particles, attr_name='name')

        self.column_map = kv_halo | kv_subhalo | kv_particles

        for key, value in kwargs.items():
            setattr(self, key, value)


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
        elif catalog.name == "flamingo":
            return FlamingoFetcher(catalog, **kwargs)
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
        super().__init__(catalog, **kwargs)
        logger.info(f"Illustris300Fetcher initialized with config: {catalog}")

    def _get_fields(self):
        path = self.catalog.path + '/output'
        halo_fields = list(il.groupcat.loadHalos(path, self.catalog.snapnum, fields=None).keys())
        subhalo_fields = list(il.groupcat.loadSubhalos(path, self.catalog.snapnum, fields=None).keys())
        return halo_fields + subhalo_fields

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
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        if 'GroupFirstSub' not in self.halo_fields:
            self.halo_fields.append('GroupFirstSub')
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.halo["GroupFirstSub"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "fsub"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map['GroupFirstSub'] = 'fsub'

        halos = il.groupcat.loadHalos(path, self.catalog.snapnum, fields=self.halo_fields)
        if len(halos) > 0:
            table = 'halo'
            halo_conditions = {col: data for col, data in self.catalog.fields.halo.items() if 'filter' in data}
            steps = [
                lambda halos: assign_id(halos, 'halo_id'),
                lambda halos: validate_and_process_fields(halos, self.halo_fields, table, self.catalog),
                lambda halos: convert_to_dataframe(halos),
                lambda halos: apply_filters(halos, halo_conditions, self.catalog.fields.halo),
                lambda halos: apply_unit_conversion(halos, self.halo_fields, table, self.catalog)
            ]

            processed_halos = run_pipeline(halos, steps)
            logger.debug("Halo data fetched successfully.")
            halo_context = HaloContext(processed_halos, self.column_map)
            return halo_context
        else:
            logger.warning("No halos found.")
            return HaloContext(pd.DataFrame(data=[]), self.column_map)

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetches subhalo data from the Illustris-300 simulation.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        logger.debug("Fetching subhalos from Illustris-300...")

        fields = self.subhalo_fields
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        if 'SubhaloGrNr' not in self.halo_fields:
            self.subhalo_fields.append('SubhaloGrNr')
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.subhalo["SubhaloGrNr"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "id_parents"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map['SubhaloGrNr'] = 'id_parents'

        subhalos = il.groupcat.loadSubhalos(path, self.catalog.snapnum, fields=fields)

        if isinstance(subhalos, np.ndarray):
            tmp = subhalos
            subhalos = {self.fields[0]: tmp}
            size = len(subhalos[self.fields[0]])
            subhalos['count'] = np.array(size)

        if len(subhalos) > 0:
            table = 'subhalo'
            subhalo_conditions = {col: data for col, data in self.catalog.fields.subhalo.items() if 'filter' in data}
            steps = [
                lambda subhalos: assign_id(subhalos, 'subhalo_id'),
                lambda subhalos: validate_and_process_fields(subhalos, self.subhalo_fields, table, self.catalog),
                lambda subhalos: convert_to_dataframe(subhalos),
                lambda subhalos: apply_filters(subhalos, subhalo_conditions, self.catalog.fields.subhalo),
                lambda subhalos: apply_unit_conversion(subhalos, self.subhalo_fields, table, self.catalog)
            ]

            processed_subhalos = run_pipeline(subhalos, steps)
            logger.debug("Subhalo data fetched successfully.")
            subhalo_context = SubhaloContext(processed_subhalos, self.column_map)
            return subhalo_context
        else:
            logger.warning("No subhalos found.")
            return SubhaloContext(pd.DataFrame(data=[]), self.column_map)

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
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        try:
            part_types = self.particle_fields
            data = {}
            for part_type in part_types:
                fields = list(self.catalog.fields.particles[part_type].keys())
                particles = il.snapshot.loadSubhalo(path, self.catalog.snapnum, subhalo_id, part_type, fields=fields)

                # if numpy, convert it to a dict
                if isinstance(particles, np.ndarray):
                    tmp = particles
                    particles = {fields[0]: tmp}

                if len(particles) > 0:
                    data[part_type] = particles
                    logger.debug(f"Particle data fetched successfully, particles retrieved")
                else:
                    logger.warning(f"No particle data found for subhalo {subhalo_id}.")

            particle_conditions = {}
            for col, type_config in self.catalog.fields.particles.items():
                for t in type_config:
                    if 'filter' in type_config[t]:
                        if col not in particle_conditions:
                            particle_conditions[col] = {}
                        particle_conditions[col][t] = type_config[t]

            table = 'particles'
            steps = [
                lambda particles: validate_and_process_fields(particles, self.particle_fields, table,
                                                              self.catalog),
                lambda particles: convert_to_dataframe(particles),
                lambda particles: apply_filters(particles, particle_conditions, self.catalog.fields.particles),
                lambda particles: apply_unit_conversion(particles, self.particle_fields, table, self.catalog)
            ]

            processed_particles = run_pipeline(data, steps)
            logger.debug("Halo data fetched successfully.")
            particle_data = SubhaloParticles(processed_particles, self.catalog.fields.particles)
            return particle_data
        except Exception as e:
            raise ValueError(f"Error, Particle data could not be retrieved for subhalo_id: {subhalo_id}\n {e}")


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
        super().__init__(catalog, **kwargs)
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

    def _get_fields(self):
        pass

    def _build_query(self, table):
        def _build_conditions(conditions, table) -> str:
            expr = ''
            for key, value in conditions.items():
                if isinstance(value.filter, ListConfig):
                    for elem in value.filter:
                        symbol = elem.split(' ')[0]
                        value = eval(elem.split(' ')[1])
                        expr += f'{table}.{key} {symbol} {value} AND '
                elif isinstance(value.filter, str):
                    symbol = value.filter.split(' ')[0]
                    value = eval(value.filter.split(' ')[1])
                    expr += f'{table}.{key} {symbol} {value} AND '
            return expr[:-4]

        query = None
        halo_conditions = {col: data for col, data in self.catalog.fields.halo.items() if 'filter' in data}
        halo_conditions = _build_conditions(halo_conditions, table='FOF')

        subhalo_conditions = {col: data for col, data in self.catalog.fields.subhalo.items() if 'filter' in data}
        subhalo_conditions = _build_conditions(subhalo_conditions, table='SH')

        if table == 'halo':
            header = '''
            SELECT 
            FOF.GroupID AS halo_id,                          -- Unique identifier for each halo
            '''
            for column in self.halo_fields:
                if column not in header:
                    header += f'FOF.{column} AS {column},                     -- ... \n'

            query = f"""
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
                                            WHERE SH2.GroupID = SH.GroupID
                                            AND SH.SnapNum = {self.catalog.snapnum} 
                                            AND {subhalo_conditions}
                                        )
                                ) AS SH ON FOF.GroupID = SH.GroupID
                            WHERE 
                                FOF.SnapNum = {self.catalog.snapnum} -- Snapshot for redshift ~0.2
                                AND {halo_conditions}
                            """
            query = header + query
            return query
        elif table == 'subhalo':
            header = '''
            SELECT
            FOF.GroupID AS halo_id, -- Halo identifier
            SH.SubGroupNumber AS subhalo_id, -- Subgroup
            SH.GroupNumber AS GroupNumber, -- ID FoF, for linking particles
            '''

            for i, column in enumerate(self.subhalo_fields):
                if column not in header:
                    header += f'SH.{column} AS {column},                     -- ... \n'

            query = f"""
                        FOF.Group_M_Crit200 AS m200 -- Halo mass within R200
                        FROM
                            {self.catalog.simulation_code}_FOF AS FOF -- Halo properties
                        JOIN
                            {self.catalog.simulation_code}_Subhalo AS SH -- Subhalo properties
                        ON 
                            FOF.GroupID = SH.GroupID -- Match subhalos to their parent halos
                        WHERE
                            SH.SnapNum = {self.catalog.snapnum} -- Snapshot for redshift ~0.2
                            AND {halo_conditions} AND {subhalo_conditions}
                        ORDER BY
                            SH.SubGroupNumber ASC -- Order results by subhalo id
                        """
            query = header + query
        return query

    def _get_halos(self) -> HaloContext:
        """
        Fetches halo data from the Eagle simulation.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
        logger.debug("Fetching halos from Eagle...")

        path = os.path.join(self.catalog.path, 'output')

        if not os.path.exists(path):
            logger.info(f"Output folder does not exist, retrieving data from db ...")
            conn = get_connection()
            query = self._build_query(table='halo')

            # Run the query
            array = sql.execute_query(conn, query)
            df = pd.DataFrame(array)

            os.mkdir(path)
            df.to_csv(os.path.join(path, 'fof.csv'), index=False)
        else:
            logger.info(f"Reading from files ...")
            df = pd.read_csv(os.path.join(path, 'fof.csv'))

        if 'GroupNumber' not in self.halo_fields:
            self.subhalo_fields.append('GroupNumber')
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.subhalo["GroupNumber"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "group_number"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map['GroupNumber'] = 'group_number'

        if len(df) > 0:
            df = df.sort_values(by='halo_id')
            df['halo_id'] = df['halo_id'].astype(int)
            for column in list(df.columns):
                # Unit conversion
                if column in self.halo_fields:
                    if self.catalog.fields.halo[column].scaling is not None:
                        df[column] = apply_conversion(df[column], self.catalog.fields.halo[column].scaling,
                                                      self.catalog.a, self.catalog.H)

            halo_context = HaloContext(df, column_map=self.column_map)
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

        path = os.path.join(self.catalog.path, 'output')
        if not os.path.exists(path):
            logger.info(f"Output folder does not exist, retrieving data from db ...")
            conn = get_connection()
            query = self._build_query(table='subhalo')

            # Run the query
            array = sql.execute_query(conn, query)
            df = pd.DataFrame(array)

            os.mkdir(path)
            df.to_csv(os.path.join(path, 'subhalo.csv'), index=False)
        else:
            logger.info(f"Reading from files ...")
            df = pd.read_csv(os.path.join(path, 'subhalo.csv'))

        # Remove useless columns
        df = df.drop(columns=['m200'])
        df = df.rename(columns={'halo_id': 'id_parents'})

        if 'GroupNumber' not in self.halo_fields:
            self.subhalo_fields.append('GroupNumber')
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.subhalo["GroupNumber"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "group_number"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map['GroupNumber'] = 'group_number'

        if 'SubGroupNumber' not in self.halo_fields:
            self.subhalo_fields.append('SubGroupNumber')
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.subhalo["SubGroupNumber"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "subhalo_id"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map['SubGroupNumber'] = 'subhalo_id'

        if len(df) > 0:
            df = df.sort_values(by='subhalo_id')
            for column in list(df.columns):
                # Unit conversion
                if column in self.subhalo_fields:
                    if self.catalog.fields.subhalo[column].scaling is not None:
                        df[column] = apply_conversion(df[column], self.catalog.fields.subhalo[column].scaling,
                                                      self.catalog.a, self.catalog.H)
            df = df.sort_values(by=['subhalo_id', 'GroupNumber'])
            subhalo_context = SubhaloContext(df, self.column_map)
            logger.debug("Subhalo data fetched successfully.")
        else:
            logger.warning("No subhalo data found in Eagle database.")
            subhalo_context = SubhaloContext(pd.DataFrame(data=[]), self.column_map)
        return subhalo_context

    @staticmethod
    def __partTypeNum(partType):
        """ Mapping between common names and numeric particle types. """
        if str(partType).isdigit():
            return int(partType)

        if str(partType).lower() in ['gas', 'cells']:
            return 0
        if str(partType).lower() in ['dm', 'darkmatter']:
            return 1
        if str(partType).lower() in ['dmlowres']:
            return 2  # only zoom simulations, not present in full periodic boxes
        if str(partType).lower() in ['tracer', 'tracers', 'tracermc', 'trmc']:
            return 3
        if str(partType).lower() in ['star', 'stars', 'stellar']:
            return 4  # only those with GFM_StellarFormationTime>0
        if str(partType).lower() in ['wind']:
            return 4  # only those with GFM_StellarFormationTime<0
        if str(partType).lower() in ['bh', 'bhs', 'blackhole', 'blackholes']:
            return 5
        raise Exception("Unknown particle type name.")

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
        try:
            path = self.catalog.path
            part_types = self.particle_fields
            data = {}
            for part_type in part_types:
                data_part_type = {}
                itype = self.__partTypeNum(part_type)
                OmegaConf.set_struct(self.catalog, False)
                self.catalog.fields.particles[part_type]["GroupNumber"] = {
                    "input": None,
                    "output": None,
                    "scaling": None,
                    "name": "group_number"
                }
                self.catalog.fields.particles[part_type]["SubGroupNumber"] = {
                    "input": None,
                    "output": None,
                    "scaling": None,
                    "name": "subhalo_id"
                }
                OmegaConf.set_struct(self.catalog, True)

                fields = list(self.catalog.fields.particles[part_type].keys())
                if not os.path.exists(path):
                    logger.error(f"The path {path} does not exist.")
                    raise Exception(f'The path {path} does not exist.')

                for field in fields:
                    column = read_dataset(itype, field, path)
                    if self.catalog.fields.particles[part_type][field].scaling is not None:
                        scaling = self.catalog.fields.particles[part_type][field].scaling
                        column = apply_conversion(column, scaling, self.catalog.a, self.catalog.H,
                                                  g=self.catalog.g, g_2_solar=self.catalog.g_2_solar)
                    data_part_type[field] = column
                    # TODO apply filters

                    # Dealing with numpy arrays of > 1 dim for dataframe conversion afterward
                    if column.ndim > 1:
                        converted = to_list(data_part_type, field)
                        del data_part_type[field]
                        data_part_type[field] = converted

                tmp = pd.DataFrame(data_part_type)
                # Create index with these two columns for speed
                tmp = tmp[(tmp["GroupNumber"] == group_number) & (tmp["SubGroupNumber"] == subgroup_id)]
                if len(tmp) == 0:
                    logger.error(f'Error, there is not particle data for GroupNumber={group_number} and '
                                 f'SubGroupNumber={subgroup_id}')
                    raise ValueError(f'Error, there is not particle data for GroupNumber={group_number} and'
                                     f' SubGroupNumber={subgroup_id}')
                data[part_type] = tmp.to_dict(orient='list')
            else:
                logger.debug(f"Particle data fetched successfully, {len(tmp)} particles retrieved.")

            subhalo_particles = SubhaloParticles(data, self.catalog.fields.particles)
            return subhalo_particles

        except Exception as e:
            raise Exception(f'Something went wrong: {e}')


class FlamingoFetcher(DataFetcher):
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

    @staticmethod
    def __convert_keys(d):
        new_dict = {}

        for key, value in d.items():
            # Split the key by the '.' separator to get the parts
            keys = key.split('.')

            # Traverse and create nested dictionaries
            temp = new_dict
            for part in keys[:-1]:
                temp = temp.setdefault(part, {})

            # Assign the value to the final key
            temp[keys[-1]] = value

        return new_dict

    def __validate_and_process_fields(self, soap: SWIFTDataset, fields, super_key):
        data = {}
        for attr in fields:
            field_name = attr
            try:
                obj = copy.copy(soap)
                cfg = None
                for part in attr.split('.'):
                    if cfg is None:
                        key = f"{super_key}.{part}"
                        cfg = OmegaConf.select(self.catalog.fields, key, default=None)
                    else:
                        cfg = cfg[part]
                    if hasattr(obj, part):
                        attr = getattr(obj, part)
                        obj = attr
                units = cfg.input
                data[field_name] = obj

                if units is not None:
                    data[field_name].convert_to_units(units)
                    # Specify physical/comoving
                    data[field_name].convert_to_physical()

                data[field_name] = data[field_name].value
                # data[field_name] = data[field_name][:100]
                if isinstance(data[field_name], np.ndarray) and data[field_name].ndim > 1:
                    data[field_name] = to_list_in_batches(data, field_name)
            # TODO
            except Exception as e:
                print()
        data = self.__convert_keys(data)
        return data

    def _get_halos(self) -> HaloContext:
        """
        Fetches halo data from the Flamingo simulation.

        Returns:
        --------
        HaloContext
            A context object containing halo data such as mass, radius, position, and subhalo index.
        """
        logger.debug("Fetching halos from Flamingo...")
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        # TODO change the attribute
        if 'input_halos.halo_catalogue_index' not in self.halo_fields:
            self.halo_fields.append('input_halos.halo_catalogue_index')
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.halo["input_halos.halo_catalogue_index"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "fsub"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map['input_halos.halo_centre'] = 'fsub'

        redshift = str(self.catalog.snapnum).zfill(4)
        path = path + f'/halo_properties_{redshift}.hdf5'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        h_properties = sw.load(path)
        table = 'halo'
        if isinstance(h_properties, SWIFTDataset):
            halos = self.__validate_and_process_fields(h_properties, self.halo_fields, super_key='halo')

            halo_conditions = self._get_attr(self.catalog.fields.halo, attr_name='filter', parent_key="")
            filtered_conditions = {k: v for k, v in halo_conditions.items() if v is not None}
            steps = [
                lambda halos: convert_to_dataframe(halos),
                lambda halos: apply_filters(halos, filtered_conditions, self.catalog.fields.halo),
                lambda halos: apply_unit_conversion(halos, self.halo_fields, table, self.catalog)
            ]

            processed_halos = run_pipeline(halos, steps)
            logger.debug("Halo data fetched successfully.")
            halo_context = HaloContext(processed_halos, self.column_map)
            return halo_context

        else:
            logger.warning("No halos found.")
            return HaloContext(pd.DataFrame(data=[]), self.column_map)

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetches subhalo data from the Illustris-300 simulation.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        logger.debug("Fetching subhalos from Illustris-300...")

        fields = self.subhalo_fields
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        if 'SubhaloGrNr' not in self.halo_fields:
            self.subhalo_fields.append('SubhaloGrNr')
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.subhalo["SubhaloGrNr"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "id_parents"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map['SubhaloGrNr'] = 'id_parents'

        subhalos = il.groupcat.loadSubhalos(path, self.catalog.snapnum, fields=fields)

        if isinstance(subhalos, np.ndarray):
            tmp = subhalos
            subhalos = {self.fields[0]: tmp}
            size = len(subhalos[self.fields[0]])
            subhalos['count'] = np.array(size)

        if len(subhalos) > 0:
            table = 'subhalo'
            subhalo_conditions = {col: data for col, data in self.catalog.fields.subhalo.items() if 'filter' in data}
            steps = [
                lambda subhalos: assign_id(subhalos, 'subhalo_id'),
                lambda subhalos: validate_and_process_fields(subhalos, self.subhalo_fields, table, self.catalog),
                lambda subhalos: convert_to_dataframe(subhalos),
                lambda subhalos: apply_filters(subhalos, subhalo_conditions, self.catalog.fields.subhalo),
                lambda subhalos: apply_unit_conversion(subhalos, self.subhalo_fields, table, self.catalog)
            ]

            processed_subhalos = run_pipeline(subhalos, steps)
            logger.debug("Subhalo data fetched successfully.")
            subhalo_context = SubhaloContext(processed_subhalos, self.column_map)
            return subhalo_context
        else:
            logger.warning("No subhalos found.")
            return SubhaloContext(pd.DataFrame(data=[]), self.column_map)

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
        redshift = str(self.catalog.snapnum).zfill(4)
        path = self.catalog.path + f'/output/flamingo_{redshift}.hdf5'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        try:
            particles = sw.load(path)
            nested_particle_fields = self._get_nested_keys(self.catalog.fields.particles)
            data = self.__validate_and_process_fields(particles, nested_particle_fields, super_key='particles')

            if len(data) > 0:
                logger.debug(f"Particle data fetched successfully, particles retrieved")
            else:
                logger.warning(f"No particle data found for subhalo {subhalo_id}.")

            particle_conditions = self._get_attr(self.catalog.fields.particles, attr_name='filter', parent_key="")
            filtered_conditions = {k: v for k, v in particle_conditions.items() if v is not None}
            table = 'particles'
            steps = [
                lambda particles: convert_to_dataframe(data),
                lambda particles: apply_filters(particles, filtered_conditions, self.catalog.fields.particles),
                lambda particles: apply_unit_conversion(particles, self.particle_fields, table, self.catalog)
            ]

            processed_particles = run_pipeline(data, steps)
            logger.debug("Halo data fetched successfully.")
            particle_data = SubhaloParticles(processed_particles, self.catalog.fields.particles)
            return particle_data
        except Exception as e:
            raise ValueError(f"Error, Particle data could not be retrieved for subhalo_id: {subhalo_id}\n {e}")




