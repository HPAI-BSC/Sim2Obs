import operator
import os.path
from abc import ABC, abstractmethod
from typing import Any

import eagleSqlTools as sql
import numpy as np
import pandas as pd
import polars as pl
import illustris_python as il
from omegaconf import DictConfig, ListConfig, OmegaConf

from src.data_generation.src.halo import HaloContext
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection, read_dataset, setup_logger, to_list, apply_conversion, \
    dict_depth

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


def apply_filter_df(d, conditions):
    if isinstance(d, pd.DataFrame):
        mask = pd.Series(True, index=d.index)
        for key, condition in conditions.items():
            if key not in d.columns:
                raise Exception(f'Error, feature "{key}" is not present in the halo dataframe with columns:'
                                f' {d.columns}')
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
def apply_filters(d: Any, conditions: dict):
    if isinstance(d, pd.DataFrame):
        return apply_filter_df(d, conditions)
    elif isinstance(d, dict):
        for key in d:
            d[key] = apply_filter_df(d[key], conditions.get(key, {}))
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

    def __build_config(self, **kwargs):
        self.fields = self._get_fields()

        fields = getattr(self.catalog, "fields", None)
        if not fields:
            raise ValueError("The catalog must contain a 'fields' attribute.")

        self.halo_fields = list(self.catalog.fields.halo.keys())
        self.subhalo_fields = list(self.catalog.fields.subhalo.keys())
        self.particle_fields = list(self.catalog.fields.particles.keys())
        self.column_map = {
            column: data['name']
            for group in [self.catalog.fields.halo, self.catalog.fields.subhalo]
            for column, data in group.items()
        }

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
                lambda halos: apply_filters(halos, halo_conditions),
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
                lambda subhalos: apply_filters(subhalos, subhalo_conditions),
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
                lambda particles: apply_filters(particles, particle_conditions),
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
        conn = get_connection()
        query = self._build_query(table='halo')

        # Run the query
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)

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
        conn = get_connection()
        query = self._build_query(table='subhalo')

        # Run the query
        array = sql.execute_query(conn, query)
        df = pd.DataFrame(array)

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
                    # TODO deal with the conversion !!
                    column = read_dataset(itype, field, path)
                    if self.catalog.fields.particles[part_type][field].scaling is not None:
                        scaling = self.catalog.fields.particles[part_type][field].scaling
                        column = apply_conversion(column, scaling, self.catalog.a, self.catalog.H)
                    data_part_type[field] = column

                    # Dealing with numpy arrays of > 1 dim for dataframe conversion afterward
                    if column.ndim > 1:
                        converted = to_list(data_part_type, field)
                        del data_part_type[field]
                        data_part_type[field] = converted

                tmp = pl.DataFrame(data_part_type)
                # Create index with these two columns for speed
                tmp = tmp.filter((tmp["GroupNumber"] == group_number) & (tmp["SubGroupNumber"] == subgroup_id))
                if len(tmp) == 0:
                    logger.error(f'Error, there is not particle data for GroupNumber={group_number} and '
                                 f'SubGroupNumber={subgroup_id}')
                    raise ValueError(f'Error, there is not particle data for GroupNumber={group_number} and'
                                     f' SubGroupNumber={subgroup_id}')
                data[part_type] = tmp.to_dict()
            else:
                logger.debug(f"Particle data fetched successfully, {len(tmp)} particles retrieved.")

            subhalo_particles = SubhaloParticles(data, self.catalog.fields.particles)
            return subhalo_particles

        except Exception as e:
            raise Exception(f'Something went wrong: {e}')






