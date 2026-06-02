import copy
import gc
import operator
import os.path
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import g3read
from abc import ABC, abstractmethod
from typing import Any, Dict

import swiftsimio as sw

import eagleSqlTools as sql
import numpy as np
import pandas as pd
import illustris_python as il
from omegaconf import DictConfig, ListConfig, OmegaConf
from swiftsimio import SWIFTDataset

from src.data_generation.src.feature_computer import FeatureComputer
from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.lazy_subhalo import LazySubhaloContext, LazySubhaloParticles
from src.data_generation.src.part_types import MAGNETICUM_PART_TYPES
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import get_connection, read_dataset, setup_logger, to_list, apply_conversion, \
    dict_depth, to_list_in_batches
import src.data_generation.src.features_dataset


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
    """
    Decorator to apply a transformation in a pipeline sequence.

    Parameters:
    -----------
    func : function
        The function to be wrapped as a pipeline step.

    Returns:
    --------
    function
        A wrapper function that applies the transformation.
    """
    def wrapper(data, *args, **kwargs):
        if data is None:
            return None  # Stop processing if data is None
        return func(data, *args, **kwargs)
    return wrapper


@pipeline_step
def remove_useless_columns(d: dict, fields: list):
    diff = set(d.keys()).difference(set(fields))

    for key in diff:
        if key not in fields:
            del d[key]
    return d


@pipeline_step
def assign_id(d: dict, id_column: str, offset=0):
    """
    Assign an ID to each subhalo/halo.

    Parameters:
    -----------
    d : dict
        Dictionary containing the dataset.
    id_column : str
        Column name where IDs will be assigned.

    Returns:
    --------
    dict or None
        Updated dictionary with assigned IDs, or None if the input is empty.
    """
    if len(d) == 0:
        return None

    if 'count' in d:
        d[id_column] = np.arange(d['count'])
    else:
        first_key = next(iter(d))
        num = len(d[first_key])
        d[id_column] = np.arange(offset, offset + num)
    return d


@pipeline_step
def validate_and_process_fields(d: dict, fields: list, table: str, cfg: DictConfig):
    """
    Validates fields, applies indexing, and converts multi-dimensional arrays.

    Parameters:
    -----------
    d : dict
        Dictionary containing the dataset.
    fields : list
        List of fields to validate and process.
    table : str
        Name of the table being processed.
    cfg : DictConfig
        Configuration object defining field properties.

    Returns:
    --------
    dict
        Processed dataset with indexed and converted fields.
    """
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
                tmp = OmegaConf.select(cfg, f"fields.{table}.{key}")
                for key_ii in OmegaConf.select(cfg, f"fields.{table}.{key}"):
                    # Rename the column in case the initial config format was not appropiate, ie: POS to 'POS '
                    field_info_ii = OmegaConf.select(cfg, f"fields.{table}.{key}.{key_ii}")
                    h = dict_depth(field_info_ii)
                    if cfg.name == 'magneticum':
                        key_ii = key_ii.ljust(4)
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
    """
    Convert dictionary to Pandas DataFrame.

    Parameters:
    -----------
    d : dict
        Dictionary containing the dataset.

    Returns:
    --------
    pd.DataFrame or dict
        Converted DataFrame if input is 1-level deep,
        dictionary of DataFrames if 2 levels deep,
        otherwise raises a ValueError.
    """
    if dict_depth(d) == 1:
        return pd.DataFrame.from_dict(d)
    elif dict_depth(d) == 2:
        for key in d:
            d[key] = pd.DataFrame.from_dict(d[key])
        return d
    else:
        raise ValueError(f'The format of the data is not correct, {dict_depth(d)} levels of data detected.')

# TODO TEST
def apply_filter_arrow(table: pa.Table, conditions: dict, config):
    """
    Arrow-native filtering using boolean masks.
    """

    mask = pa.array([True] * table.num_rows)

    for key, condition in conditions.items():

        if key not in table.column_names:
            raise Exception(
                f'Feature "{key}" not in table. Columns: {table.column_names}'
            )

        col = table[key]

        # MULTI-CONDITION SUPPORT
        filters = condition.filter

        if not isinstance(filters, list):
            filters = [filters]

        for expr in filters:

            if isinstance(expr, str):
                symbol = expr.split(" ")[0]
                right_expr = " ".join(expr.split(" ")[1:])
                value = float(eval(right_expr))

                arr = col

                # map operator
                if symbol == ">":
                    mask = pc.and_(mask, pc.greater(arr, value))
                elif symbol == "<":
                    mask = pc.and_(mask, pc.less(arr, value))
                elif symbol == ">=":
                    mask = pc.and_(mask, pc.greater_equal(arr, value))
                elif symbol == "<=":
                    mask = pc.and_(mask, pc.less_equal(arr, value))
                elif symbol == "==":
                    mask = pc.and_(mask, pc.equal(arr, value))
                else:
                    raise ValueError(f"Unknown operator: {symbol}")

    return table.filter(mask)


def apply_filter_df(d, conditions, config):
    """
    Apply filtering conditions to a Pandas DataFrame.

    Parameters:
    -----------
    d : pd.DataFrame
        The DataFrame to be filtered.
    conditions : dict
        Dictionary specifying filtering conditions for each column.
    config : DictConfig
        Configuration object containing predefined filter expressions.

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame based on the specified conditions.

    Raises:
    -------
    Exception
        If a specified feature in `conditions` is not present in the DataFrame.
    """
    if isinstance(d, pd.DataFrame):
        mask = pd.Series(True, index=d.index)
        for key, condition in conditions.items():
            #condition = OmegaConf.select(config, key) #TODO review
            if key not in d.columns:
                raise Exception(f'Error, feature "{key}" is not present in the halo dataframe with columns:'
                                f' {d.columns}')
            if hasattr(condition, 'filter'):
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
            else:
                if isinstance(condition, ListConfig):
                    for expr in condition:
                        if isinstance(expr, str):
                            symbol = expr.split(' ')[0]
                            right_expr = ' '.join(expr.split(' ')[1:])
                            value = float(eval(right_expr))
                            mask &= operators[symbol](d[key], value)

        return d[mask]


@pipeline_step
def apply_filters(d: Any, conditions: dict, config: DictConfig):
    """
    Apply filtering conditions to data.

    Parameters:
    -----------
    d : Any
        Data to be filtered (dictionary or DataFrame).
    conditions : dict
        Dictionary of filtering conditions.
    config : DictConfig
        Configuration object defining filter rules.

    Returns:
    --------
    Any
        Filtered data, either as a DataFrame or dictionary of DataFrames.
    """
    if isinstance(d, pd.DataFrame):
        return apply_filter_df(d, conditions, config)
    elif isinstance(d, pa.Table):
        return apply_filter_arrow(d, conditions, config)
    elif isinstance(d, dict):
        for key in d:
            d[key] = apply_filter_df(d[key], conditions.get(key, {}), config)
        return d


# TO TEST
@pipeline_step
def apply_unit_conversion(df: Any, fields: list, table: str, cfg: DictConfig, **kwargs):
    """
    Apply filtering conditions to data.

    Parameters:
    -----------
    d : Any
        Data to be filtered (dictionary or DataFrame).
    conditions : dict
        Dictionary of filtering conditions.
    config : DictConfig
        Configuration object defining filter rules.

    Returns:
    --------
    Any
        Filtered data, either as a DataFrame or dictionary of DataFrames.
    """

    for key in fields:
        if key in df:
            field_info = OmegaConf.select(cfg, f"fields.{table}.{key}")
            height = dict_depth(field_info)
            if height == 2:
                for key_ii in field_info:
                    field_info_ii = OmegaConf.select(cfg, f"fields.{table}.{key}.{key_ii}")
                    if field_info_ii.scaling is not None:
                        if cfg.name == 'magneticum':
                            key_ii = key_ii.ljust(4)
                        df[key][key_ii] = apply_conversion(df[key][key_ii], field_info_ii.scaling, cfg.a, cfg.H,
                                                           **kwargs)
            else:
                if field_info.scaling is not None:
                    df[key] = apply_conversion(df[key], field_info.scaling, cfg.a, cfg.H)
    return df


def run_pipeline(data, steps):
    """
    Executes a list of transformation steps sequentially.

    Parameters:
    -----------
    data : Any
        The input data to be processed.
    steps : list
        List of functions representing the pipeline steps.

    Returns:
    --------
    Any
        Transformed data after applying the pipeline.
    """
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

    def get_p_halo(self, *args, **kwargs):
        """
        Abstract method to fetch particle data of all subhalos within a halo.
        """
        pass

    def _get_fields(self):
        pass

    def __get_attr_recursive(self, config, attr_name, parent_key="", branch={}):
        """
        Recursively extract a specified attribute from a nested configuration dictionary.

        Parameters:
        -----------
        config : dict
            Dictionary containing the configuration structure.
        attr_name : str
            Attribute name to extract from the nested dictionary.
        parent_key : str, optional
            Prefix used to maintain full key paths (default: "").
        branch : dict, optional
            Dictionary to store extracted key-value pairs (default: {}).

        Returns:
        --------
        dict
            Dictionary mapping full key paths to their corresponding attribute values.
        """
        for key, value in config.items():
            full_key = f"{parent_key}{key}" if parent_key else key  # Construct the full key
            if dict_depth(value) > 1:
                self.__get_attr_recursive(value, attr_name, parent_key=f"{full_key}.", branch=branch)
            else:
                branch[full_key] = value.get(attr_name)
        return branch

    def _get_attr(self, config, attr_name, parent_key=""):
        """
        Extract a specified attribute from a nested dictionary.

        Parameters:
        -----------
        config : dict
            Dictionary containing the configuration structure.
        attr_name : str
            Attribute name to extract from the dictionary.
        parent_key : str, optional
            Prefix used to maintain full key paths (default: "").

        Returns:
        --------
        dict
            Dictionary mapping full key paths to their corresponding attribute values.
        """
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
        """
        Recursively retrieve all keys up to the first level of fields in a nested dictionary.

        Parameters:
        -----------
        config : dict
            Dictionary containing the configuration structure.
        parent_key : str, optional
            Prefix used to maintain full key paths (default: "").

        Returns:
        --------
        list
            List of keys found in the configuration up to the first level.
        """
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
        """
        Recursively retrieve all keys from a nested dictionary.

        Parameters:
        -----------
        config : dict
            Dictionary containing the configuration structure.
        parent_key : str, optional
            Prefix used to maintain full key paths (default: "").

        Returns:
        --------
        list
            List of all keys found in the nested configuration.
        """
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

    def __find_keys_with_needs(self) -> dict:
        result = {}
        for k, v in self.catalog.fields.items():
            if isinstance(v, DictConfig):
                for k_ii, v_ii in v.items():
                    if 'needs' in v_ii:
                        if k not in result:
                            result[k] = {}
                        result[k][k_ii] = v_ii['needs']
        return result

    def __build_config(self, **kwargs):
        """
        Build and initialize the configuration for halo, subhalo, and particle fields.

        This method extracts relevant field names, creates key-value mappings,
        and sets instance attributes based on provided keyword arguments.

        Parameters:
        -----------
        **kwargs : dict
            Additional attributes to set dynamically on the instance.

        Raises:
        -------
        ValueError
            If the catalog does not contain a 'fields' attribute.
        """
        self.fields = self._get_fields()

        fields = getattr(self.catalog, "fields", None)
        if not fields:
            raise ValueError("The catalog must contain a 'fields' attribute.")

        catalog = self.catalog.copy()
        self.halo_fields = self._get_nested_keys(catalog.fields.halo)
        self.subhalo_fields = self._get_nested_keys(catalog.fields.subhalo)
        particle_fields = self._get_nested_keys(catalog.fields.particles)
        self.particle_fields = set([p.split('.')[0] for p in particle_fields])

        kv_halo = self._get_attr(catalog.fields.halo, attr_name='name')
        kv_subhalo = self._get_attr(catalog.fields.subhalo, attr_name='name')
        kv_particles = self._get_attr(catalog.fields.particles, attr_name='name')

        self.column_map = kv_halo | kv_subhalo | kv_particles

        # Identify functions to be computed
        public_methods = list(FeatureComputer._registry.keys())
        # TODO define whether is subhalo level, halo or particles
        logger.info(public_methods)

        # Map fuction - variables
        self.function_needs = self.__find_keys_with_needs()

        # Check functions with need are implemented
        for key in self.function_needs.keys():
            for key_ii in self.function_needs[key]:
                if not key_ii in public_methods:
                    raise NotImplementedError(f'Function {key} referenced in the config, it is not implemented.')

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
        elif catalog.name == "magneticum":
            return MagneticumFetcher(catalog, **kwargs)
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
        logger.info(f"IllustrisFetcher initialized with config: {catalog.path}")

    def _get_fields(self):
        """
        Retrieve all available halo and subhalo fields from the simulation data.

        This method loads the halo and subhalo field names from the Illustris group catalog
        and returns a combined list of available fields.

        Returns:
        --------
        list
            A list containing all field names from the halo and subhalo datasets.

        Notes:
        ------
        - The method retrieves data from the `output` directory of the catalog.
        - Uses `il.groupcat.loadHalos` and `il.groupcat.loadSubhalos` to extract field names.
        - Assumes that `self.catalog.path` points to the dataset's directory and that `self.catalog.snapnum`
          specifies the snapshot number.

        """
        values = OmegaConf.select(self.catalog, "fields")
        fields = []
        for key in values:
            if key in ['halo', 'subhalo']:
                fields.extend(list(values[key].keys()))
        return fields

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """
        Fetches the halo data.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
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

        #logger.info(f"Fetching particle data for subhalo: {subhalo_id}")
        return self._get_p_subhalo(subhalo_id)

    def get_p_halo(self, *args, **kwargs) -> SubhaloParticles:
        halo = kwargs.get('halo', None)
        if halo is None:
            logger.error("halo must be provided in kwargs")
            raise ValueError("halo must be provided in kwargs")

        #logger.info(f"Fetching particle data for halo: {halo.halo_id}")
        return self._get_p_halo(halo)

    def __get_children(self, halo_row, subhalo_df):
        halo_id = halo_row['halo_id']
        children = subhalo_df[subhalo_df['id_parents'] == halo_id]
        children_ids = list(children['subhalo_id'])
        return children_ids

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
                lambda halos: apply_unit_conversion(halos, self.halo_fields, table, self.catalog),
                lambda halos: apply_filters(halos, halo_conditions, self.catalog.fields.halo)
            ]
            # Get subhalos ids
            subhalo_context = self._get_subhalos()

            processed_halos = run_pipeline(halos, steps)
            processed_halos['subhalo_ids'] = processed_halos.apply(lambda halo: self.__get_children(halo,
                                                                                                    subhalo_context.data),
                                                                   axis=1)
            self.column_map['subhalo_ids'] = 'subhalo_ids'
            halo_context = HaloContext(processed_halos, self.column_map)
            logger.debug(f"Halo data fetched successfully: {halo_context.data['halo_id'].nunique()} halos.")
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

        group_attr = ''
        if 'illustris300' in self.catalog.path:
            group_attr = 'SubhaloGrNr'
        elif 'tng_cluster' in self.catalog.path:
            group_attr = 'SubhaloGrNr'
        elif 'millenium' in self.catalog.path:
            group_attr = 'SubhaloGroupNr'
        else:
            raise NotImplementedError('Please check the path used in your configuration.')

        if group_attr not in self.halo_fields:
            self.subhalo_fields.append(group_attr)
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.subhalo[group_attr] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "id_parents"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map[group_attr] = 'id_parents'

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
                lambda subhalos: apply_unit_conversion(subhalos, self.subhalo_fields, table, self.catalog),
                lambda subhalos: apply_filters(subhalos, subhalo_conditions, self.catalog.fields.subhalo)
            ]

            processed_subhalos = run_pipeline(subhalos, steps)
            subhalo_context = SubhaloContext(processed_subhalos, self.column_map)
            logger.debug(f"Subhalo data fetched successfully: {len(subhalo_context.data)} subhalos.")
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
        #logger.debug(f"Fetching particle data for subhalo: {subhalo_id}")
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

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
                count = particles.get('count', None)
                if count is None or count > 0:
                    data[part_type] = particles

                    for key in particles:
                        if isinstance(key, np.ndarray):
                            num_particles = particles[key].shape[0]
                            #logger.debug(f"Particle data fetched successfully, particles retrieved: {num_particles}")
                            break
                else:
                    logger.debug(f'Subhalo {subhalo_id} does not have {part_type} particles.')
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
            lambda particles: apply_unit_conversion(particles, self.particle_fields, table, self.catalog),
            lambda particles: apply_filters(particles, particle_conditions, self.catalog.fields.particles)
        ]

        processed_particles = run_pipeline(data, steps)
        #logger.debug("Particle data fetched successfully.")
        particle_data = SubhaloParticles(processed_particles, self.catalog.fields.particles)
        return particle_data

    def _get_p_halo(self, halo: Halo) -> SubhaloParticles:
        logger.debug(f"Fetching particle data for halo: {halo.halo_id}")
        path = self.catalog.path + '/output'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        subhalo_ids = halo.subhalo_ids
        particles_halo = None
        for subhalo_id in subhalo_ids:
            particles = self._get_p_subhalo(subhalo_id)
            if particles_halo is None:
                particles_halo = particles
            else:
                particles_halo += particles
        logger.debug(f"Particle data for halo {halo.halo_id}: {particles_halo}")
        return particles_halo


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
        logger.info(f"EagleFetcher initialized with config: {catalog.path}")

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
        group_number = kwargs.get('halo_id', None)

        if subgroup_id is None:
            logger.error("subhalo_id must be provided in kwargs")
            raise ValueError("subgroup_id must be provided in kwargs")
        if group_number is None:
            logger.error("halo_id must be provided in kwargs")
            raise ValueError("group_number must be provided in kwargs")

        #logger.info(f"Fetching particle data for subhalo: {subgroup_id}, Group_number: {group_number}")
        return self._get_p_subhalo(subgroup_id, group_number)

    def get_p_halo(self, *args, **kwargs) -> SubhaloParticles:
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
        halo = kwargs.get('halo', None)
        if halo is None:
            logger.error("halo must be provided in kwargs")
            raise ValueError("halo must be provided in kwargs")

        group_number_list, subhalo_id_list = zip(*halo.subhalo_ids)
        return self._get_p_halo(group_number_list, subhalo_id_list)

    def _get_fields(self):
        pass

    def _build_query(self, table):
        """
        Construct an SQL query to retrieve halo or subhalo properties from the simulation database.

        This method generates an SQL query based on the specified table type (`halo` or `subhalo`)
        and incorporates fields defined in the object's configuration.

        Parameters:
        -----------
        table : str
            The name of the table to query, either `'halo'` or `'subhalo'`.

        Returns:
        --------
        str
            The complete SQL query string.

        Raises:
        -------
        ValueError
            If the provided `table` argument is not `'halo'` or `'subhalo'`.

        Notes:
        ------
        - For `table='halo'`, the query retrieves:
            - `halo_id`: Unique identifier for each halo (`FOF.GroupID`).
            - Additional halo properties dynamically included from `self.halo_fields`.
            - `group_number`: Links to the FoF group.
            - `subhalo_id`: Identifier of the most massive subhalo.
            - `subhalo_mass`: Mass of the most massive subhalo.

        - For `table='subhalo'`, the query retrieves:
            - `halo_id`: Identifier of the parent halo.
            - `subhalo_id`: Identifier of the subhalo.
            - `group_number`: Links to the FoF group.
            - Additional subhalo properties dynamically included from `self.subhalo_fields`.
            - `m200`: The mass within `R200` of the halo.

        - Queries filter results based on the snapshot number (`SnapNum`) to ensure consistency
          with a specific redshift.
        - The results for subhalos are ordered by `SubGroupNumber` in ascending order.
        """
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
                                        )
                                ) AS SH ON FOF.GroupID = SH.GroupID
                            WHERE 
                                FOF.SnapNum = {self.catalog.snapnum} -- Snapshot for redshift ~0.2
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
            halos = pd.DataFrame(array)
        else:
            logger.info(f"Reading from files ...")
            halos = pd.read_csv(os.path.join(path, 'fof.csv'))

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

        if len(halos) > 0:
            # Use original names for later fitlering
            halos = halos.sort_values(by='GroupID')
            halos['GroupID'] = halos['GroupID'].astype(int)

            table = 'halo'
            halo_conditions = {col: data for col, data in self.catalog.fields.halo.items() if 'filter' in data}

            steps = [
                lambda halos: apply_unit_conversion(halos, self.halo_fields, table, self.catalog),
                lambda halos: apply_filters(halos, halo_conditions, self.catalog.fields.halo)
            ]

            # Get subhalos ids
            processed_halos = run_pipeline(halos, steps)
            halo_context = HaloContext(processed_halos, column_map=self.column_map)
            logger.debug(f"Halo data fetched successfully: {len(halo_context.data)} halos.")
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
            subhalos = pd.DataFrame(array)
        else:
            logger.info(f"Reading from files ...")
            subhalos = pd.read_csv(os.path.join(path, 'subhalo.csv'))

        # Remove useless columns
        subhalos = subhalos.drop(columns=['Group_M_Crit200'])
        subhalos = subhalos.rename(columns={'GroupID': 'id_parents'})

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

        if len(subhalos) > 0:
            subhalos = subhalos.sort_values(by='SubGroupNumber')

            table = 'subhalo'
            subhalo_conditions = {col: data for col, data in self.catalog.fields.subhalo.items() if 'filter' in data}
            steps = [
                lambda subhalos: apply_unit_conversion(subhalos, self.subhalo_fields, table, self.catalog),
                lambda subhalos: apply_filters(subhalos, subhalo_conditions, self.catalog.fields.subhalo)
            ]
            processed_subhalos = run_pipeline(subhalos, steps)
            processed_subhalos = processed_subhalos.sort_values(by=['SubGroupNumber', 'GroupNumber'])
            subhalo_context = SubhaloContext(processed_subhalos, self.column_map)
            logger.debug(f"Subhalo data fetched successfully: {len(subhalo_context.data)} subhalos.")
        else:
            logger.warning("No subhalo data found in Eagle database.")
            subhalo_context = SubhaloContext(pd.DataFrame(data=[]), self.column_map)
        return subhalo_context

    @staticmethod
    def __partTypeNum(partType):
        """
        Convert a particle type name into its corresponding numerical identifier.

        This method maps common particle type names (such as 'gas', 'dm', 'stars', etc.)
        to their respective numerical codes used in the simulation data.

        Parameters:
        -----------
        partType : str or int
            The particle type, either as a numeric identifier (0-5) or a string
            representing the type (e.g., 'gas', 'stars', 'bh').

        Returns:
        --------
        int
            The numerical particle type corresponding to the given name.

        Raises:
        -------
        Exception
            If the provided particle type name is not recognized.

        Notes:
        ------
        - The mapping is as follows:
            - `0` : Gas (aliases: 'gas', 'cells')
            - `1` : Dark Matter (aliases: 'dm', 'darkmatter')
            - `2` : Low-resolution Dark Matter (alias: 'dmlowres', only in zoom simulations)
            - `3` : Tracers (aliases: 'tracer', 'tracers', 'tracermc', 'trmc')
            - `4` : Stars (aliases: 'star', 'stars', 'stellar') and Wind (alias: 'wind')
            - `5` : Black Holes (aliases: 'bh', 'bhs', 'blackhole', 'blackholes')
        - For stellar particles, those with `GFM_StellarFormationTime > 0` are considered stars,
          while those with `GFM_StellarFormationTime < 0` are winds.
        - If the input is already a digit, it is directly converted to an integer.

        """
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
        group_number: int
            The index of the halo where the subhalo belongs from

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data such as mass, coordinates, and velocities.
        """
        path = self.catalog.path
        part_types = self.particle_fields
        particles = {}

        # Read particles data and change format
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
                data_part_type[field] = column
                # Dealing with numpy arrays of > 1 dim for dataframe conversion afterward
                if column.ndim > 1:
                    converted = to_list(data_part_type, field)
                    del data_part_type[field]
                    data_part_type[field] = converted

            # Filter by indexing
            tmp = pd.DataFrame(data_part_type)
            # Create index with these two columns for speed
            tmp = tmp[(tmp["GroupNumber"] == group_number) & (tmp["SubGroupNumber"] == subgroup_id)]
            if len(tmp) == 0:
                logger.error(f'Error, there is not particle data for GroupNumber={group_number} and '
                             f'SubGroupNumber={subgroup_id}')
                raise ValueError(f'Error, there is not particle data for GroupNumber={group_number} and'
                                 f' SubGroupNumber={subgroup_id}')
            tmp = tmp.drop(columns=['GroupNumber', 'SubGroupNumber'])
            particles[part_type] = tmp.to_dict(orient='list')

        else:
            pass
            #logger.debug(f"Particle data fetched successfully, {len(tmp)} particles retrieved.")

        table = 'particles'

        particle_conditions = {}
        for col, type_config in self.catalog.fields.particles.items():
            for t in type_config:
                if 'filter' in type_config[t]:
                    if col not in particle_conditions:
                        particle_conditions[col] = {}
                    particle_conditions[col][t] = type_config[t]

        steps = [
            lambda df: convert_to_dataframe(particles),
            lambda df: apply_unit_conversion(df, self.particle_fields, table, self.catalog,
                                                    g=self.catalog.g, g_2_solar=self.catalog.g_2_solar),
            lambda particles: apply_filters(particles, particle_conditions, self.catalog.fields.particles)
        ]
        processed_particles = run_pipeline(particles, steps)
        subhalo_particles = SubhaloParticles(processed_particles, self.catalog.fields.particles)
        return subhalo_particles

    # TODO need a list of subgroupids
    def _get_p_halo(self, group_number_list: list, subhalo_id_list: list):
        """
        Fetches particle data for a specific subhalo from the Eagle simulation.

        Parameters:
        -----------
        subgroup_id : int
            The index of the subhalo from which to retrieve particle data.
        group_number: int
            The index of the halo where the subhalo belongs from

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data such as mass, coordinates, and velocities.
        """
        path = self.catalog.path
        part_types = self.particle_fields
        particles = {}

        # Read particles data and change format
        for part_type in part_types:
            logger.debug(f'Particle type: {part_type}')
            data_part_type = {}
            itype = self.__partTypeNum(part_type)
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.particles[part_type]["GroupNumber"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "group_number"
            }
            OmegaConf.set_struct(self.catalog, True)

            fields = list(self.catalog.fields.particles[part_type].keys())
            fields.append('SubGroupNumber')
            logger.debug(fields)
            if not os.path.exists(path):
                logger.error(f"The path {path} does not exist.")
                raise Exception(f'The path {path} does not exist.')

            for field in fields:
                logger.debug(f'Reading field: {field}')
                column = read_dataset(itype, field, path)
                data_part_type[field] = column
                # Dealing with numpy arrays of > 1 dim for dataframe conversion afterward
                if column.ndim > 1:
                    converted = to_list(data_part_type, field)
                    del data_part_type[field]
                    data_part_type[field] = converted

            # Filter by indexing
            tmp = pd.DataFrame(data_part_type)
            logger.debug(len(tmp))
            # Create index with these two columns for speed
            tmp = tmp[(tmp["GroupNumber"].isin(group_number_list)) & (tmp['SubGroupNumber'].isin(subhalo_id_list))]
            if len(tmp) == 0:
                logger.error(f'Error, there is not particle data for GroupNumber={group_number_list} '
                             f'and subhalo ids={subhalo_id_list}')
                raise ValueError(f'Error, there is not particle data for GroupNumber={group_number_list} '
                                 f'and subhalo ids={subhalo_id_list}')
            tmp = tmp.drop(columns=['GroupNumber', 'SubGroupNumber'])
            particles[part_type] = tmp.to_dict(orient='list')
            logger.debug(len(tmp))
        else:
            pass
            #logger.debug(f"Particle data fetched successfully, {len(tmp)} particles retrieved.")

        table = 'particles'

        particle_conditions = {}
        for col, type_config in self.catalog.fields.particles.items():
            for t in type_config:
                if 'filter' in type_config[t]:
                    if col not in particle_conditions:
                        particle_conditions[col] = {}
                    particle_conditions[col][t] = type_config[t]

        steps = [
            lambda df: convert_to_dataframe(particles),
            lambda df: apply_unit_conversion(df, self.particle_fields, table, self.catalog,
                                                    g=self.catalog.g, g_2_solar=self.catalog.g_2_solar),
            lambda particles: apply_filters(particles, particle_conditions, self.catalog.fields.particles)
        ]
        logger.debug("Starting pipeline execution.")
        processed_particles = run_pipeline(particles, steps)
        logger.debug("Pipeline execution finished.")
        halo_particles = SubhaloParticles(processed_particles, self.catalog.fields.particles)
        logger.debug(f'Particles found: {halo_particles}')
        return halo_particles


class FlamingoFetcher(DataFetcher):
    def __init__(self, catalog: DictConfig, **kwargs):
        super().__init__(catalog, **kwargs)
        logger.info(f"FlamingoFetcher initialized with config: {catalog.path}")
        self.fof_subhalos = None
        self.particles = None
        self.load_files()

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """
        Fetches the halo data.

        Returns:
        --------
        HaloContext
            A context object containing halo data.
        """
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

        #logger.info(f"Fetching particle data for subhalo: {subhalo_id}")
        return self._get_p_subhalo(subhalo_id)

    def get_p_halo(self, *args, **kwargs) -> SubhaloParticles:
        """
        Fetches particle data for a given halo.

        Parameters:
        -----------
        halo_id : int
            ID of the halo for which particle data is needed.

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data.
        """
        halo = kwargs.get('halo', None)
        if halo is None:
            logger.error("halo must be provided in kwargs")
            raise ValueError("halo must be provided in kwargs")

        return self._get_p_halo(halo)

    def _validate_and_process_fields(self, soap: SWIFTDataset, fields, super_key, mask=None, n_rows=None):
        """
        Validate and extract field values from a SWIFT dataset, applying unit conversions and optional filtering.

        This method iterates over the requested `fields`, extracts their values from the `soap` dataset,
        applies unit conversions, and optionally filters the data based on `mask` or `n_rows` constraints.

        Parameters:
        -----------
        soap : SWIFTDataset
            The dataset containing simulation data.
        fields : list of str
            The list of field names to extract, following a hierarchical format (e.g., `"subhalo.mass"`).
        super_key : str
            The base key prefix used to locate field metadata in the catalog.
        mask : dict or ndarray, optional
            - If a dictionary, it applies a mask per field key.
            - If an ndarray, it is applied directly as a boolean filter.
        n_rows : int, optional
            Limits the number of rows to extract from each field.

        Returns:
        --------
        dict
            A dictionary where keys are field names and values are extracted, filtered, and unit-converted arrays.

        Notes:
        ------
        - Extracted values are converted to physical units when applicable.
        - If a field is multi-dimensional, it is processed into batches.
        - Errors during extraction/logging are caught, and missing fields are skipped.
        """
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

                if n_rows is not None:
                    data[field_name] = data[field_name][:n_rows]

                if mask is not None:
                    if isinstance(mask, dict):
                        mask_key = field_name.split('.')[0]
                        is_full_false = np.sum(mask[mask_key])
                        tmp = data[field_name][mask[mask_key]]
                        data[field_name] = tmp
                    else:
                        data[field_name] = data[field_name][mask]
                if isinstance(data[field_name], np.ndarray) and data[field_name].ndim > 1:
                    data[field_name] = to_list_in_batches(data, field_name)
            except Exception as e:
                logger.error(e)
                continue
        return data

    def __filter_non_primary(self, halos):
        """
        Filter halos to retain only those that are the primary FoF halos.

        This function removes non-primary subhalos by checking the `soap.host_halo_index` field.

        Parameters:
        -----------
        halos : pandas.DataFrame or similar structured array
            The dataset containing halo properties.

        Returns:
        --------
        pandas.DataFrame
            A filtered version of the input `halos` where only primary halos (FoFs) are retained.

        Notes:
        ------
        - This method removes subhalos by keeping only entries where `soap.host_halo_index == -1`.
        """
        # Check just positions allocating fofs and primary subhalos
        halos = halos[halos['soap.host_halo_index'] == -1]
        return halos

    def __filter_primary(self, halos, halos_ids):
        """
        Filter halos to retain only those that are the NON primary FoF halos.

        This function removes non-primary subhalos by checking the `soap.host_halo_index` field.

        Parameters:
        -----------
        halos : pandas.DataFrame or similar structured array
            The dataset containing halo properties.

        Returns:
        --------
        pandas.DataFrame
            A filtered version of the input `halos` where only primary halos (FoFs) are retained.

        Notes:
        ------
        - This method removes subhalos by keeping only entries where `soap.host_halo_index != -1`.
        """
        # Check just positions allocating fofs and primary subhalos
        halos = halos[((halos['soap.host_halo_index'] == -1) & (halos.index.isin(halos_ids))) |
                      (halos['soap.host_halo_index'].isin(halos_ids))]
        return halos

    def load_files(self):
        """
        Load and validate the necessary dataset files for analysis.

        This method loads two HDF5 files:
        - `flamingo_<redshift>.hdf5`: Contains subhalo and particle data.
        - `halo_properties_<redshift>.hdf5`: Contains halo-level properties.

        If either file is missing, an error is logged, and an exception is raised.

        Raises:
        -------
        Exception
            If the required dataset files do not exist at the expected paths.

        Notes:
        ------
        - The file paths are determined based on the snapshot number (`snapnum`).
        - The loaded datasets are stored in `self.fof_subhalos` (halo properties) and `self.particles` (subhalos/particles).
        """
        redshift = str(self.catalog.snapnum).zfill(4)

        path = self.catalog.path + f'/SOAP-HBT/flamingo_{redshift}.hdf5'
        fof_path = self.catalog.path + f'/SOAP-HBT/halo_properties_{redshift}.hdf5'

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        if not os.path.exists(fof_path):
            logger.error(f"The path {fof_path} does not exist.")
            raise Exception(f'The path {fof_path} does not exist.')

        self.fof_subhalos = sw.load(fof_path)
        self.particles = sw.load(path)

    def _get_halos(self) -> HaloContext:
        """
        Fetches halo data from the Flamingo simulation.

        Returns:
        --------
        HaloContext
            A context object containing halo data such as mass, radius, position, and subhalo index.
        """
        logger.debug("Fetching halos from Flamingo...")

        self.column_map['input_halos.halo_centre'] = 'gpos'
        self.column_map['soap.host_halo_index'] = 'is_primary'

        table = 'halo'

        # Filter primary subhalos
        if 'soap.host_halo_index' not in self.halo_fields:
            OmegaConf.set_struct(self.catalog, False)

            if 'soap' not in self.catalog.fields.halo:
                self.catalog.fields.halo["soap"] = {}

            self.catalog.fields.halo["soap"]["host_halo_index"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "parent_id_satellite"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.halo_fields.append('soap.host_halo_index')

        if isinstance(self.fof_subhalos, SWIFTDataset):
            halos = self._validate_and_process_fields(self.fof_subhalos, self.halo_fields, super_key='halo')
            halo_conditions = self._get_attr(self.catalog.fields.halo, attr_name='filter', parent_key="")
            filtered_conditions = {k: v for k, v in halo_conditions.items() if v is not None}
            steps = [
                lambda halos: assign_id(halos, 'halo_id'),
                lambda halos: assign_id(halos, 'fsub'),
                lambda halos: convert_to_dataframe(halos),
                lambda halos: self.__filter_non_primary(halos),
                lambda halos: apply_unit_conversion(halos, self.halo_fields, table, self.catalog),
                lambda halos: apply_filters(halos, filtered_conditions, self.catalog.fields.halo),
            ]
            processed_halos = run_pipeline(halos, steps)
            logger.debug("Halo data fetched successfully.")
            halo_context = HaloContext(processed_halos, self.column_map)
            logger.debug(f"Halo data fetched successfully: {len(halo_context.data)} halos.")
            return halo_context

        else:
            logger.warning("No halos found.")
            return HaloContext(pd.DataFrame(data=[]), self.column_map)

    @staticmethod
    def __reindexing_satellites(df: pd.DataFrame):
        """
        Reindex satellite galaxies by updating their parent halo IDs.

        This method modifies the `id_parents` column for satellite halos, ensuring they reference their
        corresponding host halo. It then removes the `soap.host_halo_index` column.

        Parameters:
        -----------
        df : pd.DataFrame
            A DataFrame containing halo and satellite properties. It must include the `soap.host_halo_index` column.

        Returns:
        --------
        pd.DataFrame
            The modified DataFrame with:
            - The `id_parents` column updated for satellites.
            - The `soap.host_halo_index` column removed.

        Notes:
        ------
        - A halo is identified as a satellite if its `soap.host_halo_index` is not `-1`.
        - The `id_parents` column is assigned the `soap.host_halo_index` value for satellites.
        """
        # Update father of satellites
        # satellites_mask = self.fof_subhalos.soap.host_halo_index == fof_id
        df.loc[df['soap.host_halo_index'] != -1, 'id_parents'] = df.loc[df['soap.host_halo_index'] != -1, 'soap.host_halo_index']
        df = df.drop(columns=['soap.host_halo_index'])
        return df

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetches subhalo data from the Flamingo simulation.

        Returns:
        --------
        SubhaloContext
            A context object containing subhalo data.
        """
        logger.debug("Fetching subhalos from Flamingo...")
        self.column_map['input_halos.halo_centre'] = 'spos'

        if 'soap.host_halo_index' not in self.subhalo_fields:
            OmegaConf.set_struct(self.catalog, False)
            if 'soap' not in self.catalog.fields.subhalo:
                self.catalog.fields.subhalo["soap"] = {}

            self.catalog.fields.subhalo["soap"]["host_halo_index"] = {
                "input": None,
                "output": None,
                "scaling": None,
                "name": "parent_id_satellite"
            }
            self.subhalo_fields.append('soap.host_halo_index')  # to filter satellites
            self.column_map['soap.host_halo_index'] = 'parent_id_satellite'
            OmegaConf.set_struct(self.catalog, True)

        if isinstance(self.fof_subhalos, SWIFTDataset):

            if 'soap.host_halo_index' not in self.halo_fields:
                OmegaConf.set_struct(self.catalog, False)

                if 'soap' not in self.catalog.fields.halo:
                    self.catalog.fields.halo["soap"] = {}

                self.catalog.fields.halo["soap"]["host_halo_index"] = {
                    "input": None,
                    "output": None,
                    "scaling": None,
                    "name": "parent_id_satellite"
                }
                OmegaConf.set_struct(self.catalog, True)
                self.halo_fields.append('soap.host_halo_index')

            # Add parents as subhalos => check fof filtering
            table = "halo"
            halos = self._validate_and_process_fields(self.fof_subhalos, self.halo_fields, super_key='halo')
            halo_conditions = self._get_attr(self.catalog.fields.halo, attr_name='filter', parent_key="")
            filtered_conditions = {k: v for k, v in halo_conditions.items() if v is not None}
            steps = [
                lambda halos: assign_id(halos, 'halo_id'),
                lambda halos: assign_id(halos, 'fsub'),
                lambda halos: convert_to_dataframe(halos),
                lambda halos: self.__filter_non_primary(halos),
                lambda halos: apply_unit_conversion(halos, self.halo_fields, table, self.catalog),
                lambda halos: apply_filters(halos, filtered_conditions, self.catalog.fields.halo),
            ]
            processed_halos = run_pipeline(halos, steps)
            logger.debug("Halo data fetched successfully.")
            halo_context = HaloContext(processed_halos, self.column_map)
            halos_ids = halo_context.halo_id

            table = 'subhalo'
            subhalos = self._validate_and_process_fields(self.fof_subhalos, self.subhalo_fields, super_key='subhalo')
            subhalo_conditions = self._get_attr(self.catalog.fields.subhalo, attr_name='filter', parent_key="")
            filtered_conditions = {k: v for k, v in subhalo_conditions.items() if v is not None}
            steps = [
                lambda subhalos: assign_id(subhalos, 'id_parents'),
                lambda subhalos: assign_id(subhalos, 'subhalo_id'),
                lambda subhalos: convert_to_dataframe(subhalos),
                lambda subhalos: self.__filter_primary(subhalos, halos_ids),
                lambda subhalos: self.__reindexing_satellites(subhalos),
                lambda subhalos: apply_unit_conversion(subhalos, self.subhalo_fields, table, self.catalog),
                lambda subhalos: apply_filters(subhalos, filtered_conditions, self.catalog.fields.subhalo),
            ]
            processed_subhalos = run_pipeline(subhalos, steps)
            subhalo_context = SubhaloContext(processed_subhalos, self.column_map)

            logger.debug(f"Subhalo data fetched successfully: {len(subhalo_context.data)} subhalos.")
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
            The index of the subhalo from which to retrieve particle data, corresponds to
            fof_subhalos.input_halos.halo_catalogue_index.

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data such as mass, coordinates, and velocities.
        """
        #logger.debug(f"Fetching particle data for subhalo: {subhalo_id}")

        # Get the position for the subhalo
        # TODO
        subhalo_position = np.where(self.fof_subhalos.input_halos.halo_catalogue_index == subhalo_id)[0]

        if len(subhalo_position) > 0:
            host_halo_index = self.fof_subhalos.soap.host_halo_index[subhalo_position[0]]
        else:
            raise ValueError(f'Subhalo: {subhalo_id} is not a primary subhalo')

        if host_halo_index != -1:
            # Not a primary index
            raise ValueError(f'Subhalo: {subhalo_id} is not a primary subhalo')

        # Add halo_catalogue_index for all the particle types
        for part_type in self.particle_fields:
            OmegaConf.set_struct(self.catalog, False)
            if 'halo_catalogue_index' not in self.catalog.fields.particles[part_type]:
                self.catalog.fields.particles[part_type]['halo_catalogue_index'] = {
                    "input": None,
                    "output": None,
                    "scaling": None,
                    "name": "parent_id"
                }
            OmegaConf.set_struct(self.catalog, True)

        part_types = self.particle_fields
        data = {}

        type_mask = {}
        # Build particles mask
        for part_type in part_types:
            particles_of_type = getattr(self.particles, part_type)
            # Mask to get particles belonging to the primary subhalo
            particles_mask_primary = particles_of_type.halo_catalogue_index == subhalo_id
            type_mask[part_type] = particles_mask_primary

            if np.all(particles_mask_primary == False):
                logger.warning(f'Subhalo {subhalo_id} has no particles associated.')

        particle_conditions = self._get_attr(self.catalog.fields.particles, attr_name='filter', parent_key="")
        filtered_conditions = {k: v for k, v in particle_conditions.items() if v is not None}
        particle_full_names = self._get_nested_keys(self.catalog.fields.particles)
        data = self._validate_and_process_fields(self.particles, particle_full_names, super_key='particles',
                                                 mask=type_mask)
        table = 'particles'
        steps = [
            lambda particles: convert_to_dataframe(data),
            lambda particles: apply_unit_conversion(particles, self.particle_fields, table, self.catalog),
            lambda particles: apply_filters(particles, filtered_conditions, self.catalog.fields.particles)
        ]

        processed_particles = run_pipeline(data, steps)
        logger.debug("Particles data fetched successfully.")

        # SubhaloParticles is expecting a dict with part_types - df
        d = {}
        columns = list(processed_particles.columns)
        for part_type in part_types:
            filtered_columns = [column for column in columns if part_type in column]
            renamed_columns = [column.replace(f'{part_type}.', '') for column in filtered_columns]
            rename_dict = dict(zip(filtered_columns, renamed_columns))
            tmp = processed_particles[filtered_columns].rename(mapper=rename_dict, axis='columns')
            d[part_type] = tmp

        particle_data = SubhaloParticles(d, self.catalog.fields.particles)
        return particle_data

    def _get_p_halo(self, halo: Halo) -> SubhaloParticles:
        """
        Fetches particle data for a specific halo.

        Parameters:
        -----------
        isubhalo : int
            The index of the subhalo from which to retrieve particle data, corresponds to
            fof_subhalos.input_halos.halo_catalogue_index.

        Returns:
        --------
        SubhaloParticles
            A context object containing particle data such as mass, coordinates, and velocities.
        """
        if halo.is_primary != -1:
            # Not a primary index
            raise ValueError(f'Subhalo: {halo.halo_id} is not a primary subhalo')

        # Add halo_catalogue_index for all the particle types
        for part_type in self.particle_fields:
            OmegaConf.set_struct(self.catalog, False)
            if 'halo_catalogue_index' not in self.catalog.fields.particles[part_type]:
                self.catalog.fields.particles[part_type]['halo_catalogue_index'] = {
                    "input": None,
                    "output": None,
                    "scaling": None,
                    "name": "parent_id"
                }
            OmegaConf.set_struct(self.catalog, True)

        part_types = self.particle_fields
        data = {}

        type_mask = {}
        # Build particles mask
        for part_type in part_types:
            particles_of_type = getattr(self.particles, part_type)
            # Mask to get particles belonging to the primary subhalo
            particles_mask_primary = particles_of_type.halo_catalogue_index == halo.halo_id
            is_full_false = np.sum(particles_mask_primary)

            # Get satellite subhalos
            satellites_mask = self.fof_subhalos.soap.host_halo_index == halo.halo_id

            # Loop over the halo_catalogue_index of the satellites to get all bound particles
            satellites_idx_array = self.fof_subhalos.input_halos.halo_catalogue_index[satellites_mask]
            particles_mask_satellites = np.isin(particles_of_type.halo_catalogue_index, satellites_idx_array)
            particles_mask = particles_mask_primary | particles_mask_satellites

            type_mask[part_type] = particles_mask

            if np.all(particles_mask_satellites == False):
                logger.warning(f'Halo {halo.halo_id} has no satellites.')

            if np.all(particles_mask_primary == False):
                logger.warning(f'Halo {halo.halo_id} has no particles.')

        particle_conditions = self._get_attr(self.catalog.fields.particles, attr_name='filter', parent_key="")
        filtered_conditions = {k: v for k, v in particle_conditions.items() if v is not None}
        particle_full_names = self._get_nested_keys(self.catalog.fields.particles)
        data = self._validate_and_process_fields(self.particles, particle_full_names, super_key='particles',
                                                 mask=type_mask)
        logger.debug(data)
        table = 'particles'
        steps = [
            lambda particles: convert_to_dataframe(data),
            lambda particles: apply_unit_conversion(particles, self.particle_fields, table, self.catalog),
            lambda particles: apply_filters(particles, filtered_conditions, self.catalog.fields.particles)
        ]

        processed_particles = run_pipeline(data, steps)
        logger.debug("Halo particles data fetched successfully.")
        logger.debug(processed_particles)

        # SubhaloParticles is expecting a dict with part_types - df
        d = {}
        columns = list(processed_particles.columns)
        for part_type in part_types:
            logger.debug(f'part type:{part_type}')
            filtered_columns = [column for column in columns if part_type in column]
            renamed_columns = [column.replace(f'{part_type}.', '') for column in filtered_columns]
            rename_dict = dict(zip(filtered_columns, renamed_columns))
            tmp = processed_particles[filtered_columns].rename(mapper=rename_dict, axis='columns')
            d[part_type] = tmp
            logger.debug(len(tmp))
            logger.debug(type(tmp))

        logger.debug(d)
        particle_data = SubhaloParticles(d, self.catalog.fields.particles)
        logger.debug(f'Stellar particle size: {len(particle_data.star_p)}')
        return particle_data


class MagneticumFetcher(DataFetcher):
    def __init__(self, catalog: DictConfig, **kwargs):
        super().__init__(catalog, **kwargs)
        logger.info(f"MagneticumFetcher initialized with config: {catalog.path}")

    # ── cache loading ──────────────────────────────────────────────────────────

    def _load_catalog_cache(self):
        """Load lightweight metadata only. No snapshot data."""
        path = self.catalog.path
        logger.debug(f"Catalog path: {path}")

        groups_dir = os.path.join(path, f"groups_{self.catalog.snapnum}")
        group_base = os.path.join(
            groups_dir,
            os.path.basename(groups_dir).replace("groups", "sub")
        )
        logger.debug(f"Group dir base: {group_base}")

        snapshot_dir = os.path.join(path, f"snapdir_{self.catalog.snapnum}")
        snap_base = os.path.join(
            snapshot_dir,
            os.path.basename(snapshot_dir).replace("snapdir", "snap")
        )
        logger.debug(f"Snapdir base: {snap_base}")

        self._group_base = group_base
        self._snap_base  = snap_base

        # Batch read halo metadata (block type 0)
        halo_catalog = g3read.read_new(
            group_base, ["GOFF", "GLEN", "FSUB", "NSUB"],
            -1, is_snap=False, multiple_files=True
        )

        if len(halo_catalog) == 0:
            logger.warning("Empty halo catalog — aborting cache load.")
            return {}

        self._goff = halo_catalog["GOFF"]
        self._glen = halo_catalog["GLEN"]
        self._fsub = halo_catalog["FSUB"]
        self._nsub = halo_catalog["NSUB"]

        # Subhalo offset/length (block type 1)
        self._soff = g3read.read_new(
            group_base, ["SOFF"], -1, is_snap=False, multiple_files=True
        )["SOFF"]
        self._slen = g3read.read_new(
            group_base, ["SLEN"], -1, is_snap=False, multiple_files=True
        )["SLEN"]

        # Halo PID file boundaries — lazy index, never loads full array
        logger.debug("Building _pid_file_boundaries...")
        self._pid_file_boundaries = self._build_pid_file_boundaries()

        # Subhalo PID file boundaries — lazy index replacing the old eager
        # full-array load (self._subhalos_pid) which OOM'd on large boxes
        logger.debug("Building _subhalo_pid_boundaries...")
        self._subhalo_pid_boundaries = self._build_subhalo_pid_file_boundaries()

        self._cache_loaded = True
        logger.debug("Catalog metadata cache loaded successfully.")

    def _load_particle_cache(self):
        """Pre-compute per-particle-type field lists and filter conditions."""
        self._snap_cache_config = {}
        for part_type, type_config in self.catalog.fields.particles.items():
            fields = list(type_config.keys())
            if "ID" not in fields:
                fields.append("ID")
            self._snap_cache_config[part_type] = {
                "fields_padded": [f.ljust(4) for f in fields],
                "ptype_int":     MAGNETICUM_PART_TYPES[part_type],
            }

        # FIX: removed dead first assignment that was immediately overwritten
        self._particle_conditions = {}

    # ── PID helpers (halos) ────────────────────────────────────────────────────

    def _build_pid_file_boundaries(self) -> list:
        """
        Build a lookup table of {file, start, end} for halo PID files (block 0).
        Reads only file content for PID block — not full snapshot data.
        """
        boundaries     = []
        current_offset = 0

        for group_file in g3read.yield_all_files(self._group_base):
            f           = g3read.GadgetFile(group_file, is_snap=False)
            pid_in_file = f.read("PID ", 2)

            if pid_in_file is None or len(pid_in_file) == 0:
                continue

            n = len(pid_in_file)
            boundaries.append({
                "file":  group_file,
                "start": current_offset,
                "end":   current_offset + n,
            })
            current_offset += n
            del pid_in_file

        logger.debug(
            f"PID boundaries: {len(boundaries)} files, total PIDs: {current_offset}"
        )
        return boundaries

    def _get_pid_for_entity(self, offset: int, length: int) -> np.ndarray:
        """
        Read halo PIDs for a specific halo on demand.
        Only reads the file(s) that contain the required offset range.
        """
        if length == 0:
            return np.array([], dtype=np.int64)

        entity_start = offset
        entity_end   = offset + length
        pids         = []

        for boundary in self._pid_file_boundaries:
            file_start = boundary["start"]
            file_end   = boundary["end"]

            overlap_start = max(entity_start, file_start)
            overlap_end   = min(entity_end,   file_end)

            if overlap_start >= overlap_end:
                continue

            f           = g3read.GadgetFile(boundary["file"], is_snap=False)
            pid_in_file = f.read("PID ", 2)

            local_start = overlap_start - file_start
            local_end   = overlap_end   - file_start
            pids.append(pid_in_file[local_start:local_end])
            del pid_in_file

            if file_end >= entity_end:
                break

        return np.concatenate(pids) if pids else np.array([], dtype=np.int64)

    # ── PID helpers (subhalos) ─────────────────────────────────────────────────

    def _build_subhalo_pid_file_boundaries(self) -> list:
        """
        Build a lookup table of {file, start, end} for subhalo PID files (block 2).
        Mirrors _build_pid_file_boundaries for the subhalo PID block.
        Replaces the old eager full-array load of self._subhalos_pid.
        """
        boundaries     = []
        current_offset = 0

        for group_file in g3read.yield_all_files(self._group_base):
            f           = g3read.GadgetFile(group_file, is_snap=False)
            pid_in_file = f.read("PID ", 2)

            if pid_in_file is None or len(pid_in_file) == 0:
                continue

            n = len(pid_in_file)
            boundaries.append({
                "file":  group_file,
                "start": current_offset,
                "end":   current_offset + n,
            })
            current_offset += n
            del pid_in_file

        logger.debug(
            f"Subhalo PID boundaries: {len(boundaries)} files, "
            f"total subhalo PIDs: {current_offset}"
        )
        return boundaries

    def _get_subhalo_pid_for_entity(self, offset: int, length: int) -> np.ndarray:
        """
        Read subhalo PIDs for a specific subhalo on demand using SOFF/SLEN.
        Only reads the file(s) that contain the required offset range.
        """
        if length == 0:
            return np.array([], dtype=np.int64)

        entity_start = offset
        entity_end   = offset + length
        pids         = []

        for boundary in self._subhalo_pid_boundaries:
            file_start = boundary["start"]
            file_end   = boundary["end"]

            overlap_start = max(entity_start, file_start)
            overlap_end   = min(entity_end,   file_end)

            if overlap_start >= overlap_end:
                continue

            f           = g3read.GadgetFile(boundary["file"], is_snap=False)
            pid_in_file = f.read("PID ", 2)

            local_start = overlap_start - file_start
            local_end   = overlap_end   - file_start
            pids.append(pid_in_file[local_start:local_end])
            del pid_in_file

            if file_end >= entity_end:
                break

        return np.concatenate(pids) if pids else np.array([], dtype=np.int64)

    # ── halo fetching ──────────────────────────────────────────────────────────

    def get_halos(self, *args, **kwargs) -> HaloContext:
        return self._get_halos()

    def _get_halos(self) -> HaloContext:
        logger.debug("Fetching halos from Magneticum...")
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()

        if "FSUB" not in self.halo_fields:
            self.halo_fields.append("FSUB")
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.halo["FSUB"] = {
                "input": None, "output": None, "scaling": None, "name": "fsub"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map["FSUB"] = "fsub"

        halos = g3read.read_new(
            self._group_base, self.halo_fields.copy(),
            -1, is_snap=False, multiple_files=True
        )

        if len(halos) == 0:
            logger.warning("No halos found.")
            return HaloContext(pd.DataFrame(), self.column_map)

        table           = "halo"
        halo_conditions = {
            col: data
            for col, data in self.catalog.fields.halo.items()
            if "filter" in data
        }
        steps = [
            lambda h: remove_useless_columns(h, self.halo_fields),
            lambda h: assign_id(h, "halo_id"),
            lambda h: validate_and_process_fields(h, self.halo_fields, table, self.catalog),
            lambda h: convert_to_dataframe(h),
            lambda h: apply_unit_conversion(h, self.halo_fields, table, self.catalog),
            lambda h: apply_filters(h, halo_conditions, self.catalog.fields.halo),
        ]
        processed_halos = run_pipeline(halos, steps)

        # Vectorised subhalo children assignment
        halo_ids  = processed_halos["halo_id"].values
        first_sub = self._fsub[halo_ids]
        n_subs    = self._nsub[halo_ids]
        processed_halos["subhalo_ids"] = [
            np.arange(first_sub[i], first_sub[i] + n_subs[i])
            for i in range(len(halo_ids))
        ]
        self.column_map["subhalo_ids"] = "subhalo_ids"

        halo_context = HaloContext(processed_halos, self.column_map)
        logger.debug(
            f"Halo data fetched: {halo_context.data['halo_id'].nunique()} halos."
        )
        return halo_context

    # ── subhalo fetching ───────────────────────────────────────────────────────

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext | LazySubhaloContext:
        return self._get_subhalos()

    def _get_subhalos(self, batch_size: int = 10_000) -> SubhaloContext | LazySubhaloContext:
        logger.debug("Fetching subhalos from Magneticum in batches...")
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()

        path             = self.catalog.path
        subhalos_parquet = os.path.join(path, "subhalos_parquet")

        if os.path.exists(subhalos_parquet):
            logger.info(f"Loading preprocessed subhalos from: {subhalos_parquet}")
            subhalo_context = LazySubhaloContext(subhalos_parquet)
            logger.debug(
                f"Subhalo data fetched: {subhalo_context.dataset.count_rows()} rows."
            )
            return subhalo_context

        if "GRNR" not in self.subhalo_fields:
            self.subhalo_fields.append("GRNR")
            OmegaConf.set_struct(self.catalog, False)
            self.catalog.fields.subhalo["GRNR"] = {
                "input": None, "output": None, "scaling": None, "name": "id_parents"
            }
            OmegaConf.set_struct(self.catalog, True)
            self.column_map["GRNR"] = "id_parents"

        # TODO
        computed_columns   = self.function_needs.get("subhalo", {})
        catalog_fields     = [f for f in self.subhalo_fields if f not in computed_columns]
        table              = "subhalo"
        subhalo_conditions = {
            col: data
            for col, data in self.catalog.fields.subhalo.items()
            if "filter" in data
        }

        processed_batches = []
        global_offset     = 0

        for group_file in g3read.yield_all_files(self._group_base):
            try:
                raw = g3read.read_new(
                    group_file, catalog_fields, -1,
                    is_snap=False, multiple_files=False
                )
                if raw is None or len(raw) == 0:
                    continue

                n_in_file = len(next(iter(raw.values())))
                logger.debug(f"File {group_file}: {n_in_file} subhalos")

                for batch_start in range(0, n_in_file, batch_size):
                    batch_end = min(batch_start + batch_size, n_in_file)
                    batch     = {k: v[batch_start:batch_end] for k, v in raw.items()}

                    # default-arg trick captures current offset correctly
                    steps = [
                        lambda s: remove_useless_columns(s, self.subhalo_fields),
                        lambda s, o=global_offset + batch_start: assign_id(s, "subhalo_id", offset=o),
                        lambda s: validate_and_process_fields(s, self.subhalo_fields, table, self.catalog),
                        lambda s: convert_to_dataframe(s),
                        lambda s: apply_unit_conversion(s, self.subhalo_fields, table, self.catalog),
                        lambda s: apply_filters(s, subhalo_conditions, self.catalog.fields.subhalo),
                    ]
                    processed = run_pipeline(batch, steps)

                    if processed is not None and len(processed) > 0:
                        processed_batches.append(processed)

                    del batch, processed
                    gc.collect()

                global_offset += n_in_file
                del raw
                gc.collect()

            except Exception as e:
                logger.warning(f"Skipping file {group_file}: {e}")
                continue

        if not processed_batches:
            logger.warning("No subhalos found.")
            return SubhaloContext(pd.DataFrame(), self.column_map)

        logger.debug(f"Storing {len(processed_batches)} batches to {subhalos_parquet}...")

        for df in processed_batches:
            df = df.rename(columns=self.column_map)
            df["parent_bucket"]      = df["id_parents"] % 128
            df["parent_bucket_data"] = df["parent_bucket"]
            arrow_table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_to_dataset(
                arrow_table,
                root_path=subhalos_parquet,
                partition_cols=["parent_bucket"],
                use_threads=True,
            )

        del processed_batches
        gc.collect()

        subhalo_context = LazySubhaloContext(subhalos_parquet)
        logger.debug(
            f"Subhalo data fetched: {subhalo_context.dataset.count_rows()} rows."
        )
        return subhalo_context

    # ── halo interval index (for particle → halo assignment) ──────────────────

    def _build_halo_intervals(self):
        """
        Build compact halo representation without expanding particle IDs.
        Each halo is represented by (min_pid, max_pid, halo_id).
        """
        intervals = []

        for halo_id in range(len(self._glen)):
            # Guard against empty halos (GLEN == 0) to avoid pids.min() raising
            if int(self._glen[halo_id]) == 0:
                continue

            pids = self._get_pid_for_entity(
                offset=int(self._goff[halo_id]),
                length=int(self._glen[halo_id]),
            )

            if len(pids) == 0:
                continue

            intervals.append((int(pids.min()), int(pids.max()), halo_id))

        intervals.sort(key=lambda x: x[0])
        self._halo_intervals = intervals

    def _assign_halo_ids(self, ids: np.ndarray) -> np.ndarray:
        """
        Vectorised halo assignment via interval scanning.
        Returns array of halo_ids, -1 where unassigned.
        Note: uses min/max PID bounds per halo — particles whose PID falls
        inside an interval but doesn't truly belong are misassigned. This is
        a known approximation; replace with exact set lookup for full accuracy.
        """
        if len(ids) == 0:
            return np.array([], dtype=np.int32)

        halo_ids = np.full(len(ids), -1, dtype=np.int32)
        id_max   = ids.max()

        for start, end, halo_id in self._halo_intervals:
            if start > id_max:
                break
            mask           = (ids >= start) & (ids <= end)
            halo_ids[mask] = halo_id

        return halo_ids

    # ── subhalo index (for particle → subhalo assignment) ─────────────────────

    def _build_subhalo_index(self):
        """
        Build a compact subhalo index using exact SOFF/SLEN slices
        into the subhalo PID file boundaries.

        Structure:
            self._subhalo_index[subhalo_idx] = (start, end)
            where start/end are logical offsets into the subhalo PID stream,
            consumed on demand via _get_subhalo_pid_for_entity.
        """
        soff = self._soff
        slen = self._slen

        self._subhalo_index = []

        for sub_idx in range(len(soff)):
            start = int(soff[sub_idx])
            end   = start + int(slen[sub_idx])
            self._subhalo_index.append((start, end))

    def _build_subhalo_pid_arrays(self):
        """
        Build two sorted numpy arrays for subhalo PID lookup:
            _subhalo_sorted_pids     : int64 — every PID belonging to any subhalo
            _subhalo_sorted_ids      : int32 — corresponding subhalo index

        Replaces the Python dict approach (_pid_to_subhalo) which OOM'd at
        ~5M subhalos because a Python dict costs ~200 bytes per entry.
        Two int arrays cost 12 bytes per entry — ~16x more memory-efficient.

        Memory estimate for 5M subhalos with avg 100 PIDs each (500M entries):
            dict  : ~100 GB
            arrays: ~6 GB   ← still large but fits a cluster node
        """
        logger.info("Building subhalo PID sorted arrays...")
        total_pids = int(self._slen.sum())
        logger.info(f"Total subhalo PIDs to index: {total_pids:,}")
        logger.info(f"Subhalo PID boundary files : {len(self._subhalo_pid_boundaries)}")
        logger.info(f"Subhalo index entries      : {len(self._subhalo_index)}")

        if len(self._subhalo_pid_boundaries) == 0:
            logger.warning(
                "No subhalo PID boundary files found — "
                "_subhalo_sorted_pids will be empty. "
                "Check that _build_subhalo_pid_file_boundaries ran correctly."
            )
        if total_pids == 0:
            logger.warning("All SLEN values are 0 — no subhalo PIDs to index.")

        all_pids = np.empty(total_pids, dtype=np.int64)
        all_sids = np.empty(total_pids, dtype=np.int32)

        cursor = 0
        for sub_idx, (start, end) in enumerate(self._subhalo_index):
            length = end - start
            if length == 0:
                continue
            pids = self._get_subhalo_pid_for_entity(offset=start, length=length)
            n = len(pids)
            all_pids[cursor:cursor + n] = pids
            all_sids[cursor:cursor + n] = sub_idx
            cursor += n

            if sub_idx % 100_000 == 0:
                logger.info(f"  indexed {sub_idx:,} / {len(self._subhalo_index):,} subhalos")

        # trim in case some subhalos were empty
        all_pids = all_pids[:cursor]
        all_sids = all_sids[:cursor]

        sorter = np.argsort(all_pids, kind="stable")
        self._subhalo_sorted_pids = all_pids[sorter]
        self._subhalo_sorted_ids  = all_sids[sorter]

        logger.info(f"Subhalo PID arrays built: {cursor:,} entries")

    def _assign_subhalo_ids(self, particle_ids: np.ndarray) -> np.ndarray:
        """
        Vectorised subhalo assignment via np.searchsorted on sorted PID arrays.
        Returns array of subhalo indices, -1 where unassigned.
        No Python dict — O(log N) per lookup, O(1) extra memory per call.
        """
        if not hasattr(self, "_subhalo_sorted_pids"):
            self._build_subhalo_pid_arrays()

        if len(particle_ids) == 0:
            return np.array([], dtype=np.int32)

        if len(self._subhalo_sorted_pids) == 0:
            logger.warning("_subhalo_sorted_pids is empty — no subhalo PIDs indexed.")
            return np.full(len(particle_ids), -1, dtype=np.int32)

        indices = np.searchsorted(self._subhalo_sorted_pids, particle_ids)
        indices = np.clip(indices, 0, len(self._subhalo_sorted_pids) - 1)

        result          = np.full(len(particle_ids), -1, dtype=np.int32)
        matched         = self._subhalo_sorted_pids[indices] == particle_ids
        result[matched] = self._subhalo_sorted_ids[indices[matched]]
        return result

    # ── particle pipeline ──────────────────────────────────────────────────────

    def _run_particle_pipeline(self, data: dict) -> dict:
        """Shared pipeline logic for both halo and subhalo particles."""
        table = "particles"
        steps = [
            lambda particles: validate_and_process_fields(
                particles, self.particle_fields, table, self.catalog
            ),
            lambda particles: convert_to_dataframe(particles),
            lambda particles: apply_unit_conversion(
                particles, self.particle_fields, table, self.catalog
            ),
            lambda particles: apply_filters(
                particles, self._particle_conditions, self.catalog.fields.particles
            ),
        ]
        return run_pipeline(data, steps)

    def _store_particles(self) -> LazySubhaloParticles:
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()
            self._load_particle_cache()

        logger.info("Starting star particle ingestion pipeline")
        path = self.catalog.path
        particles_parquet = os.path.join(path, "particles_parquet")

        # Build global maps once before the file loop
        self._build_halo_intervals()
        self._build_subhalo_index()
        self._build_subhalo_pid_arrays()

        # Pre-compute which buckets are already done
        n_buckets = 128
        completed_buckets = set()
        if os.path.exists(particles_parquet):
            for entry in os.scandir(particles_parquet):
                if entry.is_dir() and entry.name.startswith("parent_bucket="):
                    try:
                        bucket_id = int(entry.name.split("=")[1])
                        completed_buckets.add(bucket_id)
                    except ValueError:
                        continue
            logger.info(
                f"Resuming — {len(completed_buckets)}/{n_buckets} buckets already done: {sorted(completed_buckets)}")
        else:
            logger.info("No existing particles_parquet — starting fresh")

        if len(completed_buckets) == n_buckets:
            logger.info("All buckets already processed — skipping ingestion")
            return LazySubhaloParticles(particles_parquet)

        for part_type in self.particle_fields:
            if part_type not in ("stars", "star"):
                continue

            config = self._snap_cache_config[part_type]
            fields_padded = config["fields_padded"]
            ptype_int = config["ptype_int"]

            for snap_file in g3read.yield_all_files(self._snap_base):
                snap_data = g3read.read_new(
                    snap_file, fields_padded, ptype_int,
                    multiple_files=False, do_flatten=False
                )[ptype_int]

                if snap_data is None or len(snap_data) == 0:
                    continue

                ids = snap_data["ID  "]
                halo_ids = self._assign_halo_ids(ids)
                subhalo_ids = self._assign_subhalo_ids(ids)

                mask = halo_ids != -1
                if not np.any(mask):
                    continue

                data = {k: v[mask] for k, v in snap_data.items()}
                data["parent_id"] = halo_ids[mask]
                data["subhalo_id"] = subhalo_ids[mask]
                data["parent_bucket"] = halo_ids[mask] % 128
                data["parent_bucket_data"] = data["parent_bucket"].copy()

                # ── Skip rows that belong to already completed buckets ─────────
                pending_mask = ~np.isin(data["parent_bucket"], list(completed_buckets))
                if not np.any(pending_mask):
                    logger.debug(f"All particles in {snap_file} belong to completed buckets — skipping")
                    del snap_data, data
                    gc.collect()
                    continue

                data = {k: v[pending_mask] for k, v in data.items()}
                # ─────────────────────────────────────────────────────────────────

                for field_name, array in data.items():
                    if isinstance(array, np.ndarray) and array.ndim > 1:
                        data[field_name] = to_list_in_batches(data, field_name)

                data_particles = {part_type: data}
                processed = self._run_particle_pipeline(data_particles)
                arrow_table = pa.Table.from_pydict(processed[part_type])
                pq.write_to_dataset(
                    arrow_table,
                    root_path=particles_parquet,
                    partition_cols=["parent_bucket"],
                )

                del snap_data, data, processed, arrow_table
                gc.collect()

        gc.collect()
        logger.info("Star particle ingestion completed")

        subhalo_particles = LazySubhaloParticles(particles_parquet)
        logger.debug(f"Particles fetched: {subhalo_particles.dataset.count_rows()} rows.")
        return subhalo_particles

    def get_p_subhalo(self, *args, **kwargs):
        subhalo_id = kwargs.get('subhalo_id', None)

        if subhalo_id is None:
            logger.error("subhalo_id must be provided in kwargs")
            raise ValueError("subhalo_id must be provided in kwargs")

        # logger.info(f"Fetching particle data for subhalo: {subhalo_id}")
        return self._get_p_subhalo(subhalo_id)

    def _get_p_subhalo(self, subhalo_id: int) -> SubhaloParticles:
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()
            self._load_particle_cache()

        path = self.catalog.path
        particles_parquet = os.path.join(path, "particles_parquet")

        if not os.path.exists(particles_parquet):
            self._store_particles()

        return self._load_subhalo_particles(subhalo_id, particles_parquet)

    def _load_subhalo_particles(self, subhalo_id: int, particles_parquet: str) -> SubhaloParticles:
        """
        Load particles for a specific subhalo directly from parquet.
        No PID reindexing needed — subhalo_id is already stored as a column.
        Uses parent_bucket = halo_id % 128 to avoid full scan.
        """
        # Resolve parent halo to get the correct bucket
        halo_id = self._get_halo_id_for_subhalo(subhalo_id)
        bucket = halo_id % 128 if halo_id != -1 else subhalo_id % 128

        logger.debug(f"Loading particles for subhalo_id={subhalo_id} bucket={bucket}")

        try:
            filters = [
                ('parent_bucket', '==', bucket),
                ('subhalo_id', '==', subhalo_id),
            ]
            if halo_id != -1:
                filters.append(('parent_id', '==', halo_id))

            df = pd.read_parquet(particles_parquet, filters=filters)

        except Exception as e:
            logger.warning(f"Failed to load particles for subhalo_id={subhalo_id}: {e}")
            return SubhaloParticles({}, self.catalog.fields.particles)

        if len(df) == 0:
            logger.warning(f"No particles found for subhalo_id={subhalo_id}")
            return SubhaloParticles({}, self.catalog.fields.particles)

        logger.debug(f"Loaded {len(df)} particles for subhalo_id={subhalo_id}")

        data = {}
        for col in df.columns:
            if col in ('parent_id', 'subhalo_id', 'parent_bucket', 'parent_bucket_data'):
                continue
            data[col] = df[col].to_numpy()

        return SubhaloParticles(data=data, column_map=None)

    def _get_halo_id_for_subhalo(self, subhalo_id: int) -> int:
        """
        Resolve the parent halo_id for a given subhalo_id using FSUB/NSUB metadata.
        Returns -1 if not found.
        """
        for halo_id, (first_sub, n_sub) in enumerate(zip(self._fsub, self._nsub)):
            if first_sub <= subhalo_id < first_sub + n_sub:
                return halo_id
        return -1

    def get_p_halo(self, *args, **kwargs) -> SubhaloParticles | LazySubhaloParticles:
        halo = kwargs.get("halo", None)
        if halo is None:
            raise ValueError("halo must be provided in kwargs")
        return self._get_p_halo(halo)

    def _load_halo_particles(self, halo_id: int, particles_parquet: str) -> SubhaloParticles:
        """
        Load particles for a specific halo from parquet.
        Uses parent_bucket partition to avoid full scan,
        then filters by parent_id == halo_id.
        """
        bucket = halo_id % 128

        logger.debug(f"Loading particles for halo_id={halo_id} bucket={bucket}")

        try:
            df = pd.read_parquet(
                particles_parquet,
                filters=[
                    ('parent_bucket', '==', bucket),
                    ('parent_id', '==', halo_id),
                ]
            )
        except Exception as e:
            logger.warning(f"Failed to load particles for halo_id={halo_id}: {e}")
            return SubhaloParticles({}, self.catalog.fields.particles)

        if len(df) == 0:
            logger.warning(f"No particles found for halo_id={halo_id}")
            return SubhaloParticles({}, self.catalog.fields.particles)

        logger.debug(f"Loaded {len(df)} particles for halo_id={halo_id}")

        # Convert DataFrame back to SubhaloParticles format
        data = {}
        for col in df.columns:
            if col in ('parent_id', 'subhalo_id', 'parent_bucket', 'parent_bucket_data'):
                continue
            data[col] = df[col].to_numpy()

        return SubhaloParticles(data=data, column_map=None)

    def _get_p_halo(self, halo) -> SubhaloParticles | LazySubhaloParticles:
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()
            self._load_particle_cache()

        path              = self.catalog.path
        particles_parquet = os.path.join(path, "particles_parquet")

        if not os.path.exists(particles_parquet):
            self._store_particles()

        return self._load_halo_particles(halo.halo_id, particles_parquet)

    def _compute_features(self, table):
        # function needs dictionary subhalo: {'stellar_half_mass_radius': {'star': ['POS', 'MASS']}}
        logger.info(self.function_needs)
        methods_to_compute = self.function_needs.get(table, {})
        for method in methods_to_compute:
            logger.debug(f'Computing feature: {method}')
            #result = FeatureComputer.compute(method, empty_particles, {})










