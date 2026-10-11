import copy
import gc
import operator
import os.path
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import g3read
import h5py
from cython_fortran_file import FortranFile
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
from src.data_generation.src.subhalo_particles_v2 import SubhaloParticlesV2
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
        """Call `func(data, *args, **kwargs)`, short-circuiting to None if `data` is None."""
        if data is None:
            return None  # Stop processing if data is None
        return func(data, *args, **kwargs)
    return wrapper


@pipeline_step
def remove_useless_columns(d: dict, fields: list):
    """
    Drop every key of `d` not present in `fields`, in place.

    Parameters:
    -----------
    d : dict
        Dictionary of column name to array, as returned by a raw catalogue
        read.
    fields : list
        Column names to keep (typically `self.halo_fields`/
        `self.subhalo_fields`/`self.particle_fields`, including any
        dynamically-injected structural fields).

    Returns:
    --------
    dict
        `d`, with unlisted keys removed.
    """
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
                    if cfg.name.startswith('magneticum'):
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

    # Lbox: box size, needed by any catalogue whose raw positions are in
    # box units (e.g. Obelisk's x/y/z in [0,1]) rather than physical/comoving
    # length units already. Exposed unconditionally in the scaling eval
    # context below; catalogues that don't reference it in their scaling
    # strings are unaffected.
    lbox = OmegaConf.select(cfg, "Lbox")

    for key in fields:
        if key in df:
            field_info = OmegaConf.select(cfg, f"fields.{table}.{key}")
            height = dict_depth(field_info)
            if height == 2:
                for key_ii in field_info:
                    field_info_ii = OmegaConf.select(cfg, f"fields.{table}.{key}.{key_ii}")
                    if field_info_ii.scaling is not None:
                        if cfg.name.startswith('magneticum'):
                            key_ii = key_ii.ljust(4)
                        df[key][key_ii] = apply_conversion(df[key][key_ii], field_info_ii.scaling, cfg.a, cfg.H,
                                                           Lbox=lbox, **kwargs)
            else:
                if field_info.scaling is not None:
                    df[key] = apply_conversion(df[key], field_info.scaling, cfg.a, cfg.H, Lbox=lbox, **kwargs)
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
        """
        Store the catalogue configuration and derive field lists/column mappings from it.

        Parameters:
        -----------
        catalog : DictConfig
            Catalogue configuration (e.g. `configs/catalog/magneticum.yaml`),
            with a `fields` section (`halo`/`subhalo`/`particles`).
        **kwargs
            Extra attributes to set on this instance (see
            `__build_config`).
        """
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
        """No-op default; subclasses may override to report their available halo/subhalo field names."""
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
        """
        Collect every field's `needs` declaration (a FeatureComputer dependency), grouped by table.

        Scans `self.catalog.fields` (halo/subhalo/particles) for entries
        with a `needs` key, e.g. `stellar_half_mass_radius: {needs: {star:
        [POS, MASS]}}`. Used by `__build_config` to validate that every
        referenced FeatureComputer function is actually registered.

        Returns:
        --------
        dict
            `{table_name: {field_name: needs_value}}` for every field that
            declares `needs`.
        """
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
        elif catalog.name == "flamingo_v2":
            return FlamingoFetcherV2(catalog, **kwargs)
        elif catalog.name == "magneticum_v2":
            return MagneticumFetcherV2(catalog, **kwargs)
        elif catalog.name == "obelisk":
            return ObeliskFetcher(catalog, **kwargs)
        else:
            raise ValueError(f"Unsupported catalog={catalog.name} type")


class Illustris300Fetcher(DataFetcher):
    """
    Data fetcher for the Illustris-300 catalog.
    """
    # Particle types loaded for the overlapping neighbour subhalos (get_p_subhalo, only used by the neighbour
    # path of the data loader). Neighbours only need their stars: the heavy dark matter of the neighbours is
    # not used by default, and with several particle types per catalog it would multiply the loading time.
    # Only applied when the catalog sets `particles_v2: true` (see `_particles_v2`).
    NEIGHBOUR_PARTICLE_TYPES = ('star',)

    @property
    def _particles_v2(self) -> bool:
        """
        bool: True if the catalog opts in to the multi particle type behaviour with `particles_v2: true`.

        The flag switches on three things together: `SubhaloParticlesV2` instead of `SubhaloParticles`, the
        constant particle masses (dark matter) taken from the snapshot header, and stars only for the overlapping
        neighbours. Catalogs without it behave exactly as before, even if they declare several particle types.
        """
        return bool(self.catalog.get('particles_v2', False))

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
        return self._get_p_subhalo(subhalo_id, particle_types=self.NEIGHBOUR_PARTICLE_TYPES)

    def get_p_halo(self, *args, **kwargs) -> SubhaloParticles:
        """
        Fetch particle data for a halo.

        Parameters:
        -----------
        halo : Halo
            Required keyword argument: the halo to fetch particles for.

        Returns:
        --------
        SubhaloParticles
            Particle data for the halo.

        Raises:
        -------
        ValueError
            If `halo` is not provided in kwargs.
        """
        halo = kwargs.get('halo', None)
        if halo is None:
            logger.error("halo must be provided in kwargs")
            raise ValueError("halo must be provided in kwargs")

        #logger.info(f"Fetching particle data for halo: {halo.halo_id}")
        return self._get_p_halo(halo)

    def __get_children(self, halo_row, subhalo_df):
        """
        Return the subhalo IDs belonging to one halo row.

        Parameters:
        -----------
        halo_row : pd.Series
            A row from the halo DataFrame; only `halo_id` is used.
        subhalo_df : pd.DataFrame
            Subhalo DataFrame with an `id_parents` column to match against.

        Returns:
        --------
        list
            `subhalo_id` values whose `id_parents` equals this halo's ID.
        """
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

    def _get_mass_table(self) -> np.ndarray:
        """
        Return the `MassTable` of the snapshot header, read once from chunk 0 and cached.

        Particle types with a constant mass (dark matter in Illustris, TNG and MTNG) have no per-particle
        `Masses` dataset in the snapshot, their mass is in this table, in the snapshot's mass units
        (10^10 Msun/h). `illustris_python` has no snapshot header reader (its group catalogue header has no
        `MassTable`), so the file that `il.snapshot.snapPath` returns is opened with h5py.

        Returns:
        --------
        np.ndarray
            Mass of each of the 6 particle types, 0 for the types that store per-particle masses.
        """
        if getattr(self, '_mass_table', None) is None:
            snapshot_file = il.snapshot.snapPath(self.catalog.path + '/output', self.catalog.snapnum)
            with h5py.File(snapshot_file, 'r') as snapshot:
                self._mass_table = np.asarray(snapshot['Header'].attrs['MassTable'], dtype=np.float64)
        return self._mass_table

    def _constant_mass(self, part_type: str, fields: list):
        """
        Return the constant particle mass of `part_type` if its `Masses` have to come from the header.

        Following the Gadget convention, a particle type with a non-zero `MassTable` entry stores no
        per-particle masses, so a `Masses` field requested for it cannot be read from the snapshot.

        Parameters:
        -----------
        part_type : str
            Particle type key of the catalog YAML (e.g. 'star', 'dm').
        fields : list
            Snapshot fields requested for this particle type.

        Returns:
        --------
        float or None
            The header mass (snapshot mass units) when `Masses` is requested and the type has a constant
            mass, otherwise None (everything is read from the snapshot).
        """
        # Catalogs without `particles_v2` keep the original behaviour exactly: no header access
        if 'Masses' not in fields or not self._particles_v2:
            return None
        mass = float(self._get_mass_table()[il.util.partTypeNum(part_type)])
        return mass if mass > 0 else None

    @staticmethod
    def _particle_count(particles: dict) -> int:
        """Number of particles in a loaded particle dict (length of its first array)."""
        for value in particles.values():
            if isinstance(value, np.ndarray):
                return len(value)
        return 0

    def _select_particle_types(self, particle_types=None) -> list:
        """
        Particle types to load, in the order of the catalog YAML.

        `particle_types` restricts the selection, but only for catalogs with `particles_v2: true` that declare
        more than one particle type, so every other catalog loads all its types as before. If the restriction
        matches nothing, every type is loaded.

        Parameters:
        -----------
        particle_types : iterable of str, optional
            Particle type keys to keep. None keeps every declared type.
        """
        declared = [part_type for part_type in self.catalog.fields.particles.keys()
                    if part_type in self.particle_fields]
        if particle_types is None or not self._particles_v2 or len(declared) <= 1:
            return declared
        selected = [part_type for part_type in declared if part_type in particle_types]
        if not selected:
            logger.warning(f'None of the requested particle types {list(particle_types)} is declared in the '
                           f'catalog {declared}, loading all of them.')
            return declared
        return selected

    def _get_p_subhalo(self, subhalo_id: int, particle_types=None) -> SubhaloParticles:
        """
        Fetches particle data for a specific subhalo.

        Particle types with a constant mass in the snapshot header (dark matter) do not have a `Masses`
        dataset: if the catalog declares `Masses` for them, it is not requested from the snapshot and is
        filled with the header mass instead, so it goes through the usual unit conversion of the YAML.

        Catalogs with `particles_v2: true` get a `SubhaloParticlesV2` (several particle types need their own
        overlap boundaries), all the others keep `SubhaloParticles`.

        Parameters:
        -----------
        subhalo_id : int
            The index of the subhalo from which to retrieve particle data.
        particle_types : iterable of str, optional
            Restrict the loading to these particle type keys of the catalog (see `_select_particle_types`).

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

        part_types = self._select_particle_types(particle_types)
        data = {}
        for part_type in part_types:
            fields = list(self.catalog.fields.particles[part_type].keys())
            constant_mass = self._constant_mass(part_type, fields)
            request = fields if constant_mass is None else [field for field in fields if field != 'Masses']
            if not request:
                raise ValueError(f"Particle type '{part_type}' only declares 'Masses', which comes from the "
                                 f"header, so there is no snapshot field to size the particles with.")
            particles = il.snapshot.loadSubhalo(path, self.catalog.snapnum, subhalo_id, part_type, fields=request)

            # if numpy, convert it to a dict
            if isinstance(particles, np.ndarray):
                tmp = particles
                particles = {request[0]: tmp}

            if len(particles) > 0:
                count = particles.get('count', None)
                if count is None or count > 0:
                    if constant_mass is not None:
                        particles['Masses'] = np.full(self._particle_count(particles), constant_mass,
                                                      dtype=np.float64)
                    data[part_type] = particles
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
        particles_class = SubhaloParticlesV2 if self._particles_v2 else SubhaloParticles
        particle_data = particles_class(processed_particles, self.catalog.fields.particles)
        return particle_data

    def _get_p_halo(self, halo: Halo) -> SubhaloParticles:
        """
        Fetch and concatenate particle data for every subhalo of a halo.

        Parameters:
        -----------
        halo : Halo
            Halo whose `subhalo_ids` are fetched and merged via
            `_get_p_subhalo` + `SubhaloParticles.__add__`.

        Returns:
        --------
        SubhaloParticles
            The combined particle data for all of the halo's subhalos,
            with `overlapping_particles_id` reset to None (this merge
            isn't the overlapping-neighbours kind `non_overlapping` slices
            on).

        Raises:
        -------
        Exception
            If the catalogue's `output` directory does not exist.
        """
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

        particles_halo.overlapping_particles_id = None
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
        """Not implemented for EagleFetcher; inherits the no-op base behaviour (fields are declared via config, not introspected)."""
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
    """
    Data fetcher for the Flamingo SOAP catalogue (SWIFT-based simulation).

    Loads `self.fof_subhalos` (halo/subhalo properties) and `self.particles`
    (particle data) eagerly, as SWIFTDataset objects, in `__init__` via
    `load_files()` — unlike Magneticum's lazy, cache-flag-gated loading.
    Halos and subhalos are the same underlying rows (`self.fof_subhalos`),
    filtered by `soap.host_halo_index == -1` for primaries (halos) vs. all
    rows (subhalos). See `FlamingoFetcherV2` for the row-position vs.
    true-catalogue-index particle-matching fix.
    """
    def __init__(self, catalog: DictConfig, **kwargs):
        """
        Initialize the fetcher and eagerly load the SOAP catalogue and particle files.

        Parameters:
        -----------
        catalog : DictConfig
            Catalogue configuration, with `path` and `snapnum` used by
            `load_files()` to locate the HDF5 files.
        **kwargs
            Forwarded to `DataFetcher.__init__`.
        """
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

        # configs/catalog/flamingo.yaml's Lbox is the box's COMOVING size
        # (677400 kpc/h == 1000 comoving Mpc, the "L1000" in L1000N0900,
        # times h — verified against this file's own metadata.boxsize).
        # But _validate_and_process_fields() converts every SOAP/particle
        # position field to PHYSICAL units via .convert_to_physical()
        # before this fetcher's "x * h" scaling ever runs. Comparing those
        # physical deltas against a still-comoving Lbox/2 in the periodic-
        # boundary correction (SubhaloContext.preprocess /
        # SubhaloParticles.preprocess) silently mis-wraps objects near the
        # box edge. Converting Lbox to physical once here, using the
        # snapshot's own true scale factor (not the YAML's separately
        # hand-entered `a`, which is never cross-checked against the file),
        # keeps it consistent with the positions everywhere it's later read
        # from self.catalog.Lbox — no changes needed to the shared PBC code
        # or to other fetchers, which stay fully comoving and are
        # unaffected by this mutation (it's scoped to this instance's copy
        # of the config). Lbox itself stays optional (some test catalogs
        # omit it, since not every test exercises PBC-dependent behaviour).
        lbox = OmegaConf.select(self.catalog, "Lbox", default=None)
        if lbox is not None:
            self.catalog.Lbox = lbox * float(self.fof_subhalos.metadata.a)

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
                logger.warning(f'Halo {halo.halo_id} has no particles (all particles_mask_primary are false).')
                raise ValueError(f'Halo {halo.halo_id} has no particles (all particles_mask_primary are false).')

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

        particle_data = SubhaloParticles(d, self.catalog.fields.particles)
        logger.debug(f'Stellar particle size: {len(particle_data.star_p)}')
        return particle_data


class FlamingoFetcherV2(FlamingoFetcher):
    """
    Fixes FlamingoFetcher's star-particle -> halo/subhalo assignment bug.

    The SOAP catalogue exposes two coexisting, non-interchangeable ID spaces
    for the SAME row-space (self.fof_subhalos, loaded from
    halo_properties_<redshift>.hdf5 — halos and subhalos are literally the
    same rows, just filtered differently by soap.host_halo_index):

      - row-position space: the index of a row within self.fof_subhalos, as
        it was loaded. V1's assign_id() assigns exactly this (np.arange(N))
        as `halo_id` / `subhalo_id`. soap.host_halo_index (used for
        satellite<->parent linkage) is ALSO expressed in this space —
        verified on real data: interpreting HostHaloIndex as a direct row
        index gives physically sane satellite-to-parent distances
        (sub-Mpc); any other interpretation gives distances of hundreds of
        Mpc. So host/satellite linkage under V1's row-position halo_id is
        already correct.
      - true catalogue-index space: input_halos.halo_catalogue_index (halo
        table) and the per-particle halo_catalogue_index fields
        (PartType4/HaloCatalogueIndex, etc. in flamingo_<redshift>.hdf5) use
        a separate, shuffled numbering that has nothing to do with row
        position — confirmed on real data: HaloCatalogueIndex is not
        np.arange(N), and a halo's row-position id (e.g. 639531) generally
        refers to an entirely different physical halo than the one whose
        TRUE catalogue index happens to equal 639531.

    V1's bug: _get_p_halo's primary-particle mask
    (particles_of_type.halo_catalogue_index == halo.halo_id) compares a
    row-position halo_id directly against a true-catalogue-index particle
    field, with no conversion — matching star particles to an essentially
    unrelated halo elsewhere in the box. The satellite-gathering code three
    lines below it in the SAME method already performs the correct
    conversion (self.fof_subhalos.input_halos.halo_catalogue_index[mask]),
    it's just never applied to the primary-particle line.

    V2 fix (surgical, not a rewrite): convert the row-position id to its
    true catalogue index — via self.fof_subhalos.input_halos.halo_catalogue_index,
    the same SOAP field the satellite-gathering code already uses correctly
    — at exactly the two points that compare against particle data:

      - _get_p_halo: convert halo.halo_id before building the primary mask.
      - _get_p_subhalo: convert the incoming subhalo_id argument once, up
        front, since the rest of the method already treats it as if it were
        a true catalogue index (its own lookup at
        np.where(halo_catalogue_index == subhalo_id) already only makes
        sense in that space).

    Everything else — assign_id, _get_halos(), _get_subhalos(),
    __reindexing_satellites, and the satellite-gathering lines themselves —
    is left untouched (inherited unmodified): it is internally consistent
    under the row-position convention today, and rewriting it would be a
    larger change that risks breaking the currently-correct host/satellite
    linkage for no benefit.
    """

    def _get_p_subhalo(self, subhalo_id: int) -> SubhaloParticles:
        """
        Same as FlamingoFetcher._get_p_subhalo, with one addition: the
        incoming `subhalo_id` is the row-position index produced by
        _get_subhalos() (same convention as halo.halo_id). V1's version
        instead searched for a row whose TRUE catalogue index equals
        `subhalo_id` (`np.where(halo_catalogue_index == subhalo_id)`) — a
        lookup that only makes sense if `subhalo_id` were already a true
        catalogue index, which it isn't. Since `subhalo_id` is a row
        position, the row IS `self.fof_subhalos[subhalo_id]` directly; no
        search is needed, just a bounds check and a direct read of that
        row's true catalogue index (for comparing against particle data,
        which is keyed by true catalogue index, not row position).
        """
        catalogue_index_field = self.fof_subhalos.input_halos.halo_catalogue_index

        if not (0 <= subhalo_id < len(catalogue_index_field)):
            raise ValueError(f'Subhalo: {subhalo_id} is not a primary subhalo')

        subhalo_catalogue_id = catalogue_index_field[subhalo_id]
        host_halo_index = self.fof_subhalos.soap.host_halo_index[subhalo_id]

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
            # Mask to get particles belonging to the primary subhalo — compared
            # in true-catalogue-index space, matching the SOAP lookup above.
            particles_mask_primary = particles_of_type.halo_catalogue_index == subhalo_catalogue_id
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
        Same as FlamingoFetcher._get_p_halo, with one addition: halo.halo_id
        (row-position) is converted to its true SOAP catalogue index, via
        self.fof_subhalos.input_halos.halo_catalogue_index, before it is
        compared against the particle data's halo_catalogue_index field.
        This is the exact conversion the satellite-gathering code below
        already performs (self.fof_subhalos.input_halos.halo_catalogue_index
        [satellites_mask]) — it was just never applied to the primary mask.
        """
        if halo.is_primary != -1:
            # Not a primary index
            raise ValueError(f'Subhalo: {halo.halo_id} is not a primary subhalo')

        # halo.halo_id is a row-position index into self.fof_subhalos — bounds
        # check it before indexing, same as _get_p_subhalo does for
        # subhalo_id, instead of letting an out-of-range id raise a raw
        # IndexError a few lines below.
        catalogue_index_field = self.fof_subhalos.input_halos.halo_catalogue_index
        if not (0 <= halo.halo_id < len(catalogue_index_field)):
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

        # halo.halo_id is a row-position index into self.fof_subhalos; convert
        # it to the true SOAP catalogue index before comparing against
        # particle data, which is keyed by true catalogue index.
        halo_catalogue_id = catalogue_index_field[halo.halo_id]

        part_types = self.particle_fields
        data = {}

        type_mask = {}
        # Build particles mask
        for part_type in part_types:
            particles_of_type = getattr(self.particles, part_type)
            # Mask to get particles belonging to the primary subhalo
            particles_mask_primary = particles_of_type.halo_catalogue_index == halo_catalogue_id
            is_full_false = np.sum(particles_mask_primary)

            # Get satellite subhalos — host_halo_index is row-position, same
            # space as halo.halo_id, so this comparison is already correct.
            satellites_mask = self.fof_subhalos.soap.host_halo_index == halo.halo_id

            # Loop over the halo_catalogue_index of the satellites to get all bound particles
            satellites_idx_array = self.fof_subhalos.input_halos.halo_catalogue_index[satellites_mask]
            particles_mask_satellites = np.isin(particles_of_type.halo_catalogue_index, satellites_idx_array)
            particles_mask = particles_mask_primary | particles_mask_satellites

            type_mask[part_type] = particles_mask

            if np.all(particles_mask_satellites == False):
                logger.warning(f'Halo {halo.halo_id} has no satellites.')

            if np.all(particles_mask_primary == False):
                logger.warning(f'Halo {halo.halo_id} has no particles (all particles_mask_primary are false).')
                raise ValueError(f'Halo {halo.halo_id} has no particles (all particles_mask_primary are false).')

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

        particle_data = SubhaloParticles(d, self.catalog.fields.particles)
        logger.debug(f'Stellar particle size: {len(particle_data.star_p)}')
        return particle_data


class MagneticumFetcherV2(DataFetcher):
    """
    Data fetcher for the Magneticum simulation (g3read/Gadget-family binary format).

    Unlike FlamingoFetcher, catalogue metadata and particle data are
    loaded lazily and independently: `_load_catalog_cache()` (halo/subhalo
    offsets, gated by `self._cache_loaded`) and `_load_particle_cache()`
    (per-particle-type field config, gated by `self._particle_cache_loaded`)
    are each triggered on first use by `_get_halos`/`_get_subhalos`/
    `_store_particles`/`_get_p_subhalo`/`_get_p_halo`, and cached for
    reuse. Particle data is written to/read from a bucketed parquet
    dataset (`_store_particles`) rather than kept fully in memory.

    Star-particle -> halo/subhalo assignment by exact PID matching, under a
    hard memory ceiling (no swap, no bigger --mem available on the cluster).

    The earlier interval-based implementation (a since-removed V1 class)
    approximated "does particle X belong to halo H?" with a [min(PID of H), max(PID of H)] range test (_build_halo_intervals /
    _assign_halo_ids). That is invalid: particle IDs are not ordered by
    spatial/halo membership, so a halo's interval ends up spanning almost the
    entire ID space. Measured on real Magneticum group-catalog data: 17x-103x
    over-inclusion (a halo with e.g. 900K true member particles matches
    ~90M IDs against its own interval). That version also rebuilt each halo's/subhalo's
    interval via a per-entity call (_get_pid_for_entity /
    _get_subhalo_pid_for_entity) that reopens and re-reads the relevant group
    file's "PID " block from disk on every single call, with no caching —
    for millions of halos/subhalos this reads multi-GB blocks off disk
    repeatedly.

    A first corrected version of this class fixed correctness and the redundant
    re-reads by building ONE global sorted (PID -> halo_id, subhalo_id)
    index across the whole group catalogue, held in RAM. That is correct and
    reads each group file only once, but its peak memory is O(N) in the
    TOTAL group-catalogue particle count — it OOM-killed a real production
    run, and this cluster's queue cannot grant more memory. A smaller
    constant factor (dropping a stable sort, freeing intermediates eagerly)
    only bought ~13% (measured: 2.80 GB -> 2.44 GB on a 90M-entry local
    fixture) — nowhere near enough headroom for a production box, since the
    approach still scales with total N regardless of tuning.

    This version restructures the algorithm instead of tuning it: rather
    than building one global index, _assign_halo_ids/_assign_subhalo_ids
    stream the group catalogue's "PID " block once PER CALL (i.e. once per
    snapshot-file chunk from _store_particles()'s existing loop), holding
    only one group file's chunk in memory at a time, and match it directly
    against that chunk's (much smaller) particle IDs via a sorted search.
    Peak memory is now bounded by max(one group file's chunk, one snapshot
    file's particle count) — independent of the total group-catalogue size —
    at the cost of re-reading the group catalogue once per snapshot file
    instead of once total. Since snapshot files are typically far fewer than
    halos (tens, not millions), this trade is favourable under a hard memory
    cap: it fully fixes the O(N) memory scaling that the global-index
    version could not escape, in exchange for a bounded, small multiple of
    extra disk I/O.

    _store_particles() calls
    self._assign_halo_ids(ids) then self._assign_subhalo_ids(ids) with the
    SAME `ids` array back to back, once per snapshot file. Streaming the
    group catalogue separately for each of those two calls would double the
    re-read cost for no reason, so the first call does the full streaming
    pass and computes both results, caching them (keyed by identity of the
    `ids` array) for the second call to reuse.

    """

    def __init__(self, catalog: DictConfig, **kwargs):
        """
        Initialize the fetcher; catalogue/particle data is loaded lazily on first use, not here.

        Parameters:
        -----------
        catalog : DictConfig
            Catalogue configuration, with `path` and `snapnum` used by
            `_load_catalog_cache()`/`_load_particle_cache()` once triggered.
        **kwargs
            Forwarded to `DataFetcher.__init__`.
        """
        super().__init__(catalog, **kwargs)
        logger.info(f"MagneticumFetcherV2 initialized with config: {catalog.path}")

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
        self._particle_cache_loaded = True

    # ── PID helpers (halos) ────────────────────────────────────────────────────

    # ── PID helpers (subhalos) ─────────────────────────────────────────────────

    # ── halo fetching ──────────────────────────────────────────────────────────

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """Fetch halo data from the Magneticum catalogue. See `_get_halos`."""
        return self._get_halos()

    def _get_halos(self) -> HaloContext:
        """
        Load (if not already cached) the catalogue metadata and fetch all halos.

        Also vectorized-assigns each halo's `subhalo_ids` from the FSUB
        (first subhalo index)/NSUB (subhalo count) contiguous-block
        convention, since Magneticum's subhalos are stored as a
        contiguous per-halo range rather than linked via a parent-id
        column.

        Returns:
        --------
        HaloContext
            All halos passing the configured filters, or an empty
            HaloContext if the catalogue has none.
        """
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

        table = "halo"
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
        halo_ids = processed_halos["halo_id"].values
        first_sub = self._fsub[halo_ids]
        n_subs = self._nsub[halo_ids]
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
        """Fetch subhalo data from the Magneticum catalogue. See `_get_subhalos`."""
        return self._get_subhalos()

    def _get_subhalos(self, batch_size: int = 10_000) -> SubhaloContext | LazySubhaloContext:
        """
        Fetch all subhalos, building (or reusing) a bucketed parquet cache.

        If `{catalog.path}/subhalos_parquet` already exists, returns a
        `LazySubhaloContext` over it directly (no re-processing). Otherwise
        streams the group catalogue file by file, processing subhalos in
        `batch_size`-sized chunks (to bound memory) through the standard
        field pipeline, writes the result to that parquet dataset
        (partitioned by `id_parents % 128`, for efficient per-halo lookups
        — see `LazySubhaloContext.get_items_by_halo`), and returns an
        eager `SubhaloContext` built from all processed batches.

        Parameters:
        -----------
        batch_size : int, optional
            Number of subhalos to process per chunk within each group
            file (default 10,000).

        Returns:
        --------
        SubhaloContext or LazySubhaloContext
            An eager SubhaloContext if freshly processed (or empty), or a
            LazySubhaloContext if an existing parquet cache was found.
        """
        logger.debug("Fetching subhalos from Magneticum in batches...")
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()

        path = self.catalog.path
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
        computed_columns = self.function_needs.get("subhalo", {})
        catalog_fields = [f for f in self.subhalo_fields if f not in computed_columns]
        table = "subhalo"
        subhalo_conditions = {
            col: data
            for col, data in self.catalog.fields.subhalo.items()
            if "filter" in data
        }

        processed_batches = []
        global_offset = 0

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
                    batch = {k: v[batch_start:batch_end] for k, v in raw.items()}

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
            df["parent_bucket"] = df["id_parents"] % 128
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

    # ── subhalo index (for particle → subhalo assignment) ─────────────────────

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
        """
        Stream star particles from the snapshot, assign each to a halo/subhalo, and write them to a bucketed parquet dataset.

        Resumable: buckets (partitions of `parent_bucket = halo_id % 128`)
        already present under `{catalog.path}/particles_parquet` are
        skipped, so a re-run only processes what's missing. For each
        snapshot file, reads star particles, assigns halo/subhalo
        ownership via `_assign_halo_ids`/`_assign_subhalo_ids`, drops
        unowned particles (`halo_ids == -1`) and particles already in a
        completed bucket, runs the result through the standard particle
        field pipeline, and appends to the partitioned parquet dataset.

        Returns:
        --------
        LazySubhaloParticles
            A lazy wrapper over the (now complete) parquet dataset.
        """
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()
        if not getattr(self, "_particle_cache_loaded", False):
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
        """
        Fetch particle data for a subhalo.

        Parameters:
        -----------
        subhalo_id : int
            Required keyword argument: the subhalo to fetch particles for.

        Returns:
        --------
        SubhaloParticles
            Particle data for the subhalo.

        Raises:
        -------
        ValueError
            If `subhalo_id` is not provided in kwargs.
        """
        subhalo_id = kwargs.get('subhalo_id', None)

        if subhalo_id is None:
            logger.error("subhalo_id must be provided in kwargs")
            raise ValueError("subhalo_id must be provided in kwargs")

        # logger.info(f"Fetching particle data for subhalo: {subhalo_id}")
        return self._get_p_subhalo(subhalo_id)

    def _get_p_subhalo(self, subhalo_id: int) -> SubhaloParticles:
        """
        Load (building the particle parquet cache first, if needed) and return one subhalo's particles.

        Parameters:
        -----------
        subhalo_id : int
            Subhalo to fetch particles for.

        Returns:
        --------
        SubhaloParticles
            Particle data for the subhalo, loaded via
            `_load_subhalo_particles`.
        """
        print('_get_p_subhalo')
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()
        if not getattr(self, "_particle_cache_loaded", False):
            self._load_particle_cache()

        path = self.catalog.path
        particles_parquet = os.path.join(path, "particles_parquet")
        print(particles_parquet)

        if not os.path.exists(particles_parquet):
            print('Store_particle call')
            self._store_particles()

        return self._load_subhalo_particles(subhalo_id, particles_parquet)

    @staticmethod
    def _column_to_numpy(series: pd.Series) -> np.ndarray:
        """
        Convert a parquet column to numpy, stacking per-particle vectors into 2-D.

        Parquet list columns (POS/VEL) read back through pandas as a 1-D object
        array of shape (N,), each element a length-3 array. Stack those into a
        proper (N, 3) array (dtype preserved) so downstream feature code, which
        assumes real 2-D arrays, doesn't fail on them. Other columns (e.g. MASS)
        are returned untouched.
        """
        arr = series.to_numpy()
        if arr.dtype == object and len(arr) > 0 and isinstance(arr[0], (list, np.ndarray)):
            return np.stack(arr)
        return arr

    def _load_subhalo_particles(self, subhalo_id: int, particles_parquet: str) -> SubhaloParticles:
        """
        Load particles for a specific subhalo directly from parquet.
        No PID reindexing needed — subhalo_id is already stored as a column.
        Uses parent_bucket = halo_id % 128 to avoid full scan.
        """
        # Resolve parent halo to get the correct bucket
        print('_load_subhalo_particles')
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
        print(df.columns)
        for col in df.columns:
            if col in ('parent_id', 'subhalo_id', 'parent_bucket', 'parent_bucket_data'):
                continue
            data[col] = self._column_to_numpy(df[col])
            print(f'{col}: {len(data[col])}')

        # Remove white spaces on the right
        data = {key.rstrip(): value for key, value in data.items()}
        for key, value in data.items():
            print(key, len(value))

        # TODO avoid hardcoding this => centralize
        return SubhaloParticles(data=data, column_map={'POS': 'p', 'MASS': 'm', 'VEL': 'v'})

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
        """
        Fetch particle data for a halo.

        Parameters:
        -----------
        halo : Halo
            Required keyword argument: the halo to fetch particles for.

        Returns:
        --------
        SubhaloParticles
            Particle data for the halo.

        Raises:
        -------
        ValueError
            If `halo` is not provided in kwargs.
        """
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
            data[col] = self._column_to_numpy(df[col])

        data = {key.rstrip(): value for key, value in data.items()}
        # Using part type 4
        return SubhaloParticles(data=data, column_map={'POS': 'p', 'MASS': 'm', 'VEL': 'v'})

    def _get_p_halo(self, halo) -> SubhaloParticles | LazySubhaloParticles:
        """
        Load (building the particle parquet cache first, if needed) and return one halo's particles.

        Parameters:
        -----------
        halo : Halo
            Halo to fetch particles for.

        Returns:
        --------
        SubhaloParticles
            Particle data for the halo, loaded via `_load_halo_particles`.
        """
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()
        if not getattr(self, "_particle_cache_loaded", False):
            self._load_particle_cache()

        path = self.catalog.path
        particles_parquet = os.path.join(path, "particles_parquet")

        if not os.path.exists(particles_parquet):
            self._store_particles()

        return self._load_halo_particles(halo.halo_id, particles_parquet)

    def _build_pid_file_boundaries(self) -> list:
        """
        No per-entity lazy PID readers exist to index (see _assign_halo_ids),
        so there is nothing to build here. Kept as a no-op because
        _load_catalog_cache() still calls it.
        """
        return []

    def _build_subhalo_pid_file_boundaries(self) -> list:
        """See _build_pid_file_boundaries."""
        return []

    def _build_halo_intervals(self) -> None:
        """No upfront index is built — ownership is streamed and
        matched per snapshot-file chunk inside _assign_halo_ids/
        _assign_subhalo_ids, to keep peak memory bounded by one group file's
        chunk rather than the total group-catalogue size."""
        pass

    def _build_subhalo_index(self) -> None:
        """See _build_halo_intervals."""
        pass

    def _build_subhalo_pid_arrays(self) -> None:
        """See _build_halo_intervals."""
        pass

    @staticmethod
    def _unwrap_int32_overflow(raw: np.ndarray) -> np.ndarray:
        """
        GOFF/SOFF are stored on disk as signed 4-byte integers (confirmed
        via the group catalogue's own INFO-table type declaration AND by
        directly measuring the block's on-disk byte length — this is a
        genuine on-disk width limit, not a g3read misread). For a large
        box, the true cumulative particle offset for later halos/subhalos
        exceeds the 4-byte signed limit (2**31 - 1) and wraps around to a
        negative number when g3read reads it back (verified in production:
        halo 139520's GOFF was still valid at 2147483622, and the very next
        halo's true GOFF, 2147486779, read back as -2147480517).

        This is recoverable, not lost: the whole catalogue's true offsets
        stay under 2**32, so each value wraps at most once, and a wrapped
        value read back as int64 always comes back as exactly
        `true_value - 2**32`. Undoing that is just adding 2**32 back
        whenever the value is negative — independently per element, with
        no dependency on row order or on a parent halo's own value (an
        earlier fix attempt computed a subhalo's offset relative to its
        parent halo's raw GOFF, which breaks whenever the 2**31 boundary
        falls in the middle of a halo's particle range: the halo's own
        GOFF can still be unwrapped while a subhalo sitting further into
        that same halo has already wrapped on its own).
        """
        raw = np.asarray(raw, dtype=np.int64)
        return np.where(raw < 0, raw + (1 << 32), raw)

    def _load_group_offsets(self) -> None:
        """
        Cheap, small-memory setup (O(n_halos) + O(n_subhalos), not O(N)):
        cumulative halo end-positions and subhalo start-positions sorted for
        binary search. Used by _owners_for_chunk to derive, for one group
        file's chunk at a time, which halo/subhalo owns each position in it
        — without ever materialising a per-position array across the WHOLE
        group catalogue.

        GOFF/SOFF are unwrapped (see _unwrap_int32_overflow) before
        anything else, since sorting or comparing the raw, possibly
        negative-from-overflow values would silently produce wrong results
        rather than an error.

        np.searchsorted (used by _owners_for_chunk) requires its input
        sorted by GOFF — GOFF is NOT guaranteed to already be in that order:
        multi-file group catalogues concatenate halos in on-disk/file order
        (see g3read.read_new(..., multiple_files=True)), which only matches
        ascending GOFF for a catalogue where per-file halo numbering lines
        up globally. That doesn't hold for every layout (e.g. a zoom/
        multi-resolution box, where MAGNETICUM_PART_TYPES's 'boundary'/
        'tracers' types come from), so halos are explicitly sorted here —
        exactly like subhalos already are below — rather than trusted as
        pre-sorted. _halo_id_sorted maps a sorted position back to the real
        halo_id (positional index into the original, un-sorted GOFF/GLEN/
        FSUB/NSUB arrays that _get_halos() etc. index by), the same way
        _sub_id_sorted already does for subhalos.
        """
        if getattr(self, "_group_offsets_loaded", False):
            return

        goff = self._unwrap_int32_overflow(self._goff)
        soff = self._unwrap_int32_overflow(self._soff)

        order_h = np.argsort(goff, kind="stable")  # tiny: n_halos-sized
        self._halo_goff = goff[order_h]
        self._halo_gend = self._halo_goff + np.asarray(self._glen, dtype=np.int64)[order_h]
        self._halo_id_sorted = order_h.astype(np.int32)

        order = np.argsort(soff, kind="stable")  # tiny: n_subhalos-sized
        self._sub_soff_sorted = soff[order]
        self._sub_send_sorted = self._sub_soff_sorted + np.asarray(self._slen, dtype=np.int64)[order]
        self._sub_id_sorted = order.astype(np.int32)

        self._group_offsets_loaded = True

    def _owners_for_chunk(self, chunk_start: int, chunk_end: int):
        """
        halo_id_per_position, subhalo_id_per_position for the global PID
        position range [chunk_start, chunk_end) — i.e. for exactly one group
        file's chunk. Memory scales with the number of halos/subhalos
        overlapping this chunk (found via binary search, cheap) plus the
        chunk's own length for the two output arrays — never with the total
        group-catalogue size.
        """
        n = chunk_end - chunk_start

        # halos: GOFF/GLEN partitions the whole catalogue with no gaps, so
        # every position in this chunk is covered by exactly one halo.
        # _halo_goff/_halo_gend are sorted by GOFF (see _load_group_offsets),
        # so lo/hi index into that sorted order — map back to real halo_ids
        # via _halo_id_sorted, the same way subhalos do below.
        lo = np.searchsorted(self._halo_gend, chunk_start, side='right')
        hi = np.searchsorted(self._halo_goff, chunk_end, side='left')
        halo_ids = self._halo_id_sorted[lo:hi]
        starts = np.clip(self._halo_goff[lo:hi] - chunk_start, 0, n)
        ends = np.clip(self._halo_gend[lo:hi] - chunk_start, 0, n)
        halo_id_per_position = np.repeat(halo_ids, ends - starts)
        if len(halo_id_per_position) != n:
            raise ValueError(
                f"halo ownership for chunk [{chunk_start},{chunk_end}) covers "
                f"{len(halo_id_per_position)} positions, expected {n}; the "
                f"GOFF/GLEN contiguous-layout assumption no longer holds."
            )

        # subhalos: SOFF/SLEN nest within their parent halo's range but may
        # leave gaps (diffuse, unbound halo particles) — vectorised
        # run-length expansion over just the overlapping subhalos, with -1
        # filling any gaps.
        lo_s = np.searchsorted(self._sub_send_sorted, chunk_start, side='right')
        hi_s = np.searchsorted(self._sub_soff_sorted, chunk_end, side='left')
        if hi_s > lo_s:
            s_ids = self._sub_id_sorted[lo_s:hi_s]
            s_starts = np.clip(self._sub_soff_sorted[lo_s:hi_s] - chunk_start, 0, n)
            s_ends = np.clip(self._sub_send_sorted[lo_s:hi_s] - chunk_start, 0, n)
            s_lengths = s_ends - s_starts

            gap_before = np.empty(len(s_starts), dtype=np.int64)
            gap_before[0] = s_starts[0]
            gap_before[1:] = s_starts[1:] - s_ends[:-1]
            gap_before = np.clip(gap_before, 0, None)

            ids_interleaved = np.empty(2 * len(s_ids), dtype=np.int32)
            lens_interleaved = np.empty(2 * len(s_ids), dtype=np.int64)
            ids_interleaved[0::2] = -1
            ids_interleaved[1::2] = s_ids
            lens_interleaved[0::2] = gap_before
            lens_interleaved[1::2] = s_lengths

            trailing_gap = n - int(s_ends[-1])
            if trailing_gap > 0:
                ids_interleaved = np.concatenate((ids_interleaved, [-1]))
                lens_interleaved = np.concatenate((lens_interleaved, [trailing_gap]))

            subhalo_id_per_position = np.repeat(ids_interleaved, lens_interleaved)
        else:
            subhalo_id_per_position = np.full(n, -1, dtype=np.int32)

        return halo_id_per_position, subhalo_id_per_position

    def _stream_match(self, ids: np.ndarray):
        """
        Bounded-memory halo_id/subhalo_id assignment for one batch of
        particle IDs (in practice: one snapshot file's worth of star
        particles). Streams the group catalogue's "PID " block one group
        file at a time — never holding more than one file's chunk plus the
        (much smaller) `ids` array in memory — matching each chunk against
        `ids` via a sorted search, and filling in owners only for positions
        that match.

        Returns (halo_ids, subhalo_ids), both shaped like `ids`, -1 where
        a particle ID has no match in the group catalogue at all (e.g. an
        unbound particle).
        """
        self._load_group_offsets()

        halo_result = np.full(len(ids), -1, dtype=np.int32)
        subhalo_result = np.full(len(ids), -1, dtype=np.int32)

        id_sorter = np.argsort(ids)  # small: bounded by this chunk's size
        ids_sorted = ids[id_sorter]

        cumulative_offset = 0
        for group_file in g3read.yield_all_files(self._group_base):
            f = g3read.GadgetFile(group_file, is_snap=False)
            pid_chunk = f.read("PID ", 2)
            if pid_chunk is None or len(pid_chunk) == 0:
                continue
            pid_chunk = np.asarray(pid_chunk, dtype=np.int64)
            chunk_start = cumulative_offset
            chunk_end = cumulative_offset + len(pid_chunk)
            cumulative_offset = chunk_end

            probe = np.searchsorted(ids_sorted, pid_chunk)
            probe = np.clip(probe, 0, len(ids_sorted) - 1)
            is_match = ids_sorted[probe] == pid_chunk
            if not np.any(is_match):
                del pid_chunk, probe, is_match
                continue

            halo_owner, sub_owner = self._owners_for_chunk(chunk_start, chunk_end)

            matched_ids_sorted_pos = probe[is_match]
            matched_original_pos = id_sorter[matched_ids_sorted_pos]
            halo_result[matched_original_pos] = halo_owner[is_match]
            subhalo_result[matched_original_pos] = sub_owner[is_match]

            del pid_chunk, probe, is_match, halo_owner, sub_owner

        return halo_result, subhalo_result

    def _assign_halo_ids(self, ids: np.ndarray) -> np.ndarray:
        """
        Exact-match halo assignment, streamed (see _stream_match) instead of
        looked up in a prebuilt global index. Caches its result (alongside
        the matching subhalo_ids) keyed by the identity of `ids`, since
        _store_particles() calls this and _assign_subhalo_ids back to back
        with the same array — avoids streaming the group catalogue twice per
        snapshot-file chunk.
        """
        if len(ids) == 0:
            return np.array([], dtype=np.int32)

        if getattr(self, "_stream_cache_key", None) != id(ids):
            halo_ids, subhalo_ids = self._stream_match(ids)
            self._stream_cache_key = id(ids)
            self._stream_cache_halo_ids = halo_ids
            self._stream_cache_subhalo_ids = subhalo_ids

        return self._stream_cache_halo_ids

    def _assign_subhalo_ids(self, ids: np.ndarray) -> np.ndarray:
        """
        Exact-match subhalo assignment, streamed the same way as
        _assign_halo_ids (and sharing its cache — see there).
        """
        if len(ids) == 0:
            return np.array([], dtype=np.int32)

        if getattr(self, "_stream_cache_key", None) != id(ids):
            halo_ids, subhalo_ids = self._stream_match(ids)
            self._stream_cache_key = id(ids)
            self._stream_cache_halo_ids = halo_ids
            self._stream_cache_subhalo_ids = subhalo_ids

        return self._stream_cache_subhalo_ids

    def verify_assignment_integrity(self, max_snap_files: int = 1, max_report_rows: int = 20) -> dict:
        """
        Smoke test for the real halo/subhalo -> star-particle assignment,
        run against one or a few real snapshot files. Companion to
        src/data_generation/test/manual_tests/diagnose_magneticum_pid_offsets.py
        (which validated the GOFF/SOFF offset arithmetic in isolation) —
        this instead runs the actual assignment path
        (_assign_halo_ids/_assign_subhalo_ids) end-to-end and checks its
        output, without needing any external ground truth:

        1. Consistency check: for every particle assigned BOTH a halo_id
           and a subhalo_id (neither -1), the subhalo's own recorded
           parent (GRNR) must equal the assigned halo_id. A mismatch is
           unambiguous proof of a real assignment bug — this needs no
           external data to know it's wrong, unlike e.g. checking whether
           a specific particle count "looks right".
        2. Bound check: a halo/subhalo can never be assigned MORE star
           particles than its own GLEN/SLEN (star particles are a strict
           subset of all particle types in a halo/subhalo).

        Prints a per-halo and per-subhalo (id, particle count) table, for
        eyeballing, similar in spirit to the diagnose script's tables.
        Meant to be run manually against real cluster data (e.g. from a
        notebook or a manual_tests script), not part of the automated
        test suite — see test_fetcher_magneticum_v2.py for the synthetic,
        fixture-based unit tests covering this same logic.

        Returns a dict summary (n_particles_checked, n_consistency_failures,
        n_halo_bound_failures, n_subhalo_bound_failures, ok) for scripted use.
        """
        if not getattr(self, "_cache_loaded", False):
            self._load_catalog_cache()
        if not getattr(self, "_particle_cache_loaded", False):
            self._load_particle_cache()

        star_part_type = next((pt for pt in self.particle_fields if pt in ("stars", "star")), None)
        if star_part_type is None:
            print("No star particle type configured on this catalogue — nothing to verify.")
            return {
                "n_particles_checked": 0, "n_consistency_failures": 0,
                "n_halo_bound_failures": 0, "n_subhalo_bound_failures": 0, "ok": True,
            }
        # Only "ID  " is actually used below — deliberately NOT reusing the
        # real ingestion config's full fields_padded (POS/VEL/MASS/...),
        # which would read particle data this method never touches and
        # needlessly exposes it to any field-naming mismatch in fields it
        # doesn't need.
        ptype_int = self._snap_cache_config[star_part_type]["ptype_int"]

        grnr = np.asarray(
            g3read.read_new(self._group_base, ["GRNR"], 1, is_snap=False, multiple_files=True)["GRNR"],
            dtype=np.int64,
        )
        glen = np.asarray(self._glen, dtype=np.int64)
        slen = np.asarray(self._slen, dtype=np.int64)

        halo_counts = np.zeros(len(glen), dtype=np.int64)
        subhalo_counts = np.zeros(len(slen), dtype=np.int64)
        n_particles_checked = 0
        n_consistency_failures = 0
        consistency_examples = []
        n_files_done = 0

        for snap_file in g3read.yield_all_files(self._snap_base):
            if n_files_done >= max_snap_files:
                break
            snap_data = g3read.read_new(
                snap_file, ["ID  "], ptype_int, multiple_files=False, do_flatten=False
            )[ptype_int]
            n_files_done += 1
            if snap_data is None or len(snap_data) == 0:
                continue

            ids = snap_data["ID  "]
            halo_ids = self._assign_halo_ids(ids)
            subhalo_ids = self._assign_subhalo_ids(ids)
            n_particles_checked += len(ids)

            both_assigned = (halo_ids != -1) & (subhalo_ids != -1)
            if np.any(both_assigned):
                expected_parent = grnr[subhalo_ids[both_assigned]]
                mismatch = expected_parent != halo_ids[both_assigned]
                n_consistency_failures += int(mismatch.sum())
                if mismatch.any() and len(consistency_examples) < 10:
                    bad_idx = np.nonzero(both_assigned)[0][mismatch][:10 - len(consistency_examples)]
                    for i in bad_idx:
                        consistency_examples.append((
                            int(ids[i]), int(halo_ids[i]), int(subhalo_ids[i]), int(grnr[subhalo_ids[i]]),
                        ))

            valid_h = halo_ids[halo_ids != -1]
            if len(valid_h) > 0:
                np.add.at(halo_counts, valid_h, 1)
            valid_s = subhalo_ids[subhalo_ids != -1]
            if len(valid_s) > 0:
                np.add.at(subhalo_counts, valid_s, 1)

        print(f"Checked {n_particles_checked} star particles across {n_files_done} snapshot file(s).")
        print()

        print(f"Consistency check (subhalo's GRNR must equal assigned halo_id): "
              f"{n_consistency_failures} mismatches")
        if consistency_examples:
            print("  examples (particle_id, assigned_halo_id, assigned_subhalo_id, GRNR[subhalo]):")
            for pid, h, s, g in consistency_examples:
                print(f"    id={pid}: halo_id={h}, subhalo_id={s}, GRNR[{s}]={g}")
        print()

        bound_failures_h = np.nonzero(halo_counts > glen)[0]
        bound_failures_s = np.nonzero(subhalo_counts > slen)[0]
        print(f"Bound check (assigned particles must not exceed GLEN/SLEN): "
              f"{len(bound_failures_h)} halos over, {len(bound_failures_s)} subhalos over")
        for h in bound_failures_h[:10]:
            print(f"    halo[{h}]: assigned={halo_counts[h]}, GLEN={glen[h]}")
        for s in bound_failures_s[:10]:
            print(f"    subhalo[{s}]: assigned={subhalo_counts[s]}, SLEN={slen[s]}")
        print()

        nonempty_h = np.nonzero(halo_counts > 0)[0]
        print(f"Per-halo particle counts (first {max_report_rows} non-empty halos):")
        print(f"{'halo_id':>10} | {'n_particles':>12} | {'GLEN':>10}")
        for h in nonempty_h[:max_report_rows]:
            print(f"{h:>10} | {halo_counts[h]:>12} | {glen[h]:>10}")
        print()

        nonempty_s = np.nonzero(subhalo_counts > 0)[0]
        print(f"Per-subhalo particle counts (first {max_report_rows} non-empty subhalos):")
        print(f"{'subhalo_id':>10} | {'parent(GRNR)':>12} | {'n_particles':>12} | {'SLEN':>10}")
        for s in nonempty_s[:max_report_rows]:
            print(f"{s:>10} | {grnr[s]:>12} | {subhalo_counts[s]:>12} | {slen[s]:>10}")

        ok = n_consistency_failures == 0 and len(bound_failures_h) == 0 and len(bound_failures_s) == 0
        print()
        print("PASS — assignment integrity checks all hold." if ok else "FAIL — see mismatches above.")

        return {
            "n_particles_checked": n_particles_checked,
            "n_consistency_failures": n_consistency_failures,
            "n_halo_bound_failures": len(bound_failures_h),
            "n_subhalo_bound_failures": len(bound_failures_s),
            "ok": ok,
        }


class ObeliskFetcher(DataFetcher):
    """
    Data fetcher for the Obelisk main catalogue (AdaptaHOP halo/galaxy
    properties, post-processed into one flat HDF5 table per snapshot —
    see src/data_generation/test/data/obelisk/README.md).

    Particle-level access (get_p_halo / get_p_subhalo) reads star particles
    from the per-galaxy `GAL_{iout:05d}/gal_stars_{ID:07d}` Fortran binary
    files that sit alongside the main catalogue HDF5 (see
    src/data_generation/test/data/obelisk/galstars.py for the reference
    reader this was ported from). There is no DM or gas particle data in
    Obelisk's particle dumps — only stars, unlike the DM+gas+stars
    catalogues the generic particles pipeline was originally built for.

    Halo/subhalo split: uses AdaptaHOP's `is_central` flag, not `level`.
    Confirmed with the Obelisk team (Maxime Trebitsch): `level > 1` only
    means an object sits within another structure's AdaptaHOP "radius", not
    that it is gravitationally bound to it or should be called a satellite
    — a level-2 structure can be outside its level-1 parent's virial radius
    entirely. `is_central` is the field that actually reflects this
    ("neither within their direct nor top-level parent"), and `host`/
    `hostsub` are the recommended way to identify parents. On real data,
    ~36% of `is_central` rows have `level > 1` (up to ~1.7e12 Msun), so the
    two fields disagree often, not just at the margins. `is_central`
    objects map to this pipeline's halos, `host` (ID of the TOP-level
    parent, already flattening arbitrary nesting depth) maps non-central
    objects to their halo — the same "halos and subhalos are the same
    table, filtered differently" shape FlamingoFetcher uses.

    Unlike Flamingo, there is no row-position vs true-catalogue-index split
    to reconcile: `ID`, `host` and `hostsub` are all expressed in the same
    namespace (the catalogue's own `ID` column), so `halo_id`/`subhalo_id`
    are taken directly from `ID` rather than a positional index, and
    `id_parents` is built directly from `host`/`ID` with no id-space
    conversion step — avoiding that entire class of bug by construction.
    """

    def __init__(self, catalog: DictConfig, **kwargs):
        """
        Initialize the fetcher and eagerly load the per-snapshot catalogue HDF5 file.

        Parameters:
        -----------
        catalog : DictConfig
            Catalogue configuration, with `path` and `snapnum` used by
            `load_files()` to locate the HDF5 file.
        **kwargs
            Forwarded to `DataFetcher.__init__`.
        """
        super().__init__(catalog, **kwargs)
        logger.info(f"ObeliskFetcher initialized with config: {catalog.path}")
        self._raw = None
        self.load_files()

    def load_files(self):
        """
        Load the per-snapshot Obelisk catalogue HDF5 file into a flat dict
        of 1D numpy arrays (one entry per column), exactly as the README's
        own usage example does. Kept on self as the single source both
        _get_halos() and _get_subhalos() filter/reshape from — never
        mutated in place (see _raw_copy()), so both can be called in any
        order without one call's cleanup affecting the other.
        """
        iout = int(self.catalog.snapnum)
        path = os.path.join(self.catalog.path, f"Obelisk_{iout:05d}_catalogue.h5")

        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        group_key = f"output_{iout:05d}"
        with h5py.File(path, "r") as hf:
            if group_key not in hf:
                raise Exception(f"Group '{group_key}' not found in {path}.")
            group = hf[group_key]
            self._raw = {k: group.get(k)[()] for k in group.keys()}

    def _raw_copy(self) -> dict:
        """Shallow copy: pipeline steps reassign/delete dict keys but don't
        mutate the underlying arrays in place, so this is enough to keep
        _get_halos()/_get_subhalos() from stepping on each other's raw data."""
        return dict(self._raw)

    def get_halos(self, *args, **kwargs) -> HaloContext:
        """Fetch halo (is_central == True) data from the Obelisk catalogue. See `_get_halos`."""
        return self._get_halos()

    def get_subhalos(self, *args, **kwargs) -> SubhaloContext:
        """Fetch subhalo (all objects, any level/is_central) data from the Obelisk catalogue. See `_get_subhalos`."""
        return self._get_subhalos()

    def get_p_subhalo(self, *args, **kwargs) -> SubhaloParticles:
        """
        Fetch star-particle data for a single subhalo/galaxy. See `_get_p_subhalo`.

        Parameters:
        -----------
        subhalo_id : int
            ID of the subhalo (catalogue `ID`) to fetch star particles for.

        Raises:
        -------
        ValueError
            If `subhalo_id` is not provided in kwargs.
        """
        subhalo_id = kwargs.get('subhalo_id', None)
        if subhalo_id is None:
            logger.error("subhalo_id must be provided in kwargs")
            raise ValueError("subhalo_id must be provided in kwargs")
        return self._get_p_subhalo(subhalo_id)

    def get_p_halo(self, *args, **kwargs) -> SubhaloParticles:
        """
        Fetch and concatenate star-particle data for every subhalo of a halo. See `_get_p_halo`.

        Parameters:
        -----------
        halo : Halo
            The halo to fetch particles for; its `subhalo_ids` (set by
            `_get_halos()`) are iterated over.

        Raises:
        -------
        ValueError
            If `halo` is not provided in kwargs.
        """
        halo = kwargs.get('halo', None)
        if halo is None:
            logger.error("halo must be provided in kwargs")
            raise ValueError("halo must be provided in kwargs")
        return self._get_p_halo(halo)

    def _inject_structural_field(self, fields_list: list, key: str, name: str):
        """Register a catalogue column that's used for structure (linkage,
        splitting) rather than declared as a science field in the YAML —
        same pattern as Illustris' GroupFirstSub / Flamingo's
        soap.host_halo_index dynamic injection."""
        if key not in fields_list:
            fields_list.append(key)
        OmegaConf.set_struct(self.catalog, False)
        table = 'halo' if fields_list is self.halo_fields else 'subhalo'
        if key not in self.catalog.fields[table]:
            self.catalog.fields[table][key] = {
                "input": None, "output": None, "scaling": None, "name": name
            }
        OmegaConf.set_struct(self.catalog, True)
        self.column_map[key] = name

    @staticmethod
    def _stack_position(d: dict, out_key: str = 'pos'):
        """
        Obelisk stores x/y/z as three separate 1D columns; every other
        fetcher's position field (Magneticum's GPOS, Flamingo's
        halo_centre) is already a single packed (N, 3) column by the time
        it reaches the generic per-field pipeline. Pack them the same way
        here — x/y/z themselves are then dropped by remove_useless_columns
        since only `out_key` is declared in the YAML.
        """
        d[out_key] = np.stack([d['x'], d['y'], d['z']], axis=1)
        return d

    def _gal_stars_path(self, subhalo_id: int) -> str:
        """
        Path to one galaxy's star-particle file: `GAL_{iout:05d}/gal_stars_{ID:07d}`,
        sibling to the main catalogue HDF5. The 7-digit zero-padded suffix is
        the galaxy's own catalogue `ID` — same namespace as halo_id/subhalo_id.
        """
        iout = int(self.catalog.snapnum)
        return os.path.join(self.catalog.path, f"GAL_{iout:05d}", f"gal_stars_{int(subhalo_id):07d}")

    def _read_gal_stars_file(self, subhalo_id: int) -> dict:
        """
        Read one galaxy's star-particle Fortran binary file into a flat
        dict of numpy arrays. Record layout ported from
        src/data_generation/test/data/obelisk/galstars.py's read_galfile
        (kept in test/data as reference documentation, not imported directly
        since it isn't a production module) — a fixed-format header record
        per attribute, followed by one record per per-particle array, in a
        fixed order.

        Only mass/position/velocity are declared as science fields in
        configs/catalog/obelisk.yaml today; `particle_id`/`birth_time`/
        `metallicity` are read (for fidelity with the file format, which
        requires reading every record in order) but not currently exposed
        as SubhaloParticles attributes — out of scope until needed.

        Returns:
        --------
        dict
            `{'position': (N, 3), 'velocity': (N, 3), 'mass': (N,),
            'particle_id': (N,), 'birth_time': (N,), 'metallicity': (N,)}`.

        Raises:
        -------
        Exception
            If the file does not exist, or its own header `id` doesn't
            match `subhalo_id` (wrong file, or wrong padding assumption).
        """
        path = self._gal_stars_path(subhalo_id)
        if not os.path.exists(path):
            logger.error(f"The path {path} does not exist.")
            raise Exception(f'The path {path} does not exist.')

        with FortranFile(path) as ff:
            # nparts and the per-particle id below are int64 ('q'), not
            # int32 ('i') as galstars.py's default (longint=False) assumes —
            # confirmed against real gal_stars files: reading nparts as
            # int32 splits its 8-byte record into 2 elements instead of 1
            # ("Expected a record of length 1, got 2"); as int64 it reads
            # as a single value that exactly matches the star-particle
            # array lengths that follow, and the file is consumed to
            # exactly its last byte.
            header = ff.read_attrs([
                ('id', 1, 'i'), ('level', 1, 'i'), ('mass', 1, 'd'),
                ('position', 3, 'd'), ('velocity', 3, 'd'), ('angular_momentum', 3, 'd'),
                ('nparts', 1, 'q'),
            ])
            if int(header['id']) != int(subhalo_id):
                raise Exception(
                    f"gal_stars file {path} has header id={header['id']}, expected {subhalo_id}."
                )

            position_x = ff.read_vector('d')
            position_y = ff.read_vector('d')
            position_z = ff.read_vector('d')
            velocity_x = ff.read_vector('d')
            velocity_y = ff.read_vector('d')
            velocity_z = ff.read_vector('d')
            mass = ff.read_vector('d')
            particle_id = ff.read_vector('q')
            birth_time = ff.read_vector('d')
            metallicity = ff.read_vector('d')

        return {
            'position': np.stack([position_x, position_y, position_z], axis=1),
            'velocity': np.stack([velocity_x, velocity_y, velocity_z], axis=1),
            'mass': mass,
            'particle_id': particle_id,
            'birth_time': birth_time,
            'metallicity': metallicity,
        }

    def _get_p_subhalo(self, subhalo_id: int) -> SubhaloParticles:
        """
        Fetch star-particle data for one galaxy (see `_read_gal_stars_file`).

        Obelisk has only one particle type (stars), unlike the DM+gas+stars
        catalogues the generic particles pipeline was built for, but
        `self.particle_fields` is still the expected set of top-level
        particle-type names (here `{'stars'}`, from `fields.particles.stars`
        in the YAML) so the same nested validate_and_process_fields /
        apply_unit_conversion / apply_filters machinery Illustris300Fetcher
        uses applies unchanged.

        Returns:
        --------
        SubhaloParticles
            Star-particle data for this galaxy (`star_m`/`star_p`/`star_v`,
            per fields.particles.stars' `name` entries).
        """
        logger.debug(f"Fetching star particles for subhalo: {subhalo_id}")
        data = {'stars': self._read_gal_stars_file(subhalo_id)}

        particle_conditions = {}
        for col, type_config in self.catalog.fields.particles.items():
            for t in type_config:
                if 'filter' in type_config[t]:
                    if col not in particle_conditions:
                        particle_conditions[col] = {}
                    particle_conditions[col][t] = type_config[t]

        table = 'particles'
        steps = [
            lambda particles: validate_and_process_fields(particles, self.particle_fields, table, self.catalog),
            lambda particles: convert_to_dataframe(particles),
            lambda particles: apply_unit_conversion(particles, self.particle_fields, table, self.catalog),
            lambda particles: apply_filters(particles, particle_conditions, self.catalog.fields.particles),
        ]
        processed_particles = run_pipeline(data, steps)
        particle_data = SubhaloParticles(processed_particles, self.catalog.fields.particles)
        logger.debug(f"Star particles fetched successfully for subhalo {subhalo_id}.")
        return particle_data

    def _get_p_halo(self, halo: Halo) -> SubhaloParticles:
        """
        Fetch and concatenate star-particle data for every subhalo of a halo.

        Mirrors Illustris300Fetcher._get_p_halo: loops `halo.subhalo_ids`
        (populated by `_get_halos()`, from the same id_parents linkage
        `_get_subhalos()` computes) and merges each galaxy's particles via
        `SubhaloParticles.__add__`.

        Returns:
        --------
        SubhaloParticles
            Combined star-particle data for all of the halo's subhalos,
            with `overlapping_particles_id` reset to None. `None` if the
            halo has no subhalos at all (should not normally happen, since
            every halo is its own central subhalo).
        """
        logger.debug(f"Fetching particle data for halo: {halo.halo_id}")
        subhalo_ids = halo.subhalo_ids
        particles_halo = None
        for subhalo_id in subhalo_ids:
            particles = self._get_p_subhalo(subhalo_id)
            if particles_halo is None:
                particles_halo = particles
            else:
                particles_halo += particles

        if particles_halo is not None:
            particles_halo.overlapping_particles_id = None
        logger.debug(f"Particle data for halo {halo.halo_id}: {particles_halo}")
        return particles_halo

    def _get_halos(self) -> HaloContext:
        """
        Fetch all central (`is_central == True`) objects from the catalogue as halos.

        Uses `is_central`, not `level == 1` — confirmed with the Obelisk
        team that `level` reflects AdaptaHOP's spatial nesting radius, not
        gravitational boundedness, so a `level > 1` object can still be a
        genuine standalone halo (`is_central == True`). See the class
        docstring for the full explanation and the real-data discrepancy
        this caused under the old `level`-based logic.

        Also computes each halo's `subhalo_ids` — the raw catalogue `ID`
        values of every subhalo/galaxy (any level) whose `id_parents`
        equals this halo's ID, per `_get_subhalos()`'s own linkage — so
        `get_p_halo()` can gather star particles across a halo's full
        substructure, the same convention Illustris300Fetcher uses.

        Returns:
        --------
        HaloContext
            Halos passing the configured filters, with `halo_id` taken
            directly from the catalogue's own `ID` column.
        """
        logger.debug("Fetching halos from Obelisk...")

        self._inject_structural_field(self.halo_fields, 'ID', 'halo_id')

        # _get_subhalos() is needed up front to populate subhalo_ids below;
        # it also re-asserts self.column_map['pos'] = 'spos' and
        # ['ID'] = 'subhalo_id' (both shared raw keys with our own
        # 'pos' -> 'gpos' / 'ID' -> 'halo_id'), so re-assert both again
        # right after, before this method's own pipeline runs.
        subhalo_context = self._get_subhalos()
        self.column_map['pos'] = 'gpos'
        self.column_map['ID'] = 'halo_id'

        raw = self._raw_copy()
        central_mask = raw['is_central']
        halos = {k: v[central_mask] for k, v in raw.items()}
        halos = self._stack_position(halos)

        table = 'halo'
        halo_conditions = {col: data for col, data in self.catalog.fields.halo.items() if 'filter' in data}
        steps = [
            lambda h: remove_useless_columns(h, self.halo_fields),
            lambda h: assign_id(h, 'fsub'),  # no AdaptaHOP equivalent — placeholder, same as FlamingoFetcher
            lambda h: validate_and_process_fields(h, self.halo_fields, table, self.catalog),
            lambda h: convert_to_dataframe(h),
            lambda h: apply_unit_conversion(h, self.halo_fields, table, self.catalog),
            lambda h: apply_filters(h, halo_conditions, self.catalog.fields.halo),
        ]
        processed_halos = run_pipeline(halos, steps).copy()
        # subhalo_ids: raw catalogue IDs of every subhalo (any level)
        # belonging to this halo — the same set get_subhalos() assigns
        # id_parents == ID to. 'ID' (not yet 'halo_id') since column_map
        # renaming only happens inside HaloContext.__init__, below.
        processed_halos['subhalo_ids'] = processed_halos['ID'].apply(
            lambda hid: subhalo_context.data.loc[subhalo_context.data['id_parents'] == hid, 'subhalo_id'].tolist()
        )
        self.column_map['subhalo_ids'] = 'subhalo_ids'
        halo_context = HaloContext(processed_halos, self.column_map)
        logger.debug(f"Halo data fetched successfully: {halo_context.data['halo_id'].nunique()} halos.")
        return halo_context

    def _get_subhalos(self) -> SubhaloContext:
        """
        Fetch every object in the catalogue (any `level`, any `is_central`) as subhalos.

        Returns:
        --------
        SubhaloContext
            Subhalos passing the configured filters, with `subhalo_id`
            taken from the catalogue's own `ID` column and `id_parents`
            set to `ID` (for `is_central == True` rows) or `host` (otherwise).
        """
        logger.debug("Fetching subhalos from Obelisk...")

        self._inject_structural_field(self.subhalo_fields, 'ID', 'subhalo_id')
        self._inject_structural_field(self.subhalo_fields, 'id_parents_computed', 'id_parents')
        self.column_map['pos'] = 'spos'  # see _get_halos() for why this must be re-asserted here

        raw = self._raw_copy()
        # id_parents: self ID for central objects, `host` (already in
        # ID-space, already flattened to the top-level parent) otherwise —
        # mirrors FlamingoFetcher.__reindexing_satellites' role, but Obelisk
        # needs no row-position/true-id conversion since host/ID share a
        # namespace already. Uses `is_central`, not `level == 1` — see
        # class docstring / Obelisk team guidance.
        raw['id_parents_computed'] = np.where(raw['is_central'], raw['ID'], raw['host'])
        raw = self._stack_position(raw)

        table = 'subhalo'
        subhalo_conditions = {col: data for col, data in self.catalog.fields.subhalo.items() if 'filter' in data}
        steps = [
            lambda s: remove_useless_columns(s, self.subhalo_fields),
            lambda s: validate_and_process_fields(s, self.subhalo_fields, table, self.catalog),
            lambda s: convert_to_dataframe(s),
            lambda s: apply_unit_conversion(s, self.subhalo_fields, table, self.catalog),
            lambda s: apply_filters(s, subhalo_conditions, self.catalog.fields.subhalo),
        ]
        processed_subhalos = run_pipeline(raw, steps)
        subhalo_context = SubhaloContext(processed_subhalos, self.column_map)
        logger.debug(f"Subhalo data fetched successfully: {len(subhalo_context.data)} subhalos.")
        return subhalo_context




