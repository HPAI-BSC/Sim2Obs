import logging
from collections import Counter

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_generation.src.halo import Halo, HaloContext
from src.data_generation.src.utils import rotate, dict_depth, setup_logger

logger = setup_logger("Subhalo-Particles")


class SubHalo:
    """
    A class to represent a SubHalo object with dynamically set attributes.

    Attributes:
    -----------
    subhalo_id : int
        The unique identifier for the subhalo.
    id_parents : int
        The ID of the parent halo to which the subhalo belongs.
    other attributes : various
        Any additional attributes passed during initialization.

    Methods:
    --------
    __init__(**kwargs):
        Initializes a SubHalo object by dynamically setting attributes from keyword arguments.
    """
    def __init__(self, **kwargs):
        """
        Initialize a SubHalo object with required attributes (`subhalo_id`, `id_parents`) and any other attributes provided.

        Parameters:
        -----------
        kwargs : dict
            A dictionary of attributes to be assigned to the SubHalo object.
            The required attributes 'subhalo_id' and 'id_parents' must be included.

        Raises:
        -------
        ValueError:
            If any of the required attributes 'subhalo_id' or 'id_parents' are missing.
        """
        required_attributes = ['subhalo_id', 'id_parents']

        missing_attrs = [attr for attr in required_attributes if attr not in kwargs]
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {missing_attrs}")

        # Dynamically set attributes
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = list(value.values())[0]

            if isinstance(value, list):  # Check if the value is a list
                # Check dimensions
                value = np.array(value)  # Convert the list to a numpy array
            setattr(self, key, value)


class SubhaloContext:
    """
    A class to manage subhalo data in a context, stored in a Pandas DataFrame with properties such as positions, masses, and radii.

    Attributes:
    -----------
    data : pd.DataFrame
        A DataFrame storing properties for different subhalos, with multi-index `id_parents` and `subhalo_id`.

    Methods:
    --------
    __init__(properties, scale_factor, column_map):
        Initializes the SubhaloContext with the provided subhalo properties, scaling them appropriately.
    get_item(subhalo_id, halo_id=None):
        Retrieves the properties of a specific subhalo by its ID and optional halo ID.
    get_items_by_halo(halo_id):
        Retrieves all subhalos associated with a specific halo ID.
    id_parents:
        Property to retrieve the `id_parents` column from the stored DataFrame.
    subhalo_id:
        Property to retrieve the `subhalo_id` column from the stored DataFrame.
    """
    def __init__(self, df: pd.DataFrame, column_map) -> None:
        """
        Initialize the SubhaloContext with properties of subhalos.

        Parameters:
        -----------
        properties : dict or pd.DataFrame
            Dictionary or DataFrame containing the subhalo properties.
        scale_factor : float
            The scaling factor to be applied to specific properties like positions and radii.
        column_map : dict
            Dictionary defining the mapping of expected fields to the keys/columns in `properties`.
        """
        if column_map is None:
            self.data = df

        else:
            # Create a DataFrame and set index
            df = df.rename(columns=column_map)

            # Count occurrences of each column
            column_counts = Counter(df.columns)
            duplicates = {col: count for col, count in column_counts.items() if count > 1}

            # Join columns with the same name if any
            for duplicated in duplicates:
                tmp = df[duplicated].to_numpy()
                df.drop(columns=[duplicated], inplace=True)
                df[duplicated] = list(tmp)

            self.data = df

            for column in df.columns:
                if column not in column_map.values():
                    if column not in ['subhalo_id', 'id_parents']:
                        self.data.drop(columns=[column], inplace=True)

            # Set a multi-index for efficient querying
            self.data.set_index(["id_parents", "subhalo_id"], inplace=True, drop=False)

    def preprocess(self, halo: Halo, lbox: float = None):
        """
        ...
        Parameters:
        -----------
        halo_context : HaloContext
            A HaloContext object providing necessary information for preprocessing.
        """
        def apply_pbc_correction(subhalo_pos, halo_center, lbox):
            delta = subhalo_pos - halo_center  # Compute the relative position

            if lbox is not None:
                # Create a copy to avoid modifying original delta directly
                corrected_delta = delta.copy()

                # Prepare the mask, True => coordinate needs to be updated
                # Particles that are more than half the box length in the positive direction
                over_half = corrected_delta > (lbox / 2)
                # Particles that are more than half the box length in the negative direction
                under_half = corrected_delta < (-lbox / 2)

                # Apply correction only to the out-of-bound coordinates
                corrected_delta[over_half] -= lbox
                corrected_delta[under_half] += lbox

                # Count how many corrections per axis (x, y, z)
                num_corrected_per_axis = np.sum(over_half | under_half, axis=0)
                if np.sum(num_corrected_per_axis) > 0:
                    logger.debug(f"Subhalos corrected due to PBC (x/y/z): {num_corrected_per_axis} / {len(delta)}")
                return corrected_delta
            return delta

        # TODO Add missing columns
        indexing = self.data['id_parents'] == halo.halo_id
        is_full_false = np.sum(indexing)
        if is_full_false == 0:
            logger.warning(f'No subhalos found for: {halo.halo_id}')
        if hasattr(halo, 'group_number'):
            indexing &= self.data['group_number'] == halo.group_number
        if 'spos' in self.data:
            # Normalize subhalo positions respect to the center of its halo
            self.data = self.data.copy()
            self.data.loc[indexing, 'spos'] = self.data.loc[indexing, 'spos'].apply(
                lambda row: apply_pbc_correction(row, halo.gpos, lbox)
            )
            #self.data.loc[indexing, 'spos'] = self.data.loc[indexing, 'spos'].apply(lambda row: row - halo.gpos)
        # ...

    def rotate(self, z, degree):
        """
        Rotates the subhalo positions based on the provided parameters.

        Parameters:
        -----------
        z : float
            Z-axis parameter for rotation.
        axis : np.ndarray
            Axis of rotation.
        degree : float
            Angle of rotation in degrees.
        """
        if 'spos' in self.data.columns:
            spos = np.array(self.data['spos'].tolist())
            subhalos_spos = rotate(z, degree, spos)
            self.data['spos'] = list([subhalos_spos[i] for i in range(subhalos_spos.shape[0])])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """
        Creates an instance of SubhaloContext from a DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the subhalo data.

        Returns:
        --------
        SubhaloContext
            A SubhaloContext object initialized from the provided DataFrame.
        """
        return SubhaloContext(df, None)

    def __getitem__(self, index):
        """
        Retrieves a sliced portion of the data as a new SubhaloContext.

        Parameters:
        -----------
        index : slice or list
            Indexing parameters to slice the data.

        Returns:
        --------
        SubhaloContext
            A new SubhaloContext object containing the sliced data.
        """
        return SubhaloContext.from_dataframe(self.data[index])  # Return a new instance with sliced data

    def get_subhalo_by_id(self, subhalo_id, halo_id=None) -> SubHalo:
        """
        Retrieve a subhalo's properties given its subhalo ID and optional halo ID.

        Parameters:
        -----------
        subhalo_id : int
            The subhalo ID to query.
        halo_id : int, optional
            The halo ID to query. If not provided, all subhalos with the specified `subhalo_id` are returned.

        Returns:
        --------
        SubHalo
            A SubHalo object containing the properties of the specified subhalo.

        Raises:
        -------
        Exception:
            If the provided `subhalo_id` (and `halo_id` if given) is not found in the DataFrame.
        """
        try:
            # Query using the multi-index
            if halo_id is not None:
                if 'group_number' in self.data:
                    # halo id here is the group number - Eagle case
                    result = self.data.loc[(self.data['group_number'] == halo_id) &
                                           (self.data['subhalo_id'] == subhalo_id)].to_dict()
                    return SubHalo(**result)
                elif 'id_parents' in self.data:
                    result = self.data.loc[(self.data['id_parents'] == halo_id) &
                                           (self.data['subhalo_id'] == subhalo_id)].to_dict()
                    return SubHalo(**result)
            else:
                # For Illustris-like single-index lookup, query all matching rows
                result = self.data.xs(subhalo_id, level="subhalo_id", drop_level=False).to_dict()
            return SubHalo(**result)
        except KeyError:
            raise Exception(f"Subhalo not found for halo_id={halo_id}, subhalo_id={subhalo_id}.")

    def get_items_by_halo(self, halo_id):
        """
        Retrieve all subhalos associated with a specific halo ID.

        Parameters:
        -----------
        halo_id : int
            The halo ID to query.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing all subhalos for the given `halo_id`.

        Raises:
        -------
        Exception:
            If no subhalos are found for the given `halo_id`.
        """
        try:
            df = self.data.loc[halo_id]
            return SubhaloContext.from_dataframe(df)
        except KeyError:
            raise Exception(f"No subhalos found for halo_id={halo_id}, ignoring halo.")

    def remove_subhalos_belonging_unvalid_halos(self, halo_context: HaloContext):
        halo_ids = halo_context.data['halo_id'].unique()
        logger.debug(f"Number of subhalos before filtering: {len(self.data)}")
        self.data = self.data[self.data['id_parents'].isin(halo_ids)]
        logger.debug(f"Number of subhalos after filtering: {len(self.data)}")

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        data = {}
        for key in self.__dict__:
            self_val = getattr(self, key)
            other_val = getattr(other, key)

            if isinstance(self_val, pd.DataFrame) and isinstance(other_val, pd.DataFrame):
                data[key] = pd.concat([self_val, other_val], axis=0, ignore_index=True)
            else:
                raise TypeError(f"Cannot concatenate attribute '{key}' of type {type(self_val)}")

        return self.__class__(df=data['data'], column_map=None)

    # Properties for direct access to columns
    @property
    def id_parents(self) -> pd.Series:
        """
        Retrieve the `id_parents` column from the DataFrame.

        Returns:
        --------
        pd.Series
            A Series containing the parent halo IDs.
        """
        return self.data["id_parents"].tolist()

    @property
    def subhalo_id(self) -> pd.Series:
        """
        Retrieve the `subhalo_id` column from the DataFrame.

        Returns:
        --------
        pd.Series
            A Series containing the subhalo IDs.
        """
        return self.data["subhalo_id"].tolist()

    def __len__(self):
        """
        Return the length of the DataFrame.

        Returns:
        --------
        int
            Length of the dataframe data.
        """
        return len(self.data)


def convert_to_dict(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")

    if isinstance(obj, dict):
        return {key: convert_to_dict(value) for key, value in obj.items()}

    # Keep numpy arrays and lists unchanged
    return obj


class SubhaloParticles:
    """
    A class to manage the particles of a subhalo, including their mass, position, and velocity data.

    Attributes:
    -----------
    m : np.ndarray
        The mass of particles.
    p : np.ndarray
        The positions of particles.
    v : np.ndarray
        The velocities of particles.

    Methods:
    --------
    __init__(m, p, v, scale_factor=None, init=True):
        Initializes the SubhaloParticles object with particle mass, position, and velocity data.
    """
    def __init__(self, data: dict, column_map: DictConfig) -> None:
        """
        Initialize the SubhaloParticles object with particle mass, position, and velocity data.

        Parameters:
        -----------
        data : dict
            Dictionary containing particle data for different particle types.
        column_map : DictConfig
            Mapping of column names for each particle type.
        """
        self.overlapping_particles_id = None

        # TODO complete
        fix_names = {'stars': 'star', 'dark_matter': 'dm'}
        print(f'column_map: {column_map}')

        data = convert_to_dict(data)

        if column_map is None:
            if dict_depth(data) == 1:
                for var in data:
                    value = data[var]
                    if isinstance(value, pd.Series):
                        value = value.to_numpy()
                    elif isinstance(value, list):
                        value = np.array(value)
                    setattr(self, var, value)
            elif dict_depth(data) == 2:
                for part_type in data:
                    for var in data[part_type]:
                        value = data[part_type][var]
                        if isinstance(value, pd.Series):
                            value = value.to_numpy()
                        elif isinstance(value, list):
                            value = np.array(value)
                        setattr(self, f'{fix_names.get(part_type, part_type)}_{var}', value)
        else:
            try:
                particle_type = 'star'
                if dict_depth(data) == 1:
                    for var in column_map.keys():
                        value = data[var]
                        if isinstance(value, pd.DataFrame):
                            value = value.to_dict()
                        if isinstance(value, (pd.Series, pd.Series)):
                            print('series')
                            value = np.stack(value.to_list())
                        if isinstance(value, list):
                            value = np.array(value)

                        name = column_map[var]
                        # Add attr with that provided name
                        setattr(self, f'{particle_type}_{name}', value)
                elif dict_depth(data) == 2:
                    for part_type in data:
                        for var in data[part_type]:
                            if var in column_map[part_type]:
                                value = data[part_type][var]

                                if isinstance(value, (pd.Series, pd.Series)):
                                    value = np.stack(value.to_list())
                                elif isinstance(value, list):
                                    value = np.array(value)

                                if hasattr(column_map[part_type][var], "name"):
                                    name = column_map[part_type][var].name
                                else:
                                    name = column_map[part_type][var]

                                # Add attr with that provided name
                                attr_name = f'{fix_names.get(part_type, part_type)}_{name}'
                                setattr(self, attr_name, value)
            except Exception as e:
                print(f'need at least one array to stack: {e}')

    @property
    def non_overlapping(self):
        return self._non_overlapping_for()

    def _non_overlapping_for(self, particle_type: str = None):
        """
        Return the particles that belong to the original (non-overlapping) subhalo.

        `self.overlapping_particles_id` is the index (int) marking where the
        original subhalo's particles end and the appended/overlapping particles
        begin. Every particle array is sliced as [0:boundary].

        Parameters
        ----------
        particle_type : str, optional
            Unused now that the boundary applies globally across all particle
            arrays rather than per particle type. Kept for backward compatibility.
        """
        if self.overlapping_particles_id is None or self.overlapping_particles_id == 0:
            logger.info('There are no overlapping particles, returning self.')
            return self

        boundary = self.overlapping_particles_id

        data = {}
        for key in self.__dict__:
            self_val = getattr(self, key)
            if isinstance(self_val, np.ndarray):
                data[key] = self_val[:boundary]
            else:
                data[key] = self_val

        return self.__class__(data=data, column_map=None)

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented

        data = {}

        # Capture the boundary index once, before any concatenation
        for key in self.__dict__:
            self_val = getattr(self, key)
            if isinstance(self_val, np.ndarray) and self.overlapping_particles_id is None:
                self.overlapping_particles_id = len(self_val)
                break

        for key in self.__dict__:
            try:
                self_val = getattr(self, key)
                if hasattr(other, key):
                    other_val = getattr(other, key)

                    if isinstance(self_val, np.ndarray) and isinstance(other_val, np.ndarray):
                        if self.overlapping_particles_id is None:
                            self.overlapping_particles_id = len(self_val)
                        data[key] = np.concatenate((self_val, other_val), axis=0)
                    elif key == 'overlapping_particles_id':
                        continue
                    else:
                        # Decide what to do with non-ndarray attributes here
                        raise TypeError(f"Cannot concatenate attribute '{key}' of type {type(self_val)}")
                else:
                    data[key] = self_val
            except Exception as e:
                print(e)
                continue

        result = self.__class__(data=data, column_map=None)
        result.overlapping_particles_id = self.overlapping_particles_id
        return result

    def preprocess(self, halo: Halo, lbox: float = None):
        """
        Preprocesses the particle data by normalizing star particle positions with respect to the halo center.
        If lbox is provided, applies periodic boundary condition (PBC) correction using the minimum image convention.

        Parameters
        ----------
        halo : Halo
            The Halo object providing the central position for normalization.
        lbox : float, optional
            Size of the simulation box (in kpc/h). If provided, PBC correction is applied.
        """
        # Normalize positions respect to the center of its halo
        print(len(self.star_p))
        if hasattr(self, 'star_p') and self.star_p is not None:
            delta = self.star_p - halo.gpos

            # Calculate the distance from the particle to the halo cente
            if lbox is not None:
                # Create a copy to avoid modifying original delta directly
                corrected_delta = delta.copy()

                # Prepare the mask, True => coordinate needs to be updated
                # Particles that are more than half the box length in the positive direction
                over_half = corrected_delta > (lbox / 2)
                # Particles that are more than half the box length in the negative direction
                under_half = corrected_delta < (-lbox / 2)

                # Apply correction only to the out-of-bound coordinates
                corrected_delta[over_half] -= lbox
                corrected_delta[under_half] += lbox

                # Count how many corrections per axis (x, y, z)
                num_corrected_per_axis = np.sum(over_half | under_half, axis=0)
                if np.sum(num_corrected_per_axis) > 0:
                    logger.debug(f"Particles corrected due to PBC (x/y/z): {num_corrected_per_axis} / {len(delta)}, halo: {halo.halo_id}")
                delta = corrected_delta

        self.star_p = delta

        # ...

    def rotate(self, z, degree):
        """
        Rotates the particle positions based on the provided parameters.

        Parameters:
        -----------
        z : float
            Z-axis parameter for rotation.
        axis : np.ndarray
            Axis of rotation.
        degree : float
            Angle of rotation in degrees.
        """
        if hasattr(self, 'star_p'):
            self.star_p = rotate(z, degree, self.star_p)

    def __repr__(self):
        lines = []
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, np.ndarray):
                lines.append(f"{attr_name}: shape={attr_value.shape}")
        return " ".join(lines)

    def __len__(self):
        for key in self.__dict__:
            self_val = getattr(self, key)
            if isinstance(self_val, np.ndarray):
                return len(self_val)

    @property
    def is_empty(self):
        if self.__dict__:
            return False
        return True







