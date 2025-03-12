from collections import Counter

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_generation.src.halo import HaloContext, Halo
from src.data_generation.src.utils import rotate, dict_depth


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

    def preprocess(self, halo_context: HaloContext):
        # Add missing columns
        for id_halo in halo_context.halo_id:
            halo = halo_context.get_halo_by_id(id_halo)
            indexing = self.data['id_parents'] == id_halo
            if hasattr(halo, 'group_number'):
                indexing &= self.data['group_number'] == halo.group_number
            if 'spos' in self.data:
                # Normalize subhalo positions respect to the center of its halo
                self.data.loc[indexing, 'spos'] = self.data.loc[indexing, 'spos'].apply(lambda row: row - halo.gpos)
            # ...

    def rotate(self, z, axis, degree):
        if 'spos' in self.data.columns:
            spos = np.array(self.data['spos'].tolist())
            subhalos_spos = rotate(z, axis, degree, spos)
            self.data['spos'] = list([subhalos_spos[i] for i in range(subhalos_spos.shape[0])])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return SubhaloContext(df, None)

    def __getitem__(self, index):
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
                result = self.data.loc[(halo_id, subhalo_id)].to_dict()
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
            raise Exception(f"No subhalos found for halo_id={halo_id}.")

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
        Initialize the SubhaloParticles with mass, position, and velocity data.

        Parameters:
        -----------
        ..
        """
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
                        setattr(self, f'{part_type}_{var}', value)
        else:
            for part_type in data:
                for var in data[part_type]:
                    if var in column_map[part_type]:
                        value = data[part_type][var]
                        if isinstance(value, (pd.Series, pd.Series)):
                            value = np.stack(value.to_list())
                        elif isinstance(value, list):
                            value = np.array(value)
                        name = column_map[part_type][var].name
                        # Add attr with that provided name
                        setattr(self, f'{part_type}_{name}', value)

    def preprocess(self, halo: Halo):
        # Normalize positions respect to the center of its halo
        if hasattr(self, 'star_p'):
            self.star_p = self.star_p - halo.gpos
        # ...

    def rotate(self, z, axis, degree):
        if hasattr(self, 'star_p'):
            self.star_p = rotate(z, axis, degree, self.star_p)





