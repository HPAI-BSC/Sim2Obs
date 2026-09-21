from collections import Counter

import numpy as np
import pandas as pd


class Halo:
    """
    A class to represent a Halo object with dynamically set attributes.

    Attributes:
    -----------
    halo_id : int
        The unique identifier for the halo.
    fsub : int
        The index of the primary subhalo.
    other attributes : various
        Any additional attributes passed during initialization.

    Methods:
    --------
    ``__init__(**kwargs)``
        Initializes a Halo object by dynamically setting attributes from keyword arguments.
    """
    def __init__(self, **kwargs):
        """
        Initialize a Halo object with required attributes (`halo_id`, `fsub`) and any other attributes provided.

        Parameters:
        -----------
        kwargs : dict
            A dictionary of attributes to be assigned to the Halo object.
            The required attributes 'halo_id' and 'fsub' must be included.

        Raises:
        -------
        ValueError:
            If any of the required attributes 'halo_id' or 'fsub' are missing.
        """
        required_attributes = ['halo_id', 'fsub']
        missing_attrs = [attr for attr in required_attributes if attr not in kwargs]
        if missing_attrs:
            raise ValueError(f"Missing required attributes: {missing_attrs}")

        # Dynamically set attributes
        for key, value in kwargs.items():
            if isinstance(value, list):  # Check if the value is a list
                value = np.array(value)  # Convert the list to a numpy array
            setattr(self, key, value)

    @classmethod
    def empty_instance(cls):
        """Return a placeholder Halo with `halo_id` and `fsub` set to None, for call sites that need a Halo-shaped stand-in."""
        return cls(halo_id=None, fsub=None)  # You can adjust these default values as needed

    def add_children(self, subhalo_context):
        """
        Populate/extend `self.subhalos_ids` with the IDs of subhalos belonging to this halo.

        Looks up subhalos whose `id_parents` equals `self.halo_id` in
        `subhalo_context` (a SubhaloContext, or a plain DataFrame already
        filtered to this halo's subhalos). If a `group_number` column is
        present (Eagle-style catalogues), each entry is a
        `(group_number, subhalo_id)` tuple instead of a bare subhalo_id, to
        disambiguate subhalo IDs that are only unique within a group.

        Parameters:
        -----------
        subhalo_context : SubhaloContext or pd.DataFrame
            Subhalo data to search for this halo's children. A
            SubhaloContext is queried via `.data['id_parents']`; a
            DataFrame is assumed to already contain only this halo's
            subhalos.
        """
        subhalo_ids = []
        if hasattr(subhalo_context, 'data'):
            filtered_subhalo_context = subhalo_context.data[subhalo_context.data['id_parents'] == self.halo_id]
            if 'group_number' in subhalo_context.data:
                group_number = filtered_subhalo_context['group_number']
                subhalo_id = filtered_subhalo_context['subhalo_id']
                subhalo_ids = list(zip(group_number, subhalo_id))
            else:
                subhalo_ids = filtered_subhalo_context['subhalo_id'].tolist()
        elif isinstance(subhalo_context, pd.DataFrame):
            subhalo_ids = subhalo_context['subhalo_id'].tolist()

        # TODO create the attr in the init
        if not hasattr(self, 'subhalos_ids'):
            self.subhalos_ids = subhalo_ids
        else:
            self.subhalos_ids.extend(subhalo_ids)


class HaloContext:
    """
    A class to manage halo data in a context, including scaled properties stored in a Pandas DataFrame.

    Attributes:
    -----------
    data : pd.DataFrame
        A DataFrame storing properties for different halos, with halo_id as the index.

    Methods:
    --------
    __init__(properties, scale_factor, column_map):
        Initializes the HaloContext object with the provided halo properties, scaling them appropriately.
    get_item(halo_id):
        Retrieves the properties of a specific halo by its ID.
    halo_id:
        Property to retrieve the halo_id column from the stored DataFrame.
    """
    def __init__(self, df: pd.DataFrame, column_map):
        """
        Initialize a HaloContext object with scaled halo properties, stored in a Pandas DataFrame.

        Parameters:
        -----------
        properties : dict or pd.DataFrame
            Dictionary or DataFrame containing the halo properties (e.g., r200, m200, positions, etc.).
        scale_factor : float
            The scaling factor to be applied to specific properties like positions and radii.
        column_map : dict
            Dictionary defining the mapping of expected fields to the keys/columns in `properties`.
            Example:
                {
                    "halo_id": "halo_id",
                    "r200": "Group_R_Crit200",
                    "m200": "Group_M_Crit200",
                    "gpos": "GroupPos",
                    "fsub": "GroupFirstSub"
                }
        """
        if column_map is None:
            self.data = df

        else:
            if not isinstance(df, pd.DataFrame):
                raise Exception(f"Halo data must be a pd.DataFrame instead of {type(df)}")

            if len(df) == 0:
                raise Exception('The halo dataframe is empty.')

            # Create a DataFrame and set index
            df = df.rename(columns=column_map)
            df = df.rename(columns={'subhalo_id': 'fsub'})

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
                    if column not in ['halo_id', 'fsub']:
                        self.data.drop(columns=[column], inplace=True)

            self.data.set_index("halo_id", inplace=True, drop=False)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        """
        Build a HaloContext directly from an already-prepared DataFrame, with no column renaming/filtering.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame already in the expected shape (columns named
            `halo_id`, `fsub`, etc.), such as a previously-built
            HaloContext's `.data` after slicing.

        Returns:
        --------
        HaloContext
            A HaloContext wrapping `df` as-is (`column_map=None`).
        """
        return HaloContext(df, None)

    def __getitem__(self, index):
        """
        Slice the underlying DataFrame and wrap the result in a new HaloContext.

        Parameters:
        -----------
        index : slice or list
            Any indexer valid for `self.data[index]` (e.g. a boolean mask
            or a list of column names).

        Returns:
        --------
        HaloContext
            A new HaloContext wrapping the sliced data.
        """
        return HaloContext.from_dataframe(self.data[index])  # Return a new instance with sliced data

    def get_halo_by_id(self, halo_id) -> Halo:
        """
        Retrieve the properties for a specific halo ID and return a corresponding Halo object.

        Parameters:
        -----------
        halo_id : int
            The halo ID to query.

        Returns:
        --------
        Halo
            A Halo object containing the properties of the specified halo.

        Raises:
        -------
        Exception:
            If the provided halo_id is not found in the DataFrame.
        """
        try:
            halo_data = self.data.loc[halo_id].to_dict()
            return Halo(**halo_data)
        except KeyError:
            raise Exception(f"Halo ID {halo_id} not found.")

    @property
    def halo_id(self):
        """
        Retrieve the halo_id column from the DataFrame.

        Returns:
        --------
        pd.Series
            A Series containing the halo_id values.
        """
        return self.data["halo_id"].tolist()

    def filter_by_ids(self, halo_ids: list) -> "HaloContext":
        """
        Return a new HaloContext containing only halos with IDs in `halo_ids`.

        Parameters
        ----------
        halo_ids : list
            A list of halo IDs to keep.

        Returns
        -------
        HaloContext
            A new HaloContext instance with the filtered halos.
        """
        filtered_df = self.data[self.data["halo_id"].isin(halo_ids)]
        return HaloContext.from_dataframe(filtered_df)

    def remove_by_ids(self, halo_ids: list) -> "HaloContext":
        """
        Return a new HaloContext removing halos with IDs in `halo_ids`.

        Parameters
        ----------
        halo_ids : list
            A list of halo IDs to keep.

        Returns
        -------
        HaloContext
            A new HaloContext instance with removed halos.
        """
        filtered_df = self.data[~self.data["halo_id"].isin(halo_ids)]
        return HaloContext.from_dataframe(filtered_df)


