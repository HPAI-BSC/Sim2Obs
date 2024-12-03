import numpy as np
import pandas as pd
import pandas.core.frame


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
    def __init__(self, properties: pd.DataFrame, scale_factor: float, column_map) -> None:
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

        required_fields = ["subhalo_id", "spos", "id_parents", "shmr", "mstr"]
        for field in required_fields:
            if field not in column_map:
                raise ValueError(f"Missing required field '{field}' in column_map.")

        properties[column_map["shmr"]] = properties[column_map["shmr"]] * scale_factor
        properties[column_map["spos"]] = [spos_i * scale_factor for spos_i in properties[column_map["spos"]]]

        # Create a DataFrame
        self.data = pd.DataFrame({
            "id_parents": properties[column_map["id_parents"]],  # Halo IDs
            "subhalo_id": properties[column_map["subhalo_id"]],  # Subhalo IDs
            "spos": properties[column_map["spos"]],  # Scaled positions
            "shmr": list(properties[column_map["shmr"]]),  # Scaled half-mass radius
            "mstr": list(properties[column_map["mstr"]])  # Stellar mass
        })

        # Add any additional fields dynamically
        for key, col_name in column_map.items():
            if key not in required_fields:
                self.data[key] = properties[col_name]

        # Set a multi-index for efficient querying
        self.data.set_index(["id_parents", "subhalo_id"], inplace=True, drop=False)

    def get_item(self, subhalo_id, halo_id=None) -> SubHalo:
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

    # TODO modify this
    def get_items_by_halo(self, halo_id) -> pd.DataFrame:
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
            return self.data.loc[halo_id]
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

    @classmethod
    def empty_instance(cls):
        """
        Returns an empty instance of the SubhaloContext object with default empty values.
        """
        # Provide empty or default values for the required fields
        # For instance, we create an empty DataFrame and set scale_factor to 1.0
        empty_properties = {
            "subhalo_id": [],
            "spos": [],
            "id_parents": [],
            "shmr": [],
            "mstr": []
        }

        # Assuming the column map is the same as the one required in __init__
        empty_column_map = {
            "subhalo_id": "subhalo_id",
            "spos": "spos",
            "id_parents": "id_parents",
            "shmr": "shmr",
            "mstr": "mstr"
        }

        # Creating an empty DataFrame
        empty_df = pd.DataFrame(empty_properties)

        # Call the constructor with empty or default values
        return cls(properties=empty_df, scale_factor=1.0, column_map=empty_column_map)


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
    def __init__(self, m: np.ndarray, p: np.ndarray, v: np.ndarray,
                 scale_factor: float = None, init: bool = True) -> None:
        """
        Initialize the SubhaloParticles with mass, position, and velocity data.

        Parameters:
        -----------
        m : np.ndarray
            The mass of particles.
        p : np.ndarray
            The positions of particles. Should be of shape (N, 3) where N is the number of particles.
        v : np.ndarray
            The velocities of particles. Should also be of shape (N, 3).
        scale_factor : float, optional
            The scale factor to apply to positions and velocities. Defaults to None.
        init : bool, optional
            If True, initializes the attributes with conversion. Defaults to True.
        """
        if init:
            self.m = m
            self.p = p * scale_factor
            self.v = v
        else:
            self.m = m
            self.p = p
            self.v = v




