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
    __init__(**kwargs):
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
    def __init__(self, properties: pd.DataFrame, scale_factor, column_map):
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
        if not isinstance(properties, pd.DataFrame):
            raise Exception(f"Halo properties must be a pd.DataFrame instead of {type(properties)}")

        required_fields = ["halo_id", "r200", "m200", "gpos", "fsub"]
        for field in required_fields:
            if field not in column_map:
                raise ValueError(f"Missing required field '{field}' in column_map.")

        # Map and process data
        data = {
            "halo_id": properties[column_map["halo_id"]],
            "r200": properties[column_map["r200"]] * scale_factor,
            "m200": properties[column_map["m200"]],
            "gpos": list(properties[column_map["gpos"]].apply(lambda x: x * scale_factor)),
            "fsub": properties[column_map["fsub"]].astype(int)
        }

        # Add any additional fields dynamically
        for key, col_name in column_map.items():
            if key not in required_fields:
                data[key] = properties[col_name]

        # Create a DataFrame and set index
        self.data = pd.DataFrame(data)
        self.data.set_index("halo_id", inplace=True, drop=False)

    def get_item(self, halo_id) -> Halo:
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
