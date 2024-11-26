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
    __init__(r200, m200, gpos, fsub, scale_factor, halo_id):
        Initializes the HaloContext object with the provided halo properties, scaling them appropriately.
    get_item(halo_id):
        Retrieves the properties of a specific halo by its ID.
    halo_id:
        Property to retrieve the halo_id column from the stored DataFrame.
    """
    def __init__(self, r200, m200, gpos, fsub, scale_factor, halo_id):
        """
        Initialize a HaloContext object with scaled halo properties, stored in a Pandas DataFrame.

        Parameters:
        -----------
        r200 : np.ndarray
            Array of R200 values for halos to be scaled by the scale_factor.
        m200 : np.ndarray
            Array of M200 values for halos.
        gpos : np.ndarray
            Array of positions for halos to be scaled by the scale_factor.
        fsub : np.ndarray
            Array indicating the first subhalo indices.
        scale_factor : float
            The scaling factor to be applied to r200 and gpos.
        halo_id : np.ndarray
            Array of unique halo IDs.
        """
        # Create a DataFrame to hold halo properties
        self.data = pd.DataFrame({
            "halo_id": halo_id,
            "r200": r200 * scale_factor,        # Scaled radii
            "m200": m200,                       # Solar masses
            "gpos": list(gpos * scale_factor),  # Scaled positions
            "fsub": fsub.astype(int)            # First subhalo indices
        })

        # Set 'halo_id' as the index for efficient querying
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
        return self.data["halo_id"]