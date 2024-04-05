import os

import numpy as np

from src.preprocessing.io import IO, IOMode

Embryo = 1


class Sample:
    def __init__(self, path: str, r_id: str, img: np.array):
        self.path: str = path
        self.id: str = id
        self.img: np.ndarray = img

    def update_img(self, new_img):
        self.img = new_img


class PreprocessDataset:
    """
    Parameters
    ----------
    io : IO
        Interface that can handle multiple ways of reading and writing data according to the OS environment.

    data_root_path: str
        Path for the data folder.
    """

    def __init__(self, io: IO, data_root_path: str):
        self._io = io
        self._data_root_path = data_root_path
        # Enforce last character to be the IO separator
        if self._data_root_path[-1] != self._io.sep:
            self._data_root_path += self._io.sep

        # Create dataset without initialization.
        # To initialize it refer to `Dataset.load_data` or `Dataset.init_empty_dataset`
        self._samples = None

    ###############
    # Data access #
    ###############

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Embryo:
        image_path = self._samples[index]
        abs_instance_path = self._io.path_join(self._data_root_path, image_path)
        image = self._io.load_data(abs_instance_path, IOMode.IMG)
        return Sample(image_path, index, image)

    #################
    # Data creation #
    #################

    def init_empty_dataset(self) -> None:
        """Initialize an empty dataset with the proper data and folder structures to handle the
        data."""
        self._io.make_directory(self._data_root_path, overwrite=False)

    def load_data(self, num_studies: int = None) -> None:
        """Load the dataset file structure and metadata from the root path specified in the
        constructor.

        Parameters
        ----------
        num_studies : int, default: None
            Limits the number of samples to load in memory. With None (as default), loads all the available data.
        """
        samples = self._io.list_files(self._data_root_path, relpath_start=self._data_root_path)
        self._samples = samples[:num_studies]

    def is_initialized(self) -> bool:
        """Check if dataset is initialized.

        To initialize it refer to `Dataset.load_data` or `Dataset.init_empty_dataset`
        """
        return self._io.exists_directory(self._data_root_path)

    def check_persisted_image_is_valid(
        self, image_name: str, valid_extension: str = "jpg"
    ) -> bool:
        return image_name[-4:].lower() == valid_extension

    def dump_image(self, image):
        abs_path = os.path.join(self._data_root_path, image.path)
        self._io.dump_image(abs_path, image.img)
