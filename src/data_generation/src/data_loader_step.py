from typing import Generator
import rootutils

from src.data_generation.src.data_fetcher import DataFetcherFactory
from omegaconf import DictConfig
from src import utils

rootutils.setup_root(__file__, pythonpath=True)
log = utils.get_pylogger(__name__)


class DataLoader:
    """
    A class responsible for loading halo and subhalo data, as well as particle data.

    This class interacts with the data fetcher factory to retrieve halo, subhalo, and particle data,
    yielding the relevant data in a structured format for further processing.

    Attributes:
    -----------
    fetcher : DataFetcher
        An instance of a data fetcher responsible for fetching the required data.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initializes the DataLoader with the provided configuration.

        Parameters:
        -----------
        cfg : DictConfig
            The configuration object containing parameters for the data fetching process.
        """
        self.fetcher = DataFetcherFactory.get_data_fetcher(cfg)
        log.info("DataLoader initialized with the provided configuration.")

    def load_halo_data(self) -> Generator[tuple, None, None]:
        """
        Loads halo data and yields each halo's ID along with its associated data.

        Yields:
        -------
        tuple
            A tuple containing halo data and associated subhalo data for each halo ID.
        """
        halo_context = self.fetcher.get_halos()
        subhalo_context = self.fetcher.get_subhalos()
        for id_halo in halo_context.halo_id:
            log.debug(f"Loading data for halo_id: {id_halo}")
            halo_df = halo_context.get_item(id_halo)
            subhalo_df = subhalo_context.get_items_by_halo(id_halo)
            yield halo_df, subhalo_df

    def load_subhalo_particles_data(self, subhalo_id: int, halo_id: int=None) -> Generator[tuple, None, None]:
        """
        Loads subhalo particle data for a given subhalo ID and applies rotations.

        Parameters:
        -----------
        id_subhalo : int
            The ID of the subhalo for which particle data is to be loaded.

        Yields:
        -------
        subhalo_particles
            The particle data for the specified subhalo ID.
        """
        try:
            subhalo_particles = self.fetcher.get_p_subhalo(subhalo_id=subhalo_id, halo_id=halo_id)
            log.debug(f"Loading particles for subhalo_id: {subhalo_id}, particles count: {len(subhalo_particles.m)}")
            yield subhalo_particles
        except Exception as e:
            log.warning(f"Error fetching subhalo data for id {subhalo_id}: {e}")