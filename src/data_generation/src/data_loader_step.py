from typing import Tuple, List, Optional, Callable, Any
import os
from tqdm import tqdm
import cProfile
import pstats

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.data_fetcher import DataFetcherFactory
from omegaconf import DictConfig

from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.subhalo import SubhaloParticles
from src.data_generation.src.utils import setup_logger

logger = setup_logger("DataLoader")

os.environ['PYTHONIOENCODING'] = 'utf-8'


class DataLoader(PipelineStep):
    """
    A class responsible for loading halo and subhalo data, as well as particle data.

    This class interacts with the data fetcher factory to retrieve halo, subhalo, and particle data,
    yielding the relevant data in a structured format for further processing.

    Attributes:
    -----------
    fetcher : DataFetcher
        An instance of a data fetcher responsible for fetching the required data.
    """
    name = 'data_loader'

    def __init__(self, cfg: DictConfig):
        """
        Initializes the DataLoader with the provided configuration.

        Parameters:
        -----------
        cfg : DictConfig
            The configuration object containing parameters for the data fetching process.
        """
        attr_list = self._build_data_loader_configuration(cfg)
        self.fetcher = DataFetcherFactory.get_data_fetcher(cfg.catalog, **attr_list)
        self.cfg = cfg
        self.path = cfg.paths.datasets_dir + f'/logs/data_loader_particles_step.txt'
        self.path_f = cfg.paths.datasets_dir + f'/logs/data_loader_feature_init_step.txt'
        logger.info("DataLoader initialized with the provided configuration.")

    @classmethod
    def _build_data_loader_configuration(cls, cfg: DictConfig) -> dict:
        if cls.name not in [e['name'] for e in cfg.pipeline]:
            raise ValueError('Error, To execute a data_loader you must specify the map step in the pipeline of'
                             ' the configuration.')

        data_loader_dict = next((step for step in cfg.pipeline if step.name == cls.name), None)

        required_keys = ['subhalo_min_mass', 'halo_min_mass', 'solar_mass_convention', 'G', 'H']
        for key in required_keys:
            if key not in data_loader_dict:
                raise ValueError(f"Error, missing required configuration key: {key}")
        return data_loader_dict

    def process(self, input_feature: FeatureProcessor, output_feature: FeatureProcessor,
                cosmos_data: CosmosData, g: List[Tuple], next_step: Optional[Callable[[List[Tuple]], Any]]):
        """
        Processes the input and output features, updating the cosmos data for each halo and subhalo in the graph.

        Parameters:
        -----------
        input_feature : FeatureProcessor
            The processor for the input features to be processed.
        output_feature : FeatureProcessor
            The processor for the output features to be processed.
        cosmos_data : CosmosData
            The context and data related to the cosmos, which is used in the processing.
        g : List[Tuple]
            A list of tuples where each tuple contains halo data and its associated subhalo data.
        next_step : Optional[Callable[[List[Tuple]], Any]]
            A callable that represents the next step in the pipeline. It should accept the graph `g` and return the next value.

        Yields:
        -------
        Tuple
            A tuple containing the input feature, output feature, and cosmos data after processing.
        """
        with tqdm(total=len(g), desc="Processing Halos", unit="halo") as pbar:
            for halo, subhalo_df in g:
                logger.info(f"Processing halo {halo.halo_id} with {len(subhalo_df)} subhalos.")
                subhalo_id = halo.fsub
                halo_id = halo.halo_id
                try:
                    if hasattr(halo, 'group_number'):
                        subhalo_particles = self._load_subhalo_particles_data(subhalo_id=subhalo_id,
                                                                              halo_id=halo.group_number)
                    else:
                        profiler = cProfile.Profile()
                        profiler.enable()
                        subhalo_particles = self._load_subhalo_particles_data(subhalo_id=subhalo_id, halo_id=halo_id)
                        profiler.disable()
                        with open(self.path, "w") as f:
                            stats = pstats.Stats(profiler, stream=f)
                            stats.strip_dirs()
                            stats.sort_stats("tottime")
                            stats.print_stats()

                    if subhalo_particles is None:
                        raise ValueError(f"Invalid data for subhalo_id: {subhalo_id}, halo_id: {halo_id}")

                    if any(x is None for x in [subhalo_particles.p, subhalo_particles.v, subhalo_particles.m]):
                        raise ValueError("Missing particle data (positions, velocities, or masses).")
                except Exception as e:
                    pbar.update(1)  # Ensure progress bar updates even for skipped halos
                    logger.error(f"Failed to load particle data for halo {halo_id}: {e}")
                    continue

                logger.debug(f"Processing input features for halo {halo_id}")
                profiler = cProfile.Profile()
                profiler.enable()
                i_value = input_feature.process(
                    halo=halo,
                    particle_velocity=subhalo_particles.v,
                    particle_position=subhalo_particles.p)
                input_feature.value = i_value

                logger.debug(f"Processing output features for halo {halo_id}")
                o_value = output_feature.process(
                    halo=halo,
                    particle_velocity=subhalo_particles.v,
                    particle_position=subhalo_particles.p)
                output_feature.value = o_value
                profiler.disable()
                with open(self.path_f, "w") as f:
                    stats = pstats.Stats(profiler, stream=f)
                    stats.strip_dirs()
                    stats.sort_stats("tottime")
                    stats.print_stats()

                tmp = CosmosData(halo, subhalo_df, subhalo_id, subhalo_particles.p)
                cosmos_data.halo = tmp.halo
                cosmos_data.subhalos = tmp.subhalos
                cosmos_data.subhalo_id = tmp.subhalo_id
                cosmos_data.subhalo_particle_positions = tmp.subhalo_particle_positions
                cosmos_data.metadata = tmp.metadata
                if next_step is not None:
                    yield from next_step(g)
                else:
                    yield input_feature, output_feature, cosmos_data

                pbar.update(1)  # Update the progress bar for each successfully processed halo
                logger.info(f"Finished processing halo {halo_id}.")

    def load_halo_data(self) -> List[Tuple]:
        """
        Loads halo data and yields each halo's ID along with its associated data.

        Yields:
        -------
        list
            A list of tuples containing halo data and associated subhalo data for each halo ID.
        """
        halo_context = self.fetcher.get_halos()
        subhalo_context = self.fetcher.get_subhalos()
        n_halos = halo_context.data['halo_id'].nunique()
        n_subhalos = subhalo_context.data['subhalo_id'].nunique()
        logger.info(f'Number of halos: {n_halos}')
        logger.info(f'Number of subhalos: {n_subhalos}')
        result = []
        for id_halo in halo_context.halo_id:
            logger.debug(f"Loading data for halo_id: {id_halo}")
            halo_df = halo_context.get_item(id_halo)
            subhalo_df = subhalo_context.get_items_by_halo(id_halo)
            result.append((halo_df, subhalo_df))
        return result

    def _load_subhalo_particles_data(self, subhalo_id: int, halo_id: int=None) -> SubhaloParticles:
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
            subhalo_particles = self.fetcher.get_p_subhalo(subhalo_id=subhalo_id, group_number=halo_id)
            logger.debug(f"Loading particles for subhalo_id: {subhalo_id}, particles count: {len(subhalo_particles.m)}")
            return subhalo_particles
        except Exception as e:
            logger.warning(f"Error fetching subhalo data for id {subhalo_id}: {e}")
            return None