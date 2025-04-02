import time
from typing import Tuple, List, Optional, Callable, Any
import os

from tqdm import tqdm

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.data_fetcher import DataFetcherFactory
from omegaconf import DictConfig, ListConfig

from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.halo import Halo
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.subhalo import SubhaloParticles, SubHalo
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

    @classmethod
    def _build_data_loader_configuration(cls, cfg: DictConfig) -> dict:
        if cls.name not in [e['name'] for e in cfg.pipeline]:
            raise ValueError('Error, To execute a data_loader you must specify the map step in the pipeline of'
                             ' the configuration.')

        data_loader_dict = next((step for step in cfg.pipeline if step.name == cls.name), None)

        # TODO
        required_keys = ['G']
        for key in required_keys:
            if key not in data_loader_dict:
                raise ValueError(f"Error, missing required configuration key: {key}")
        return data_loader_dict

    def process(self, input_feature: FeatureProcessor, output_feature: FeatureProcessor,
                cosmos_data: CosmosData, g: List[Tuple], next_step: Optional[Callable[[List[Tuple]], Any]],
                **kwargs):
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
            for halo, subhalo_ctx in g:
                logger.info(f"Processing halo {halo.halo_id} with {len(subhalo_ctx.data)} subhalos.")
                try:
                    subhalo_particles = self._load_subhalo_particles_data(halo)

                except Exception as e:
                    pbar.update(1)  # Ensure progress bar updates even for skipped halos
                    logger.error(f"Failed to load particle data for halo {halo.halo_id}: {e}")
                    continue

                subhalo_particles.preprocess(halo)

                input_feature.value = input_feature.process(halo=halo, subhalo_particles=subhalo_particles)
                icl_mask = output_feature.process(halo=halo, subhalo_particles=subhalo_particles)
                output_feature.value = icl_mask

                # Compute the transition radius, r = FOV / 2
                transition_radius = output_feature.get_transition_radius(halo, subhalo_particles, icl_mask)
                if transition_radius in (0, None):
                    # Ignoring current halo
                    logger.warning(f'Ignoring halo: {halo.halo_id}')
                    continue

                tmp = CosmosData(halo, subhalo_ctx, halo.fsub, subhalo_particles)
                # Keep cosmos_data instance
                vars(cosmos_data).update(vars(tmp))

                # Add new metadata for map generating
                cosmos_data.metadata.update({"map": {"transition_radius": transition_radius,
                                                     "fov": transition_radius * 2
                                                     }})

                if next_step is not None:
                    yield from next_step([(halo, subhalo_ctx)], **kwargs)
                else:
                    yield input_feature, output_feature, cosmos_data

                pbar.update(1)  # Update the progress bar for each successfully processed halo
                logger.info(f"Finished processing halo {halo.halo_id}")

    def load_halo_subhalo_data(self) -> List[Tuple]:
        """
        Loads halo data and yields each halo's ID along with its associated data.

        Yields:
        -------
        list
            A list of tuples containing halo data and associated subhalo data for each halo ID.
        """
        halo_context = self.fetcher.get_halos()
        subhalo_context = self.fetcher.get_subhalos()

        subhalo_context.preprocess(halo_context)
        result = []

        for id_halo in halo_context.halo_id:
            try:
                halo = halo_context.get_halo_by_id(id_halo)
                subhalo_ctx = subhalo_context.get_items_by_halo(id_halo)
            except Exception as e:
                logger.error(e)
                continue
            result.append((halo, subhalo_ctx))
        return result

    def _load_subhalo_particles_data(self, halo: Halo) -> SubhaloParticles:
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
            subhalo_id = halo.fsub
            halo_id = halo.halo_id
            if hasattr(halo, 'group_number'):
                subhalo_particles = self.fetcher.get_p_subhalo(subhalo_id=subhalo_id, group_number=halo.group_number)
            else:
                subhalo_particles = self.fetcher.get_p_subhalo(subhalo_id=subhalo_id, halo_id=halo_id)
            return subhalo_particles
        except Exception as e:
            logger.warning(f"Error fetching subhalo data for id {subhalo_id}: {e}")
            raise ValueError(f"Invalid data for subhalo_id: {subhalo_id}, halo_id: {halo_id}")
