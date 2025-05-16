import traceback
from typing import Tuple, List, Optional, Callable, Any
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.data_fetcher import DataFetcherFactory
from omegaconf import DictConfig

from src.data_generation.src.feature import FeatureProcessor
from src.data_generation.src.halo import Halo, HaloContext
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.subhalo import SubhaloContext
from src.data_generation.src.utils import setup_logger, plot_transition_radius

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
        logger.debug(f"Process ID: {os.getpid()}")
        attr_list = self._build_data_loader_configuration(cfg)
        self.fetcher = DataFetcherFactory.get_data_fetcher(cfg.catalog, **attr_list)
        self.cfg = cfg

    @classmethod
    def _build_data_loader_configuration(cls, cfg: DictConfig) -> dict:
        """
        Extracts and validates the configuration for a specific data loader from the overall pipeline configuration.

        This method checks whether the current class (`cls.name`) is specified as a step in the pipeline.
        If so, it retrieves the corresponding configuration dictionary and ensures that all required keys are present.

        Parameters:
        ----------
        cfg : DictConfig
            The full Hydra configuration object containing the pipeline definition and associated parameters.

        Returns:
        -------
        dict
            The configuration dictionary corresponding to this data loader step in the pipeline.

        Raises:
        ------
        ValueError
            If the class name is not present in the pipeline, or if required configuration keys are missing.
        """
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
        logger.info('DataLoader step - Process method')

        lbox = None
        if 'Lbox' in self.cfg.catalog:
            lbox = self.cfg.catalog['Lbox']
            logger.info(f'Lbox: {lbox}')

        halo_context = self.fetcher.get_halos()
        logger.debug(len(g))

        with (tqdm(total=len(g), desc="Processing Halos", unit="halo") as pbar):
            transition_radius_df = pd.DataFrame(data=[], columns=['halo', 'r200', 'transition_radius',
                                                                  'normalized_transition_radius (r/r200)'])
            errors_overlapping = []
            halos_ignored = []
            # Note that subhalos here contains overlapping subhalos
            for halo, subhalo_ctx in g:
                logger.info(f"Processing halo {halo.halo_id} with {len(subhalo_ctx.data)} subhalos")
                try:
                    # Load particles for all the subhalos in that halo
                    logger.info(f"Get particles")
                    halo_particles_no_overlapping = self.fetcher.get_p_halo(halo=halo)

                    logger.info(f"Preprocess particles")
                    # Preprocessed all particles for current halo
                    halo_particles_no_overlapping.preprocess(halo, lbox)

                    # Load preprocessed particles for overlapping halos
                    logger.info(f"Check overlappings")
                    overlapping_subhalo_particles, subhalos_with_errors = \
                        self._load_particles_overlapping_subhalos(halo, subhalo_ctx, halo_context, lbox)

                    # Detect any errors during the retrival
                    errors_overlapping.extend(subhalos_with_errors)

                    # Group all the particles together
                    if overlapping_subhalo_particles is not None:
                        halo_particles = halo_particles_no_overlapping + overlapping_subhalo_particles
                    else:
                        halo_particles = halo_particles_no_overlapping

                except Exception as e:
                    pbar.update(1)  # Ensure progress bar updates even for skipped halos
                    logger.error(f"Failed to load particle data for halo {halo.halo_id}: {e}")
                    continue

                # Compute features with preprocessed and non-overlapping particles
                logger.info(f"Compute features")
                input_feature.process(halo=halo, subhalo_particles=halo_particles_no_overlapping)
                output_feature.process(halo=halo, subhalo_particles=halo_particles_no_overlapping)
                icl_mask = output_feature.value

                # Compute the transition radius, r = FOV / 2 if it's not in the config
                if 'transition_radius' not in self.cfg.catalog and hasattr(output_feature, 'get_transition_radius'):
                    transition_radius, normalized_transition_radius =\
                        output_feature.get_transition_radius(halo, halo_particles_no_overlapping, icl_mask)
                    if transition_radius in (0, None):
                        # Ignoring current halo
                        halos_ignored.append(halo.halo_id)
                        logger.warning(f'Ignoring halo: {halo.halo_id}')
                        continue
                    else:
                        new_row = pd.DataFrame([{
                            'halo': halo.halo_id,
                            'r200': halo.r200,
                            'transition_radius': transition_radius,
                            'normalized_transition_radius (r/r200)': normalized_transition_radius
                        }])
                        transition_radius_df = pd.concat([transition_radius_df, new_row], ignore_index=True)

                # Center subhalos positions respect to current halo, including overlapping subhalos
                logger.info(f"Preprocess subhalo_ctx")
                subhalo_ctx.preprocess(halo, lbox)
                tmp = CosmosData(halo, subhalo_ctx, halo.fsub, halo_particles)

                # Keep cosmos_data instance
                vars(cosmos_data).update(vars(tmp))

                if next_step is not None:
                    yield from next_step([(halo, subhalo_ctx)], **kwargs)
                else:
                    logger.info(f"Execute yield")
                    yield input_feature, output_feature, cosmos_data

                pbar.update(1)  # Update the progress bar for each successfully processed halo
                logger.info(f"Finished processing halo {halo.halo_id}")

        if 'transition_radius' not in self.cfg.catalog:
            transition_radius_l = transition_radius_df['transition_radius'].tolist()
            plot_transition_radius(self.cfg, transition_radius_l)
            plot_transition_radius(self.cfg, transition_radius_l, log_normalization=False)

            dataset_dir = self.cfg.dataset_dir
            name = self.cfg.catalog.path.replace(dataset_dir, '')
            name = name.replace('/', '_')
            path = self.cfg.paths.datasets_dir + f'/../logs/transition_radius_{name}.csv'
            logger.info(f'Storing transition radius df at: {path}')
            transition_radius_df.to_csv(path, index=False)

            logger.debug('transition_radius [kpc]')
            summary_stats = transition_radius_df['transition_radius'].describe()
            logger.info(summary_stats)
            logger.info(np.percentile(transition_radius_df['transition_radius'], 90))
            logger.info(np.percentile(transition_radius_df['transition_radius'], 95))

        logger.debug('normalized_transition_radius (r/r200)')
        summary_stats = transition_radius_df['normalized_transition_radius (r/r200)'].describe()
        logger.info(summary_stats)

        logger.warning(f'Halo, subhalo accessing errors when getting overlapping particles {len(errors_overlapping)}:'
                       f' {errors_overlapping}')
        logger.warning(f'Halos ignored since ICL never dominates (transition radius is None) {len(halos_ignored)}: '
                       f'{halos_ignored}')

    @staticmethod
    def load_halo_subhalo_data(subhalo_context: SubhaloContext, halo_context: HaloContext,
                               overlapping_matrix: dict) -> List[Tuple]:
        """
        Loads halo data and associated subhalo data, including overlapping subhalos from other halos.

        For each halo ID in the `halo_context`, the method:
        - Retrieves the corresponding halo object.
        - Retrieves its subhalos from `subhalo_context`.
        - Attaches those subhalos to the halo.
        - If the halo has overlapping subhalos (i.e., subhalos from other halos that are spatially near),
          these are also retrieved and merged into the subhalo context.
        - The final tuple of (halo, subhalo_ctx) is added to the result list.

        Parameters:
        ----------
        subhalo_context : SubhaloContext
            A context containing all subhalo data, with methods to filter subhalos by parent halo.

        halo_context : HaloContext
            A context containing all halo data and methods to retrieve halos by ID.

        overlapping_matrix : dict
            A dictionary mapping each halo ID to a list of tuples (parent_id, subhalo_id) representing
            subhalos from other halos that overlap with the given halo.

        Returns:
        -------
        List[Tuple]
            A list of tuples, where each tuple contains:
            - `Halo`: the halo object
            - `SubhaloContext`: the context object containing the main and overlapping subhalos for the halo.
        """
        logger.debug('Add subhalo data of overlapping halos to each subhalo context ...')
        result = []

        for id_halo in halo_context.halo_id:
            try:
                halo = halo_context.get_halo_by_id(id_halo)
                subhalo_ctx = subhalo_context.get_items_by_halo(id_halo)
                halo.add_children(subhalo_ctx)
                logger.debug(f'Halo: {halo.halo_id}, subhalos: {len(subhalo_ctx.data)}')

                # Find subhalo neighbours
                overlapping_halo_subhalo_ids = overlapping_matrix.get(id_halo, {})

                if overlapping_halo_subhalo_ids:
                    overlapping_subhalos_df = subhalo_context.data.loc[overlapping_halo_subhalo_ids]
                    overlapping_subhalo_ctx = SubhaloContext.from_dataframe(overlapping_subhalos_df)
                    subhalo_ctx += overlapping_subhalo_ctx
            except Exception as e:
                logger.error(e)
                continue
            result.append((halo, subhalo_ctx))
        return result

    def _load_particles_overlapping_subhalos(self, halo: Halo, subhalo_context: SubhaloContext,
                                             halo_context: HaloContext, lbox: float = None):
        """
        Loads and preprocesses particle data for subhalos that overlap with a given halo
        (i.e., subhalos that do not belong to the input halo).

        For each overlapping subhalo, the method:
        - Fetches particle data using the fetcher.
        - Retrieves the parent halo.
        - Preprocesses the particle data based on the halo's center and optional box size `lbox`.
        - Aggregates all valid subhalo particles into a single object.

        Parameters:
        ----------
        halo : Halo
            The central halo whose overlapping subhalos are to be processed.

        subhalo_context : SubhaloContext
            Context object containing the subhalo data. Must include:
            - 'subhalo_id'
            - 'id_parents' (or optionally 'group_number')

        halo_context : HaloContext
            Context object that provides access to halo information and retrieval
            via `get_halo_by_id`.

        lbox : float, optional
            Length of the simulation box. Used in preprocessing for periodic boundary
            conditions. Default is None.

        Returns:
        -------
        subhalo_particles : ParticleData or None
            The combined particle data from all successfully processed overlapping subhalos.
            If no subhalos are processed successfully, returns None.

        subhalos_with_errors : list of tuple
            A list of (halo_id, subhalo_id) tuples for which an error occurred during
            particle fetching or preprocessing.
        """
        logger.debug('Loading particles for overlapping subhalos ...')
        overlapping_subhalos = subhalo_context.data[subhalo_context.data['id_parents'] != halo.halo_id]
        subhalo_particles = None
        subhalos_with_errors = []
        if not overlapping_subhalos.empty:
            if 'group_number' in subhalo_context.data:
                halo_ids = overlapping_subhalos['group_number'].tolist()
            else:
                halo_ids = overlapping_subhalos['id_parents'].tolist()

            subhalo_ids = overlapping_subhalos['subhalo_id'].tolist()
            logger.debug(f'Number of overlapping halos/subhalos: {len(subhalo_ids)}')
            for i, subhalo_id in enumerate(subhalo_ids):
                halo_id = halo_ids[i]
                try:
                    logger.debug(f'Overlapping subhalo: {subhalo_id}, halo_id: {halo_id}')
                    subhalo_particles_i = self.fetcher.get_p_subhalo(subhalo_id=subhalo_id, halo_id=halo_id)
                    logger.debug(f'Overlapping particles: {subhalo_particles_i}')

                    # Get halo of this particles
                    logger.debug('Getting halo')
                    halo = halo_context.get_halo_by_id(halo_id)

                    # Preprocess the particles according to their halo center
                    logger.debug('Preprocessing halo')
                    subhalo_particles_i.preprocess(halo, lbox)

                    if subhalo_particles is None:
                        subhalo_particles = subhalo_particles_i
                    else:
                        logger.debug('Update subhalos particles with +')
                        subhalo_particles += subhalo_particles_i
                except Exception as e:
                    subhalos_with_errors.append((halo_id, subhalo_id))
                    logger.error(e)
                    error_message = traceback.format_exc()
                    logger.error(error_message)
                    continue
        logger.debug(f'Subhalos with accessing errors: {subhalos_with_errors}')
        return subhalo_particles, subhalos_with_errors

    @staticmethod
    def get_overlapping_halos(subhalo_context: SubhaloContext, halo_context: HaloContext):
        """
        Identifies overlapping halos based on spatial proximity of subhalos to halos.

        This method uses a KD-Tree to efficiently query which subhalos lie within twice the R200
        radius of each halo's center. It filters out subhalos that belong to the same halo
        (based on parent ID) to identify only external overlapping subhalos.

        Parameters:
        ----------
        subhalo_context : SubhaloContext
            Context object containing subhalo data, including:
            - 'spos': 3D positions of subhalos
            - 'id_parents': Parent halo IDs of subhalos
            - 'subhalo_id': Unique subhalo identifiers

        halo_context : HaloContext
            Context object containing halo data, including:
            - 'gpos': 3D positions of halos
            - 'r200': R200 radius values of halos
            - 'halo_id': Unique halo identifiers

        Returns:
        -------
        overlapping_matrix : dict
            A dictionary mapping each halo ID to a list of tuples of the form:
            (parent_halo_id, subhalo_id) representing overlapping subhalos that
            fall within 2 × R200 of the halo and do not belong to it.

            Example:
            {
                101: [(99, 3001), (102, 3002)],
                102: []
            }
        """
        logger.debug('Getting overlapping halos ...')
        subhalo_positions = np.vstack(subhalo_context.data['spos'])
        subhalo_tree = cKDTree(subhalo_positions)

        halo_positions = np.vstack(halo_context.data['gpos'])
        halo_r200 = halo_context.data['r200'].values * 2  # Precompute r200 * 2
        halo_ids = halo_context.data['halo_id'].values
        id_parents = subhalo_context.data['id_parents'].values
        subhalo_ids = subhalo_context.data['subhalo_id'].values

        overlapping_matrix = {}

        for i, halo_position in enumerate(halo_positions):
            # Get the radius (2 * r200 for the current halo)
            r = halo_r200[i]
            halo_id = halo_ids[i]

            # Query the subhalo tree for nearby subhalos within the radius
            nearby_sub_idxs = subhalo_tree.query_ball_point(halo_position, r)
            if nearby_sub_idxs:
                valid_subhalo_idxs = [subhalo_idx for subhalo_idx in nearby_sub_idxs if id_parents[subhalo_idx] != halo_id]
                # Only keep valid subhalos
                if valid_subhalo_idxs:
                    overlapping_matrix[halo_id] = list(
                        zip(id_parents[valid_subhalo_idxs], subhalo_ids[valid_subhalo_idxs]))
                    logger.debug(f'Halo {halo_id} has {len(valid_subhalo_idxs)} overlapping subhalos:')
                    logger.debug(f'------> {overlapping_matrix[halo_id]}')
                else:
                    overlapping_matrix[halo_id] = []
            else:
                overlapping_matrix[halo_id] = []
        return overlapping_matrix
