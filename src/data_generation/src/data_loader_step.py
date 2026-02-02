import json
import logging
import traceback
from typing import Tuple, List, Optional, Callable, Any
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.data_fetcher import DataFetcherFactory
from omegaconf import DictConfig, OmegaConf

from src.data_generation.src.feature import FeatureProcessorFactory, attach_models_to_features
from src.data_generation.src.halo import Halo, HaloContext
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles
from src.data_generation.src.utils import setup_logger, store_obj

logger = setup_logger("DataLoader-step")

os.environ['PYTHONIOENCODING'] = 'utf-8'


def filter_outliers_transition_radius(transition_radius_path):
    df = pd.read_csv(transition_radius_path)
    transition_radius = df['transition_radius']
    print(transition_radius.describe())

    percentiles = [99, 98, 97, 96, 95]

    for percentile in percentiles:
        upper_bound = np.percentile(transition_radius, percentile)
        lower_bound = 100 - upper_bound
        filtered_df = df[(transition_radius >= lower_bound) & (transition_radius <= upper_bound)]
        print(
            f'Percentile: {percentile}, transition radius [kpc]: {upper_bound}, final size: {len(filtered_df)}, Ignored halos: {len(df) - len(filtered_df)}')
        ignored_halos = set(df['halo_id']) - set(filtered_df['halo_id'])
        # print(ignored_halos)

        plt.hist(filtered_df["transition_radius"] / filtered_df['R200'], bins=100)
        plt.title(f"Transition radius/R200 distribution, percentile: {percentile}")
        plt.show()


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
        self.errors = []

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

    def compute_transition_radius(self, g: List[Tuple], **kwargs):
        """
        Processes the input and output features, updating the cosmos data for each halo and subhalo in the graph.

        Parameters:
        -----------
        g : List[Tuple]
            A list of tuples where each tuple contains halo data and its associated subhalo data.
        """
        lbox = self.cfg.catalog.get('Lbox', None)
        halo_context = kwargs.pop('halo_context', None)
        models = kwargs.pop('models', None)
        debug_halo_ids = kwargs.pop('debug_halo_ids', None)

        # Note that subhalos here contains overlapping subhalos
        for halo, subhalo_ctx, cosmos_data in g:
            logger.info('-------------------------------------------------------------------------------------')
            logger.info(f"Processing halo {halo.halo_id} with {len(subhalo_ctx.data)} subhalos")
            logger.info('-------------------------------------------------------------------------------------')

            dataset_dir = self.cfg.paths.datasets_dir
            name = self.cfg.catalog.path.replace(dataset_dir, '')
            name = name.replace('/', '_')
            commit_hash = os.getenv("COMMIT_HASH", "unknown")
            path = dataset_dir + f'/../logs/transition_radius/{name}_transition_radius_info_{halo.halo_id}.json'
            logger.debug(f'{path}')
            # Do the transition radius analysis iff the halo has not been processed before
            if not os.path.exists(path):
                try:
                    feature_input_name, feature_output_name = self.cfg.feature_names
                    input_feature = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg)
                    output_feature = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg)
                    attach_models_to_features([input_feature, output_feature], models)

                    # Load particles for all the subhalos in that halo
                    halo_particles_no_overlapping = self.fetcher.get_p_halo(halo=halo)

                    # Preprocessed all particles for current halo
                    halo_particles_no_overlapping.preprocess(halo, lbox)

                    # Load preprocessed particles for overlapping halos
                    overlapping_subhalo_particles, subhalos_with_errors = \
                        self._load_particles_overlapping_subhalos(halo, subhalo_ctx, halo_context, lbox)

                    # Group all the particles together
                    if overlapping_subhalo_particles is not None:
                        halo_particles_overlapping = halo_particles_no_overlapping + overlapping_subhalo_particles
                    else:
                        halo_particles_overlapping = halo_particles_no_overlapping

                    # Compute features with preprocessed and non-overlapping particles
                    input_feature.process(halo=halo, subhalo_particles=halo_particles_no_overlapping)
                    output_feature.process(halo=halo, subhalo_particles=halo_particles_no_overlapping)

                    # Center subhalos positions respect to current halo, including overlapping subhalos
                    subhalo_ctx.preprocess(halo, lbox)

                    # Store info for log purposes
                    halo_dict = {'halo_id': halo.halo_id, 'R200': halo.r200,
                                 'number_particles': len(halo_particles_overlapping.star_p),
                                 'transition_radius': output_feature.transition_radius}

                    with open(path, "w") as f:
                        json.dump(halo_dict, f, indent=4)
                except Exception as e:
                    logger.error(f"Error with {halo.halo_id}: {e}")
                    self.errors.append((halo.halo_id, str(e), len(halo_particles_overlapping.star_p), self.name))
                    continue

                # Update cosmos instance
                cosmos_data.subhalos = subhalo_ctx
                cosmos_data.subhalo_particles = halo_particles_overlapping
                cosmos_data.input_feature = input_feature
                cosmos_data.output_feature = output_feature
                logger.info(f"Finished processing halo {halo.halo_id}")
            else:
                logger.debug(f'Halo {halo.halo_id} has been already processed, skipping it.')
        yield self.errors

    def execute_process(self, g: List[Tuple], next_step: Optional[Callable[[List[Tuple]], Any]], **kwargs):
        """
        Processes the input and output features, updating the cosmos data for each halo and subhalo in the graph.

        Parameters:
        -----------
        g : List[Tuple]
            A list of tuples where each tuple contains halo data and its associated subhalo data.
        next_step : Optional[Callable[[List[Tuple]], Any]]
            A callable that represents the next step in the pipeline. It should accept the graph `g` and return the next value.

        Yields:
        -------
        Tuple
            A tuple containing the input feature, output feature, and cosmos data after processing.
        """
        lbox = self.cfg.catalog.get('Lbox', None)
        halo_context = kwargs.pop('halo_context', None)
        models = kwargs.pop('models', None)
        debug_halo_ids = kwargs.pop('debug_halo_ids', None)

        # Note that subhalos here contains overlapping subhalos
        for halo, subhalo_ctx, cosmos_data in g:
            logger.info('-------------------------------------------------------------------------------------')
            logger.info(f"Processing halo {halo.halo_id} with {len(subhalo_ctx.data)} subhalos")
            logger.info('-------------------------------------------------------------------------------------')
            # TODO this is apparently not filtering

            ignore_halos = OmegaConf.select(self.cfg, "catalog.ignore_halos", default=[])
            if halo not in ignore_halos:
                try:
                    feature_input_name, feature_output_name = self.cfg.feature_names
                    input_feature = FeatureProcessorFactory.get_feature(feature_input_name, cfg=self.cfg)
                    output_feature = FeatureProcessorFactory.get_feature(feature_output_name, cfg=self.cfg)
                    attach_models_to_features([input_feature, output_feature], models)

                    # Load particles for all the subhalos in that halo
                    halo_particles_no_overlapping = self.fetcher.get_p_halo(halo=halo)

                    # Preprocessed all particles for current halo
                    halo_particles_no_overlapping.preprocess(halo, lbox)

                    # Load preprocessed particles for overlapping halos
                    overlapping_subhalo_particles, subhalos_with_errors = \
                        self._load_particles_overlapping_subhalos(halo, subhalo_ctx, halo_context, lbox)

                    # Group all the particles together
                    if overlapping_subhalo_particles is not None:
                        halo_particles_overlapping = halo_particles_no_overlapping + overlapping_subhalo_particles
                    else:
                        halo_particles_overlapping = halo_particles_no_overlapping

                    # Compute features with preprocessed and non-overlapping particles
                    input_feature.process(halo=halo, subhalo_particles=halo_particles_no_overlapping)
                    output_feature.process(halo=halo, subhalo_particles=halo_particles_no_overlapping)

                    # Center subhalos positions respect to current halo, including overlapping subhalos
                    subhalo_ctx.preprocess(halo, lbox)
                except Exception as e:
                    logger.error(f"Error with {halo.halo_id}: {e}")
                    self.errors.append((halo.halo_id, str(e), self.name))
                    continue

                # Update cosmos instance
                cosmos_data.subhalos = subhalo_ctx
                cosmos_data.subhalo_particles = halo_particles_overlapping
                cosmos_data.input_feature = input_feature
                cosmos_data.output_feature = output_feature

                # Analyze objects with errors offline
                # Got to the next step
                if next_step is not None:
                    yield from next_step([(halo, subhalo_ctx, cosmos_data)], **kwargs)
                else:
                    yield cosmos_data.halo, cosmos_data.input_feature, cosmos_data.output_feature

                logger.info(f"Finished processing halo {halo.halo_id}")
        yield self.errors

    def process(self, g: List[Tuple], next_step: Optional[Callable[[List[Tuple]], Any]], **kwargs):
        """
        Processes the input and output features, updating the cosmos data for each halo and subhalo in the graph.

        Parameters:
        -----------
        g : List[Tuple]
            A list of tuples where each tuple contains halo data and its associated subhalo data.
        next_step : Optional[Callable[[List[Tuple]], Any]]
            A callable that represents the next step in the pipeline. It should accept the graph `g` and return the next value.

        Yields:
        -------
        Tuple
            A tuple containing the input feature, output feature, and cosmos data after processing.
        """
        if len(self.cfg.feature_names) > 1:
            transition_radius_compute = self.cfg.feature_names[1].get(
                "transition_radius_compute", False
            )
        else:
            transition_radius_compute = False

        if transition_radius_compute:
            yield from self.compute_transition_radius(g, **kwargs)
        else:
            yield from self.execute_process(g, next_step, **kwargs)

    def load_halo_subhalo_data(self, subhalo_context: SubhaloContext, halo_context: HaloContext,
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

                # Prepare a default cosmos_data placeholder to pass during pipeline build (adjust if needed)
                subhalo_particles = SubhaloParticles({}, None)
                cosmos_data = CosmosData(halo=halo, subhalo_id=halo.fsub, subhalos=subhalo_ctx,
                                         subhalo_particles=subhalo_particles)
                halo.add_children(subhalo_ctx)

                # Find subhalo neighbours
                overlapping_halo_subhalo_ids = overlapping_matrix.get(id_halo, {})

                if overlapping_halo_subhalo_ids:
                    overlapping_subhalos_df = subhalo_context.data.loc[overlapping_halo_subhalo_ids]
                    overlapping_subhalo_ctx = SubhaloContext.from_dataframe(overlapping_subhalos_df)
                    subhalo_ctx += overlapping_subhalo_ctx
            except Exception as e:
                logger.error(f"Error with {id_halo}: {e}")
                self.errors.append(f'Error with {id_halo}: {e} in class {self.name}')
                continue
            result.append((halo, subhalo_ctx, cosmos_data))
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
        print(f'Overlapping subhalos for {halo.halo_id}: {len(overlapping_subhalos)}')
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
                    halo = halo_context.get_halo_by_id(halo_id)

                    # Preprocess the particles according to their halo center
                    subhalo_particles_i.preprocess(halo, lbox)

                    if subhalo_particles is None:
                        subhalo_particles = subhalo_particles_i
                    else:
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
                    #logger.debug(f'Halo {halo_id} has {len(valid_subhalo_idxs)} overlapping subhalos:')
                    #logger.debug(f'------> {overlapping_matrix[halo_id]}')
                else:
                    overlapping_matrix[halo_id] = []
            else:
                overlapping_matrix[halo_id] = []
        return overlapping_matrix
