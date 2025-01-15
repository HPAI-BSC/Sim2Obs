from typing import Any

import numpy as np
from omegaconf import DictConfig

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.feature import FeatureProcessorFactory, FeatureProcessor
from src.data_generation.src.halo import Halo
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.subhalo import SubhaloContext
from src.data_generation.src.utils import setup_logger

logger = setup_logger("Pipeline")


class Pipeline:
    def __init__(self, cfg: DictConfig, steps: [PipelineStep]):
        """
        Initializes the pipeline with the provided configuration.

        Parameters:
        -----------
        cfg : DictConfig
            The configuration object containing the parameters for the pipeline steps.
        steps : list of PipelineStep
            A list of processing steps to be applied sequentially to the data.
        """
        self.cfg = cfg
        self.data_loader = DataLoader(cfg)
        self.steps = steps

    def build_pipeline(self, steps, g, cosmos_data: CosmosData, input_feature: FeatureProcessor,
                       output_feature: FeatureProcessor, **kwargs):

        if not steps:  # Base case: no steps left to apply
            return g

        # Get the current step and apply it
        current_step = steps[0]

        # Apply the current step, passing the generator and extra_param
        return current_step.process(input_feature, output_feature, cosmos_data,
                                    g, lambda g, **kwargs: self.build_pipeline(steps[1:], g, cosmos_data, input_feature,
                                                                     output_feature, **kwargs), **kwargs)

    def run(self, n=None) -> Any:
        """
        Executes the pipeline sequentially on the input data.

        Yields:
        -------
        Any
            The output from each pipeline step for the provided data.
        """
        feature_input, feature_output = self.cfg.feature_names
        input_feature = FeatureProcessorFactory.get_feature(feature_input, cfg=self.cfg)
        output_feature = FeatureProcessorFactory.get_feature(feature_output, cfg=self.cfg)
        data_loader = DataLoader(self.cfg)
        halo_subhalo_list = data_loader.load_halo_data()

        if n is not None:
            halo_subhalo_list = halo_subhalo_list[:n]

        steps = self.steps
        if len(steps) != 0:
            halo = Halo.empty_instance()
            subhalos = SubhaloContext.empty_instance()
            cosmos_data = CosmosData(halo=halo,
                                     subhalo_id=-1,
                                     subhalos=subhalos,
                                     subhalo_particle_positions=np.array([]))
            result = self.build_pipeline(steps, halo_subhalo_list, cosmos_data, input_feature, output_feature)
            output = list(result)
            return output
        else:
            logger.error("Error, please provide an non empty pipeline, steps cannot be an empty list.")
        logger.info("Pipeline processing completed.")
        return None
