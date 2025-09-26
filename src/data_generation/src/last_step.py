import pickle

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_generation.src.feature import ICLProcessor
from src.data_generation.src.utils import plot_transition_radius, setup_logger, plot_icl_bcg_profiles

logger = setup_logger("LastStep")


class LastStep:
    name = 'LastStep'

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def process(self, errors):
        logger.info('-------------------------------------------------------------------------------------')
        logger.info(f"Last step")
        logger.info('-------------------------------------------------------------------------------------')
        print(errors)
        dataset_dir = self.cfg.paths.datasets_dir
        name = self.cfg.catalog.path.replace(dataset_dir, '')
        name = name.replace('/', '_')
        errors_df = pd.DataFrame(errors)
        path = self.cfg.paths.datasets_dir + f'/../logs/errors_{name}.csv'
        logger.info(f'Storing errors df at: {path}')
        errors_df.to_csv(path, index=False)
        print(len(errors_df))
        print(errors_df.head())




