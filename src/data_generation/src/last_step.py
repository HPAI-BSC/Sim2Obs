import os.path
import pickle
import shutil

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data_generation.src.feature import ICLProcessor
from src.data_generation.src.utils import plot_transition_radius, setup_logger, plot_icl_bcg_profiles, \
    merge_json_files_to_csv

logger = setup_logger("LastStep")


class LastStep:
    name = 'LastStep'

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def process(self, errors):
        logger.info('-------------------------------------------------------------------------------------')
        logger.info(f"Last step")
        logger.info('-------------------------------------------------------------------------------------')

        if len(self.cfg.feature_names) > 1:
            transition_radius_compute = self.cfg.feature_names[1].get(
                "transition_radius_compute", False
            )
        else:
            transition_radius_compute = False

        if transition_radius_compute:
            # Transition radius analysis
            dataset_dir = self.cfg.paths.datasets_dir
            path = dataset_dir + f'/../logs/transition_radius'
            output_file = dataset_dir + f'/../logs/transition_radius/merged.csv'
            merge_json_files_to_csv(path, output_file, chunk_size=1000, length=None)

            df = pd.read_csv(output_file)
            df['transition_radius/R200'] = df['transition_radius'] / df['R200']
            print(f'n = {len(df)}')
            print(df.describe())

            # Errors
            path = dataset_dir + f'/../logs/errors'
            output_file = dataset_dir + f'/../logs/errors/errors_merged.csv'
            merge_json_files_to_csv(path, output_file, chunk_size=1000, length=None)
            logger.info(f'Storing errors at: {output_file}')
            df = pd.read_csv(output_file)
            print(f'n = {len(df)}')
            print(df['error'].value_counts())
        else:
            dataset_dir = self.cfg.paths.datasets_dir
            name = self.cfg.catalog.path.replace(dataset_dir, '')
            name = name.replace('/', '_')
            path = self.cfg.paths.datasets_dir + f'/../logs/errors_{name}.txt'
            logger.info(f'Storing errors at: {path}')

            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.isfile(path):
                os.remove(path)

            with open(path, 'w') as f:
                for item in errors:
                    if isinstance(item, list):
                        for error in item:
                            f.write(f"{error}\n")
                    else:
                        f.write(f"{item}\n")




