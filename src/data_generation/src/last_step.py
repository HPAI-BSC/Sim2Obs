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
        commit_hash = os.getenv("COMMIT_HASH", "unknown")
        logger.info(f"Commit hash:{commit_hash}")
        print(errors)

        if len(errors) > 0:
            errors_df = pd.DataFrame(columns=['halo_id', 'error', 'step'])
            error_list = errors[0]
            for e in error_list:
                if (isinstance(e, list) or isinstance(e, tuple)) and len(e) > 0:
                    error_item = e[0]
                    row = {'halo_id': error_item[0], 'error': error_item[1], 'step': error_item[2]}
                    print(f'Halo id: {error_item[0]}, error: {error_item[1]} in step: {error_item[2]}')
                    errors_df = pd.concat([errors_df, pd.DataFrame([row])], ignore_index=True)

            dataset_dir = self.cfg.paths.datasets_dir
            path = dataset_dir + f'/../logs/errors/errors_{commit_hash}.csv'
            print(f'errors saved into: {path}')
            print()
            print(errors_df['error'].value_counts())
            errors_df.to_csv(path, index=False)

        transition_radius_compute = False
        for feature in self.cfg.feature_names:
            transition_radius_compute = feature.get("transition_radius_compute", False)

        if transition_radius_compute:
            # Transition radius analysis
            dataset_dir = self.cfg.paths.datasets_dir
            path = dataset_dir + f'/../logs/transition_radius'
            output_file = dataset_dir + f'/../logs/transition_radius/merged.csv'
            name = self.cfg.catalog.path.replace(dataset_dir, '')
            name = name.replace('/', '_')
            # TODO DOUBLE CHECK WE CONSIDER JUST VALID JSONS
            print(f'simulation: {name}')
            merge_json_files_to_csv(path, output_file, chunk_size=1000, length=None, simulation=name)

            df = pd.read_csv(output_file)
            df['transition_radius/R200'] = df['transition_radius'] / df['R200']
            print(f'n = {len(df)}')
            print(df.describe())




