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

    def process(self, results):
        logger.info('-------------------------------------------------------------------------------------')
        logger.info(f"Last step")
        logger.info('-------------------------------------------------------------------------------------')

        path = self.cfg.paths.datasets_dir + f'/../logs/results_illustris.joblib'
        with open(path) as f:
            joblib.dump(results, f)

        transition_radius_analysis = True

        # Collect statistics and errors
        data = {'halo_id': [], 'r200': [], 'transition_radius': [], 'normalized_transition_radius': []}
        errors = {'halo_id': [], 'error': [], 'step': []}
        ignored_halos = 0
        for element in results:
            if isinstance(element, tuple) and isinstance(element[2], ICLProcessor):
                radius = element[2].transition_radius
                if radius is not None:
                    normalized = radius / element[0].r200
                    data['halo_id'].append(element[0].halo_id)
                    data['r200'].append(element[0].r200)
                    data['transition_radius'].append(radius)
                    data['normalized_transition_radius'].append(normalized)
                    plot_icl_bcg_profiles(self.cfg, element[2].icl_cumulative, element[2].bcg_cumulative, element[2].log_r,
                                          element[2].tr_index, element[0].halo_id)
            elif isinstance(element, list) and len(element) > 0:
                errors['halo_id'].append(element[0][0].halo_id)
                errors['error'].append(element[0][1])
                errors['step'].append(element[0][2])

                if 'ICL never dominates' in element[0][1]:
                    ignored_halos += 1

        dataset_dir = self.cfg.paths.datasets_dir
        name = self.cfg.catalog.path.replace(dataset_dir, '')
        name = name.replace('/', '_')

        if transition_radius_analysis:
            transition_radius_df = pd.DataFrame.from_dict(data)
            transition_radius_l = transition_radius_df['transition_radius'].tolist()
            plot_transition_radius(self.cfg, transition_radius_l)
            plot_transition_radius(self.cfg, transition_radius_l, log_normalization=False)
            path = self.cfg.paths.datasets_dir + f'/../logs/transition_radius_{name}.csv'
            logger.info(f'Storing transition radius df at: {path}')
            transition_radius_df.to_csv(path, index=False)
            logger.debug('transition_radius [kpc]')
            summary_stats = transition_radius_df['transition_radius'].describe()
            logger.info(summary_stats)
            logger.info(np.percentile(transition_radius_df['transition_radius'], 90))
            logger.info(np.percentile(transition_radius_df['transition_radius'], 95))
            logger.debug('normalized_transition_radius (r/r200)')
            summary_stats = transition_radius_df['normalized_transition_radius'].describe()
            logger.info(summary_stats)

        errors_df = pd.DataFrame.from_dict(errors)
        path = self.cfg.paths.datasets_dir + f'/../logs/errors_{name}.csv'
        logger.info(f'Storing errors df at: {path}')
        errors_df.to_csv(path, index=False)
        # TODO not working
        logger.error(f'Ignored halos during transition radius computation, ICL never dominates: {ignored_halos}')




