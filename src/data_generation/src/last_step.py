import os.path
import pandas as pd
from omegaconf import DictConfig

from src.data_generation.src.utils import setup_logger, merge_json_files_to_csv

logger = setup_logger("LastStep")


def get_pipeline_step(cfg: DictConfig, step_name: str):
    return next(
        (step for step in cfg.pipeline if step.name == step_name),
        None
    )


class LastStep:
    name = 'LastStep'

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    @staticmethod
    def _inform_errors(errors, dataset_dir, name):
        if len(errors) > 0:
            errors_df = pd.DataFrame(columns=['halo_id', 'error', 'number of particles', 'step'])
            error_list = errors[0]
            for e in error_list:
                if (isinstance(e, list) or isinstance(e, tuple)) and len(e) > 0:
                    row = {'halo_id': e[0], 'error': e[1], 'number of particles': e[2], 'step': e[3]}
                    print(f'Halo id: {e[0]}, error: {e[1]}, number of particles: {e[2]} in step: {e[3]}')
                    errors_df = pd.concat([errors_df, pd.DataFrame([row])], ignore_index=True)
                    print('')

            path = dataset_dir + f'/../logs/errors/errors_{name}.csv'

            if os.path.exists(path):
                existing_df = pd.read_csv(path)
                combined_df = pd.concat([existing_df, errors_df], ignore_index=True)
            else:
                combined_df = errors_df.copy()

            # Remove duplicates
            combined_df = combined_df.drop_duplicates()
            combined_df.to_csv(path, index=False)
            print(f'errors saved into: {path}')
            print(errors_df['error'].value_counts())

    def _check_compute_transition_radius_analysis(self, name):
        transition_radius_compute = False
        for feature in self.cfg.feature_names:
            transition_radius_compute = feature.get("transition_radius_compute", False)

        if transition_radius_compute:
            # Transition radius analysis
            dataset_dir = self.cfg.paths.datasets_dir
            path = dataset_dir + f'/../logs/transition_radius'
            output_file = dataset_dir + f'/../logs/transition_radius/{name}_transition_radius.csv'
            name = self.cfg.catalog.path.replace(dataset_dir, '')
            name = name.replace('/', '_')
            print(f'simulation: {name}')
            merge_json_files_to_csv(path, output_file, chunk_size=1000, length=None, simulation=name)

            df = pd.read_csv(output_file)
            df['transition_radius/R200'] = df['transition_radius'] / df['R200']
            print(f'n = {len(df)}')
            print(df.describe())

    def _check_maps(self):
        required_keys = ['path', 'name', 'spatial_resolution']
        for key in required_keys:
            if key not in self.cfg:
                if key not in self.cfg.get('catalog', {}):
                    raise ValueError(f"Error, missing required configuration key: {key}")

        map_step = get_pipeline_step(self.cfg, "map_step")

        if map_step is not None:
            dim = map_step.get("dim", None)
            n_zooms = map_step.get("n_zooms", None)
            print(dim, n_zooms)

        def get_defined_rotations(cfg):
            return {
                k: v
                for k, v in cfg.items()
                if k.startswith("rotation_")
            }

        rotations = get_defined_rotations(self.cfg)
        bin_spatial_resolution_scale = self.cfg.catalog.bin_spatial_resolution_scale

        # Take the first key
        rotation = f'no_rotation_{bin_spatial_resolution_scale}'
        for key, values in rotations.items():
            rotation = f'{key}_{bin_spatial_resolution_scale}'
            break

        features_cfg = self.cfg.feature_names
        feature_names = [f["name"] for f in features_cfg]
        if len(feature_names) != 2:
            raise ValueError(
                f"Expected exactly 2 features, got {len(feature_names)}: {feature_names}"
            )

        root_path = self.cfg.catalog.path
        catalog_name = self.cfg.catalog.name
        print(root_path, catalog_name, rotation, feature_names[0])
        path_i = os.path.join(root_path, rotation, feature_names[0])
        path_o = os.path.join(root_path, rotation, feature_names[1])

        print(path_i, path_o)
        if os.path.exists(path_i):
            print(f'Path: {path_i} exists')
        # TODO check whether the map exists and save the ones that don't

    def process(self, errors):
        logger.info('-------------------------------------------------------------------------------------')
        logger.info(f"Last step")
        logger.info('-------------------------------------------------------------------------------------')
        commit_hash = os.getenv("COMMIT_HASH", "unknown")
        logger.info(f"Commit hash:{commit_hash}")

        dataset_dir = self.cfg.paths.datasets_dir
        name = self.cfg.catalog.path.replace(dataset_dir, '')
        name = name.replace('/', '_')

        self._inform_errors(errors, dataset_dir, name)
        self._check_compute_transition_radius_analysis(name)
        self._check_maps()







