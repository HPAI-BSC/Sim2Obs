import os
import shutil
import unittest

from hydra import initialize, compose

from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.map_step import MapStep
from src.data_generation.src.pipeline import Pipeline
from src.data_generation.src.rotate_step import RotateStep


class TestPipeline(unittest.TestCase):
    def setUp(self):
        with initialize(config_path="../../../configs"):
            self.cfg_illustris = compose(config_name="data_generation.yaml")
            self.cfg_eagle = compose(config_name="data_generation.yaml")
            local_config = compose(config_name="paths/local.yaml")
            self.cfg_eagle.paths = local_config.paths
            self.cfg_illustris.paths = local_config.paths
            self.cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']
            self.cfg_eagle.catalog = compose("catalog/eagle.yaml")['catalog']
            self.cfg_eagle.catalog.simulation_code = 'RefL0050N0752'
            self.cfg_eagle.catalog.path = '../../../local/data/datasets/eagle/RefL0050N0752'

        os.environ['PROJECT_ROOT'] = '../../..'
        self.remove_folder = True

    def tearDown(self):
        if self.map_step.map_configuration['rotation_angle'] is not None:
            path = os.path.join(self.map_step.map_configuration['root_path'],
                                f'rotation_{self.map_step.map_configuration["rotation_angle"]}')
        else:
            path = os.path.join(self.map_step.map_configuration['root_path'], 'VelocityDispersion')

        if self.remove_folder:
            if os.path.exists(path):
                shutil.rmtree(path)
                if self.map_step.map_configuration['rotation_angle'] is None:
                    path = os.path.join(self.map_step.map_configuration['root_path'], 'IntraClusterLight')
                    shutil.rmtree(path)

    def test_run_illustris(self):
        rotate_step = RotateStep(self.cfg_illustris)
        self.map_step = MapStep(self.cfg_illustris)
        self.map_step.map_configuration['root_path'] = '../test/data/illustris'
        data_loader = DataLoader(self.cfg_illustris)
        self.pipeline_illustris = Pipeline(cfg=self.cfg_illustris, steps=[data_loader, rotate_step, self.map_step])
        # Execute the pipeline and capture the first 5 iterations
        self.pipeline_illustris.run(n=1)

    def test_run_eagle(self):
        rotate_step = RotateStep(self.cfg_eagle)
        self.map_step = MapStep(self.cfg_eagle)
        self.map_step.map_configuration['root_path'] = '../test/data/eagle'
        data_loader = DataLoader(self.cfg_eagle)
        self.pipeline_eagle = Pipeline(cfg=self.cfg_eagle, steps=[data_loader, rotate_step, self.map_step])
        # Execute the pipeline and capture the first 5 iterations
        self.pipeline_eagle.run(n=1)

