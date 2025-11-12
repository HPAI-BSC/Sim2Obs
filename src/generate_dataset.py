import logging
from typing import Any, List
import hydra
# from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig
# from rootutils import rootutils

from src.data_generation.src.utils import setup_logger

logger = setup_logger("Generate-dataset")


from data_generation.src.data_loader_step import DataLoader
from data_generation.src.map_step import MapStep
from data_generation.src.pipeline import Pipeline
from data_generation.src.pipeline_step import PipelineStep
from data_generation.src.rotate_step import RotateStep

#rootutils.setup_root(__file__, pythonpath=True)

# Uncomment for local testing, internet connection required
# dotenv_path = find_dotenv('data_generation/.env')
# if dotenv_path:
#    load_dotenv(dotenv_path)
# else:
#    print(".env file not found, skipping load.")


def get_pipeline_steps(cfg: DictConfig) -> List[PipelineStep]:
    if 'pipeline' not in cfg:
        raise ValueError('Error, pipeline variable must be specified in the configuration file.')

    pipeline = []
    for item in cfg['pipeline']:
        if isinstance(item, DictConfig):  # Ensure the item is a DictConfig
            if "name" in item:
                if item["name"] == "data_loader":
                    pipeline.append(DataLoader(cfg))
                elif item['name'] == 'rotate_step':
                    pipeline.append(RotateStep(cfg))
                elif item['name'] == 'map_step':
                    pipeline.append(MapStep(cfg))
                # Add more steps if required
                else:
                    raise ValueError(f'Error, name={item["name"]} does not belonging to any stage of the'
                                     f' pipeline=[data_loader, rotate_step, map_step]')
    return pipeline


@hydra.main(version_base="1.3", config_path="../configs", config_name="data_generation.yaml")
def main(cfg: DictConfig) -> Any:
    import uuid
    run_id = uuid.uuid4()
    logger.info(f"Starting pipeline run: {run_id}")
    steps = get_pipeline_steps(cfg)
    pipeline = Pipeline(cfg=cfg, steps=steps)

    starting_idx = None
    ending_idx = None

    debug_halos = [3085827, 1955843, 262661, 8344073, 6977550, 5730831, 272405, 19067932, 7557662, 16152606, 19410492, 19211838, 500302, 15694414, 18959445, 9409115, 9338460, 19128925, 19532383, 7377505, 11672170, 19066989, 8026733, 1958512, 17790064, 19291766, 5642359, 10980472, 5915770, 3205249, 12369543, 17699976, 1618056, 15297166, 13419664, 14793880, 711338, 7945900, 17935023, 14792369, 4165303, 12017857, 10980547, 6714053, 13215432, 16428746, 13582029, 3852493, 19778255, 5823694, 19151572, 15164118, 12977885, 15161055, 13932264, 11844350, 14981888, 19447552, 4475649, 13212933, 9982222, 7948057, 11430176, 3411234, 10655532, 19427629, 19427634, 18272058, 18080066, 5642563, 14037829, 4268873, 10981724, 17700189, 7286622, 10856801, 7475557, 17964397, 15386484, 1291637, 10853255, 14793619, 14522261, 11845014, 7375774, 14931372, 16190894, 16903088, 15292337, 19333556, 16501172, 19212215, 7378361, 2701753, 19777473, 18804683, 15694286, 3747794, 13985235, 18271702, 15076314, 1110492, 15386078, 14338527, 4367840, 3961826, 10851299, 18981347, 10390504, 16710122, 16501739, 17315824, 2826226, 13984243, 5365750, 6547449]
    _ = pipeline.run(starting_idx=starting_idx, ending_idx=ending_idx, debug_halo_ids=debug_halos)


if __name__ == "__main__":
    main()