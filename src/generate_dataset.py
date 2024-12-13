from typing import Any, List
import hydra
from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig

from data_generation.src.data_loader_step import DataLoader
from data_generation.src.map_step import MapStep
from data_generation.src.pipeline import Pipeline
from data_generation.src.pipeline_step import PipelineStep
from data_generation.src.rotate_step import RotateStep

dotenv_path = find_dotenv('data_generation/.env')
if dotenv_path:
    load_dotenv(dotenv_path)
else:
    print(".env file not found, skipping load.")


def get_pipeline_steps(cfg: DictConfig) -> List[PipelineStep]:
    if 'pipeline' not in cfg:
        raise ValueError('')

    pipeline_steps = {
        'data_loader': DataLoader(cfg),
        'rotate_step': RotateStep(cfg),
        'map_step':  MapStep(cfg)
        # Add more steps if required
    }

    pipeline = []

    for stage in cfg['pipeline']:
        name = stage['name']
        if name not in pipeline_steps:
            raise ValueError(f'Error, the step {name} does not exist, please add the step in the pipeline steps'
                             f' dictionary.')
        pipeline.append(pipeline_steps[name])
    return pipeline


@hydra.main(version_base="1.3", config_path="../configs", config_name="data_generation.yaml")
def main(cfg: DictConfig) -> Any:
    steps = get_pipeline_steps(cfg)
    print(f'Launching pipeline: {steps}')
    pipeline = Pipeline(cfg=cfg, steps=steps)
    # Execute the pipeline and capture the first 5 iterations
    result = pipeline.run(n=1)
    print(result)


if __name__ == "__main__":
    main()