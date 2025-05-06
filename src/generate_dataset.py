from typing import Any, List
import hydra
# from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig
from rootutils import rootutils

from data_generation.src.data_loader_step import DataLoader
from data_generation.src.map_step import MapStep
from data_generation.src.pipeline import Pipeline
from data_generation.src.pipeline_step import PipelineStep
from data_generation.src.rotate_step import RotateStep

rootutils.setup_root(__file__, pythonpath=True)

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
    steps = get_pipeline_steps(cfg)
    pipeline = Pipeline(cfg=cfg, steps=steps)
    # Execute the pipeline and capture the first 5 iterations
    _ = pipeline.run()


if __name__ == "__main__":
    main()