from typing import Any, List
import hydra
import pandas as pd
from hydra import initialize, compose
# from dotenv import load_dotenv, find_dotenv
from omegaconf import DictConfig
# from rootutils import rootutils

from src.data_generation.src.utils import setup_logger
import illustris_python as il

logger = setup_logger("GenerateDataset")

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

    # For debugging purposes
    debug_halo_ids = [724423, 1080849, 1500213]
    starting_idx = None
    ending_idx = None
    _ = pipeline.run(starting_idx=starting_idx, ending_idx=ending_idx, debug_halo_ids=debug_halo_ids)


if __name__ == "__main__":
    # main()
    with initialize(config_path="../configs"):
        cfg_tngcluster = compose(config_name="data_generation.yaml")

    path = cfg_tngcluster.catalog.path
    print(path)
    halo_fields = ['Group_M_Crit200', 'Group_R_Crit200', 'GroupFirstSub']
    halos = il.groupcat.loadHalos(path, 84, fields=halo_fields)
    df = pd.DataFrame(halos)
    print(f'Illustris (raw) halos: {len(df)}')

    h = 0.6774
    mass_filter_final_units = 1e13 * h  # Msun/h

    df['Group_M_Crit200'] = df['Group_M_Crit200'] * 1e10  # from 10^10 Msun/h to Msun/h
    filtered_df = df[df['Group_M_Crit200'] > mass_filter_final_units]
    print(len(filtered_df))

    subhalo_fields = ['SubhaloMassType', 'SubhaloGrNr']
    subhalos = il.groupcat.loadSubhalos(path, 84, fields=subhalo_fields)
    subhalos['SubhaloMassType'] = subhalos['SubhaloMassType'][:, 4]
    subhalos_df = pd.DataFrame(subhalos)
    subhalos_df['SubhaloMassType'] = subhalos_df['SubhaloMassType'] * 1e10  # from 10^10 Msun/h to Msun/h

    mass_filter = 1e10  # Msun
    mass_filter_units = 1e10 * h  # Msun / h

    filtered_subhalos = subhalos_df[subhalos_df['SubhaloMassType'] > mass_filter_units]
    print(len(filtered_subhalos))