import os
from typing import Any

from hydra import initialize, compose
from omegaconf import DictConfig

from src.data_generation.src.data_loader_step import DataLoader
from src.data_generation.src.feature import load_models
from src.data_generation.src.last_step import LastStep
from src.data_generation.src.map_step import MapStep
from src.data_generation.src.pipeline_step import PipelineStep
from src.data_generation.src.utils import setup_logger

logger = setup_logger("TestPipeline")


def get_generator(data_loader, n=10):
    halo_context = data_loader.fetcher.get_halos()
    subhalo_context = data_loader.fetcher.get_subhalos()

    # Clean subhalo context to remove subhalos that belong to halos that do not exist
    subhalo_context.remove_subhalos_belonging_unvalid_halos(halo_context)
    overlapping_matrix = data_loader.get_overlapping_halos(subhalo_context, halo_context)
    halo_subhalo_list = data_loader.load_halo_subhalo_data(subhalo_context, halo_context, overlapping_matrix)

    halo_subhalo_list = halo_subhalo_list[:n]
    for halo_data in halo_subhalo_list:
        yield halo_data  # T


class Pipeline:
    def __init__(self, cfg: DictConfig, steps: [PipelineStep]):
        self.cfg = cfg
        self.steps = steps

        for step in self.steps:
            if step.name == 'data_loader':
                self.data_loader = step
                break
        else:
            self.data_loader = DataLoader(self.cfg)

    def build_pipeline(self, steps):
        """
        Build and return a pipeline function that applies all steps recursively.
        The returned function takes a generator (or iterable) and applies the steps on it.
        """
        logger.info('build_pipeline: creating pipeline function')

        def pipeline(g, first_step_kwargs=None, **shared_kwargs):
            first_step_kwargs = first_step_kwargs or {}

            def apply_steps(i, gen, **kw):
                if i == len(steps):
                    return gen
                current_step = steps[i]
                logger.debug(f'Applying step {i + 1}/{len(steps)}: {current_step.name}')

                current_kwargs = first_step_kwargs if i == 0 else kw

                # Determine if this is the last step:
                is_last_step = (i == len(steps) - 1)

                # Remove is_last_step from current_kwargs if present
                current_kwargs = dict(current_kwargs)  # copy to avoid side effects
                current_kwargs.pop('is_last_step', None)  # remove if present

                return current_step.process(
                    gen,
                    None if is_last_step else lambda g, **k: apply_steps(i + 1, g, **k),
                    is_last_step=is_last_step,
                    **current_kwargs
                )

            return apply_steps(0, g, **shared_kwargs)

        return pipeline

    def run(self, n=None) -> Any:
        if not self.steps:
            logger.error("Error, please provide a non-empty pipeline, steps cannot be an empty list.")
            return None

        pipeline_fn = self.build_pipeline(self.steps)

        # get dataloader step
        step_list = [step for step in self.steps if step.name == 'data_loader']

        if len(step_list) > 0:
            data_loader = step_list[0]
            input_gen = get_generator(data_loader, n)
        else:
            raise Exception()

        halo_context = data_loader.fetcher.get_halos()
        models = load_models(self.cfg.feature_names)

        # Run the pipeline function
        result = list(pipeline_fn(input_gen, first_step_kwargs={'halo_context': halo_context, 'models': models}))

        final_step = LastStep(self.cfg)
        final_step.process(result)

        logger.info("Pipeline processing completed.")
        return result


if __name__ == '__main__':
    with initialize(config_path="../../../../configs", version_base="1.3"):
        cfg_illustris = compose(config_name="data_generation.yaml")
        local_config = compose(config_name="paths/local.yaml")
        cfg_illustris.paths = local_config.paths
        cfg_illustris.catalog = compose("catalog/illustris300.yaml")['catalog']

    os.environ['PROJECT_ROOT'] = '../../../..'

    # This is done by pipeline class steps = get_pipeline_steps(cfg)
    map_step = MapStep(cfg_illustris)
    map_step.map_configuration['root_path'] = '../test/data/illustris'
    data_loader = DataLoader(cfg_illustris)
    final_step = LastStep(cfg_illustris)
    steps = [data_loader, map_step]

    pipeline_illustris = Pipeline(cfg=cfg_illustris, steps=steps)
    pipeline_illustris.run(n=10)
