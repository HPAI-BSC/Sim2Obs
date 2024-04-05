import multiprocessing
from typing import Any, Dict, List, Tuple

import hydra
import lightning as L
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src import utils
from src.preprocessing.preprocessing_manager import Preprocessor

log = utils.get_pylogger(__name__)


def preprocess(cfg: DictConfig):
    """Preprocess the dataset found in input_tag attribute applying the selected steps and saves
    the result in This method is wrapped in optional @task_wrapper decorator, that controls the
    behavior during failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating preprocessing <{cfg.preprocessing._target_}>")
    prep: Preprocessor = hydra.utils.instantiate(cfg.preprocessing)
    steps = [hydra.utils.instantiate(step) for _, step in cfg.preprocessing_steps.items()]
    prep.add_steps(steps)

    if cfg.parallel <= 0:
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = min(multiprocessing.cpu_count(), cfg.parallel)

    log.info(f"Starting preprocessing!")
    prep.run(num_processes)


@hydra.main(version_base="1.3", config_path="../configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    """Main entry point for preprocessing.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)
    preprocess(cfg)


if __name__ == "__main__":
    main()
