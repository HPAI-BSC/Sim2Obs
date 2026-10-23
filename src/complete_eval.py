from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import logging
import sys

import hydra
import rootutils
from hydra.core.global_hydra import GlobalHydra
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf, open_dict

rootutils.setup_root(__file__, pythonpath=True)

from src import utils
from src.eval import evaluate

log = utils.get_pylogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
TRAIN_CONFIG_REL_PATH = Path("configs") / "train.yaml"

# config keys owned by the evaluation loop; user overrides of these are not propagated to run
# configs, everything else (e.g. `trainer=cpu`, `trainer.limit_test_batches=0.1`) is
LOOP_OWNED_KEYS = ("data", "model", "ckpt_path", "ckpt_paths", "data_names", "models", "task_name")


@contextmanager
def hydra_config_dir() -> Iterator[None]:
    """Initializes Hydra with the project configs if it is not initialized yet."""
    if GlobalHydra.instance().is_initialized():
        yield
    else:
        with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
            yield


def extract_run_overrides(overrides: Optional[List[str]]) -> List[str]:
    """Filters user overrides down to those safe to propagate to each run config.

    Overrides of keys owned by the evaluation loop (checkpoints, datasets, models, ...) are
    dropped, since the loop sets them per run.
    """
    run_overrides = []
    for override in OverridesParser.create().parse_overrides(overrides or []):
        key = override.key_or_group
        if key in LOOP_OWNED_KEYS or key.startswith("paths.output_dir"):
            continue
        run_overrides.append(override.input_line)
    return run_overrides


def find_run_dir(ckpt_path: str) -> Path:
    """Walks up from a checkpoint file to its run directory.

    Checkpoints are stored in a full copy of the repo, e.g.
    ``<run_dir>/output/checkpoints/<commit_tag>/epoch_049.ckpt``, so the run directory is the
    closest ancestor containing ``configs/train.yaml``.
    """
    ckpt = Path(ckpt_path)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found! <ckpt_path={ckpt_path}>\n"
            "Make sure the checkpoint path in `ckpt_paths` is correct."
        )
    for parent in ckpt.parents:
        if (parent / TRAIN_CONFIG_REL_PATH).is_file():
            return parent
    raise ValueError(
        f"Could not find a run directory with '{TRAIN_CONFIG_REL_PATH}' for checkpoint "
        f"<{ckpt_path}>\n"
        "The checkpoint is expected to live inside a full copy of the repo. If that is not the "
        "case, provide the model architectures explicitly in the `models` list of "
        "configs/complete_eval.yaml, one per checkpoint, in the same order as `ckpt_paths`."
    )


def extract_model_name(ckpt_path: str) -> str:
    """Extracts the model architecture used to train a checkpoint.

    Reads ``defaults.model`` from ``<run_dir>/configs/train.yaml``, where ``<run_dir>`` is the
    repo copy the checkpoint was trained in.
    """
    train_cfg = OmegaConf.load(find_run_dir(ckpt_path) / TRAIN_CONFIG_REL_PATH)
    for item in OmegaConf.to_container(train_cfg).get("defaults", []):
        if isinstance(item, dict) and "model" in item:
            model_name = item["model"]
            log.info(f"Inferred model <{model_name}> from checkpoint <{ckpt_path}>")
            return model_name
    raise ValueError(
        f"No 'model' entry found in the defaults of '{TRAIN_CONFIG_REL_PATH}' "
        f"for checkpoint <{ckpt_path}>"
    )


def resolve_model_names(cfg: DictConfig, ckpt_paths: List[str]) -> List[str]:
    """Resolves the model architecture of every checkpoint.

    If `models` is set in the config, it is used as is (one entry per checkpoint, same order).
    Otherwise, the architecture of every checkpoint is inferred from its run directory.
    """
    explicit = list(cfg.get("models") or [])
    if explicit:
        if len(explicit) != len(ckpt_paths):
            raise ValueError(
                f"Number of explicit models <{len(explicit)}> does not match the number of "
                f"checkpoints <{len(ckpt_paths)}>! `models` and `ckpt_paths` must have the same "
                "length (one model per checkpoint, in the same order), or `models` must be empty "
                "to infer the architectures from the checkpoints."
            )
        log.info(f"Using {len(explicit)} explicitly provided model architectures")
        return explicit
    return [extract_model_name(ckpt_path) for ckpt_path in ckpt_paths]


def run_label(ckpt_path: str) -> str:
    """Builds a unique label for a checkpoint: ``<run_dir_name>_<ckpt_file_stem>``."""
    ckpt = Path(ckpt_path)
    try:
        run_name = find_run_dir(ckpt_path).name
    except (FileNotFoundError, ValueError):
        run_name = ckpt.parent.name
    return f"{run_name}_{ckpt.stem}"


@utils.task_wrapper
def complete_evaluate(cfg: DictConfig) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    """Evaluates every checkpoint in `ckpt_paths` on every dataset in `data_names`.

    :param cfg: DictConfig configuration composed by Hydra. Optionally carries
        `complete_eval_overrides` (the list of Hydra-style overrides the config was composed
        with); all overrides except the ones owned by the loop are propagated to every run
        config (e.g. `paths=local`, `trainer=cpu`, `trainer.limit_test_batches=0.1`).
    :return: Tuple with a dict of metric dicts keyed by (run label, dataset name) and a dict with
        the base configuration.
    """
    ckpt_paths = list(cfg.ckpt_paths)
    data_names = list(cfg.data_names)
    if not ckpt_paths:
        raise ValueError("`ckpt_paths` must contain at least one checkpoint!")
    if not data_names:
        raise ValueError("`data_names` must contain at least one dataset!")
    model_names = resolve_model_names(cfg, ckpt_paths)

    log.info(
        f"Evaluating {len(ckpt_paths)} checkpoint(s) on {len(data_names)} dataset(s) "
        f"({len(ckpt_paths) * len(data_names)} runs in total)"
    )
    log.info("Resolved checkpoints and their model architectures:")
    for ckpt_path, model_name in zip(ckpt_paths, model_names):
        log.info(f"  {run_label(ckpt_path)} -> model <{model_name}>  ({ckpt_path})")

    run_user_overrides = extract_run_overrides(cfg.get("complete_eval_overrides"))
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    models_by_label = {run_label(ckpt): model for ckpt, model in zip(ckpt_paths, model_names)}
    with hydra_config_dir():
        for ckpt_path, model_name in zip(ckpt_paths, model_names):
            label = run_label(ckpt_path)
            for data_name in data_names:
                output_dir = Path("output") / "complete_eval" / label / data_name
                output_dir.mkdir(parents=True, exist_ok=True)
                log.info(
                    f"Evaluating model <{model_name}> on dataset <{data_name}> "
                    f"(checkpoint <{ckpt_path}>)"
                )

                run_cfg = hydra.compose(
                    config_name="complete_eval",
                    overrides=[
                        f"data={data_name}",
                        f"model={model_name}",
                        f"ckpt_path={ckpt_path}",
                        "task_name=complete_eval",
                        f"paths.output_dir={output_dir.as_posix()}",
                        # these rely on HydraConfig/`hydra:` resolvers only available in a
                        # regular @hydra.main run
                        "extras.print_config=False",
                        "extras.enforce_tags=False",
                    ]
                    + run_user_overrides,
                )
                metric_dict, _ = evaluate(run_cfg)
                results[(label, data_name)] = metric_dict

    log.info("Complete evaluation finished! Results:")
    for (label, data_name), metric_dict in results.items():
        model_name = models_by_label[label]
        log.info(f"[{label} | model <{model_name}> | {data_name}] {dict(metric_dict)}")

    return results, {"cfg": cfg}


def main() -> None:
    """Main entry point for complete evaluation over multiple checkpoints and datasets.

    Accepts optional Hydra-style command line overrides, e.g.:
        python src/complete_eval.py paths=local data_names=[galaxy_flamingo,galaxy_dianoga]
    """
    # without @hydra.main, no one configures logging for us
    logging.basicConfig(level=logging.INFO)
    with hydra_config_dir():
        cfg = hydra.compose(config_name="complete_eval", overrides=sys.argv[1:])
    with open_dict(cfg):
        cfg.complete_eval_overrides = list(sys.argv[1:])
    complete_evaluate(cfg)


if __name__ == "__main__":
    main()
