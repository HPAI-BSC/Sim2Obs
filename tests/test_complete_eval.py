from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.complete_eval import (
    complete_evaluate,
    extract_model_name,
    extract_run_overrides,
    find_run_dir,
    resolve_model_names,
    run_label,
)
from src.train import train


def test_extract_run_overrides() -> None:
    assert extract_run_overrides(None) == []
    assert extract_run_overrides(["save_predictions=False"]) == ["save_predictions=False"]
    assert extract_run_overrides(["paths=local", "trainer=cpu"]) == [
        "paths=local",
        "trainer=cpu",
    ]
    # loop-owned overrides are not propagated
    assert extract_run_overrides(["data=galaxy_dianoga", "ckpt_path=foo.ckpt"]) == []
    assert extract_run_overrides(["ckpt_paths=[a,b]", "data_names=[x]", "models=[m1,m2]"]) == []
    assert extract_run_overrides(["paths.output_dir=foo", "task_name=bar"]) == []


def _make_fake_run_dir(base: Path, run_id: str = "0534d449", model: str = "mamba_unet") -> Path:
    """Creates a fake repo copy with a checkpoint and a train config defining `model`."""
    run_dir = base / run_id
    ckpt = run_dir / "output" / "checkpoints" / run_id / "epoch_049.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    train_cfg = OmegaConf.create(
        {
            "defaults": [
                {"data": "combined_vd_icl"},
                {"model": model},
                {"trainer": "default"},
            ],
            "task_name": "train",
        }
    )
    configs_dir = run_dir / "configs"
    configs_dir.mkdir()
    OmegaConf.save(train_cfg, configs_dir / "train.yaml")
    return run_dir


def test_find_run_dir(tmp_path: Path) -> None:
    run_dir = _make_fake_run_dir(tmp_path)
    ckpt_path = str(run_dir / "output" / "checkpoints" / run_dir.name / "epoch_049.ckpt")
    assert find_run_dir(ckpt_path) == run_dir


def test_find_run_dir_missing_ckpt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_run_dir(str(tmp_path / "does_not_exist.ckpt"))


def test_find_run_dir_no_train_config(tmp_path: Path) -> None:
    ckpt = tmp_path / "output" / "checkpoints" / "abc123" / "epoch_049.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    with pytest.raises(ValueError, match="models"):
        find_run_dir(str(ckpt))


def test_extract_model_name(tmp_path: Path) -> None:
    run_dir = _make_fake_run_dir(tmp_path, run_id="abc123", model="attention_unet")
    ckpt_path = str(run_dir / "output" / "checkpoints" / "abc123" / "epoch_049.ckpt")
    assert extract_model_name(ckpt_path) == "attention_unet"


def test_extract_model_name_no_model_entry(tmp_path: Path) -> None:
    run_dir = tmp_path / "abc123"
    ckpt = run_dir / "output" / "checkpoints" / "abc123" / "epoch_049.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.touch()
    (run_dir / "configs").mkdir()
    OmegaConf.save(
        OmegaConf.create({"defaults": [{"data": "combined_vd_icl"}]}),
        run_dir / "configs" / "train.yaml",
    )
    with pytest.raises(ValueError, match="model"):
        extract_model_name(str(ckpt))


def test_resolve_model_names_inferred(tmp_path: Path) -> None:
    run_dir_a = _make_fake_run_dir(tmp_path, run_id="run_a", model="unet")
    run_dir_b = _make_fake_run_dir(tmp_path, run_id="run_b", model="transunet")
    ckpt_a = str(run_dir_a / "output" / "checkpoints" / "run_a" / "epoch_049.ckpt")
    ckpt_b = str(run_dir_b / "output" / "checkpoints" / "run_b" / "epoch_049.ckpt")
    cfg = OmegaConf.create({"models": []})
    assert resolve_model_names(cfg, [ckpt_a, ckpt_b]) == ["unet", "transunet"]


def test_resolve_model_names_explicit(tmp_path: Path) -> None:
    run_dir = _make_fake_run_dir(tmp_path, run_id="run_a", model="unet")
    ckpt = str(run_dir / "output" / "checkpoints" / "run_a" / "epoch_049.ckpt")
    cfg = OmegaConf.create({"models": ["attention_unet"]})
    assert resolve_model_names(cfg, [ckpt]) == ["attention_unet"]


def test_resolve_model_names_explicit_length_mismatch(tmp_path: Path) -> None:
    run_dir = _make_fake_run_dir(tmp_path, run_id="run_a", model="unet")
    ckpt = str(run_dir / "output" / "checkpoints" / "run_a" / "epoch_049.ckpt")
    cfg = OmegaConf.create({"models": ["unet", "transunet"]})
    with pytest.raises(ValueError, match="does not match"):
        resolve_model_names(cfg, [ckpt])


def test_run_label(tmp_path: Path) -> None:
    run_dir = _make_fake_run_dir(tmp_path, run_id="0534d449")
    ckpt_path = str(run_dir / "output" / "checkpoints" / "0534d449" / "epoch_049.ckpt")
    assert run_label(ckpt_path) == "0534d449_epoch_049"


def test_compose_complete_eval() -> None:
    """Smoke test that complete_eval.yaml composes with per-run data/model/ckpt overrides."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="complete_eval.yaml",
            overrides=[
                "paths=local",
                "data=galaxy_flamingo",
                "model=unet",
                "ckpt_path=.",
                "extras.print_config=False",
            ],
        )
        assert cfg.task_name == "complete_eval"
        assert cfg.data._target_ == "src.data.galaxy_datamodule.GalaxyDataModule"
        assert "_target_" in cfg.model
        assert cfg.ckpt_path == "."
        assert len(cfg.ckpt_paths) >= 1
        assert len(cfg.data_names) >= 1
    GlobalHydra.instance().clear()


@pytest.mark.slow
def test_train_complete_eval(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Trains for 1 epoch with `train.py` and evaluates the checkpoint with `complete_eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    ckpt = Path(cfg_train.paths.output_dir) / "checkpoints" / "commit_hash" / "last.ckpt"
    assert ckpt.is_file()

    data_name = cfg_train.hydra.choices.data
    model_name = cfg_train.hydra.choices.model
    overrides = [
        "paths=local",
        "trainer=cpu",
        f'ckpt_paths=["{ckpt}"]',
        f"data_names=[{data_name}]",
        f"models=[{model_name}]",
        "save_predictions=False",
        "extras.print_config=False",
        "extras.enforce_tags=False",
    ]
    with initialize(version_base="1.3", config_path="../configs"):
        ce_cfg = compose(config_name="complete_eval.yaml", overrides=overrides)
    with open_dict(ce_cfg):
        ce_cfg.complete_eval_overrides = overrides

    results, _ = complete_evaluate(ce_cfg)

    assert len(results) == 1
    (label, evaluated_data), metric_dict = next(iter(results.items()))
    assert label.endswith("_last")
    assert evaluated_data == data_name
    assert metric_dict["test/acc"] > 0.0
    GlobalHydra.instance().clear()
