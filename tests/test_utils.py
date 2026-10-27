from omegaconf import OmegaConf

from src.utils.utils import set_model_channels_from_data


def _cfg(mask_interlopers=True, **model):
    data = {} if mask_interlopers is None else {"mask_interlopers": mask_interlopers}
    return OmegaConf.create({"data": data, "model": dict(model)})


def test_mask_interlopers_false_sets_two_channels():
    """All known in/out channel key names, at any nesting depth, are overwritten to 2."""
    cfg = _cfg(
        False,
        net={
            "_target_": "some.Net",
            "config": {"in_chans": 1},
            "num_classes": 1,
            "n_heads": 4,
        },
        other={"n_channels": 1, "n_classes": 1},
        another={"in_channels": 1, "out_channels": 1},
        interpolator={"input_channels": 1, "base_channels": 32},
    )
    set_model_channels_from_data(cfg)

    assert cfg.model.net.config.in_chans == 2
    assert cfg.model.net.num_classes == 2
    assert cfg.model.net.n_heads == 4  # unrelated keys untouched
    assert cfg.model.other.n_channels == 2
    assert cfg.model.other.n_classes == 2
    assert cfg.model.another.in_channels == 2
    assert cfg.model.another.out_channels == 2
    assert cfg.model.interpolator.input_channels == 2
    assert cfg.model.interpolator.base_channels == 32  # unrelated keys untouched


def test_mask_interlopers_true_leaves_config_untouched():
    cfg = _cfg(True, net={"n_channels": 1, "n_classes": 1})
    set_model_channels_from_data(cfg)
    assert cfg.model.net.n_channels == 1
    assert cfg.model.net.n_classes == 1


def test_mask_interlopers_absent_leaves_config_untouched():
    cfg = _cfg(None, net={"n_channels": 1, "n_classes": 1})
    set_model_channels_from_data(cfg)
    assert cfg.model.net.n_channels == 1
    assert cfg.model.net.n_classes == 1
