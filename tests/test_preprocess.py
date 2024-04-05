import os
from pathlib import Path

import pytest
from omegaconf import DictConfig

from src.preprocessing.mnist_preprocess import Preprocessor
from tests.helpers.test_step import TestStep


@pytest.fixture
def step():
    return TestStep()


@pytest.fixture
def prep(tmp_path: Path):
    source_dir = "tests/images"
    target_dir = os.path.join(tmp_path, "test-0")
    return Preprocessor(source_dir, target_dir)


def test_add_steps(prep: Preprocessor, step: TestStep) -> None:
    assert len(prep.steps) == 0
    prep.add_steps([step])
    assert len(prep.steps) == 1


def test_add_step(prep: Preprocessor, step: TestStep) -> None:
    assert len(prep.steps) == 0
    prep.add_step(step)
    assert len(prep.steps) == 1


def test_create_target_dataset(tmp_path: Path, prep: Preprocessor) -> None:
    source_dir = "tests/images"
    target_dir = os.path.join(tmp_path, "test-0")
    assert os.path.exists(target_dir) == False
    prep.run()
    prep.target_dataset.load_data()
    assert os.path.exists(target_dir) == True
    assert len(os.listdir(target_dir)) == len(os.listdir(source_dir))


def test_apply_test_preprocess(prep: Preprocessor, step: TestStep):
    prep.add_step(step)
    prep.run()
    prep.target_dataset.load_data()
    preprocessed_image = prep.target_dataset[0].img
    # test step returns zero_like(image)
    assert preprocessed_image.sum() == 0
