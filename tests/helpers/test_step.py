import numpy as np

from src.preprocessing.components.dataset import Sample
from src.preprocessing.steps.generic import PreprocessStep


class TestStep(PreprocessStep):
    def __init__(self):
        pass

    def apply(self, sample: Sample) -> None:
        new_img = np.zeros_like(sample.img)
        sample.update_img(new_img)
