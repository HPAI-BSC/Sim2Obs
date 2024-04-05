import numpy as np

from src.preprocessing.components.dataset import Sample
from src.preprocessing.steps.generic import PreprocessStep


class Frame(PreprocessStep):
    def __init__(self, frame_width: int = 4, frame_height: int = 2, frame_value: int = 255):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_val = frame_value

    def apply(self, sample: Sample) -> None:
        new_img = sample.img.copy()
        new_img[0 : self.frame_height, :] = self.frame_val
        new_img[-self.frame_height :, :] = self.frame_val
        new_img[:, 0 : self.frame_width] = self.frame_val
        new_img[:, -self.frame_width :] = self.frame_val
        sample.update_img(new_img)
