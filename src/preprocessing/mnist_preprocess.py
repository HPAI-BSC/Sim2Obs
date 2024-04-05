import logging
import multiprocessing
from typing import List

from src.preprocessing.components.dataset import PreprocessDataset
from src.preprocessing.io import IO
from src.preprocessing.steps.generic import PreprocessStep


class Preprocessor:
    def __init__(self, source_dir: str = "local/data/raw", target_dir: str = "local/data/frame-0"):
        self.source_dataset = PreprocessDataset(io=IO(), data_root_path=source_dir)
        self.source_dataset.load_data()
        self.target_dataset = PreprocessDataset(io=IO(), data_root_path=target_dir)
        self.steps = []

    def add_step(self, step: PreprocessStep) -> None:
        self.steps.append(step)

    def add_steps(self, steps: List[PreprocessStep]) -> None:
        self.steps += steps

    def run(self, processes: int = 1) -> None:
        if self.target_dataset.is_initialized():
            logging.info("Resuming preprocessing")

        if not self.target_dataset.is_initialized():
            self.target_dataset.init_empty_dataset()

        self.preprocess_dataset(processes)

    def preprocess_dataset(self, processes: int):
        self.preprocess_image(0)
        if processes > 1:
            print(f"Running preprocessing with {processes} processes")
            with multiprocessing.Pool(processes) as pool:
                _ = pool.map(self.preprocess_image, range(len(self.source_dataset)))
                pool.close()
                pool.join()
        else:
            for idx in range(len(self.source_dataset)):
                self.preprocess_image(idx)

    def preprocess_image(self, idx: int):
        image = self.source_dataset[idx]
        if self.target_dataset.check_persisted_image_is_valid(image.path):
            print(f"Skipping image: {idx}")
        else:
            for step in self.steps:
                step.apply(image)

            self.target_dataset.dump_image(image)
