from abc import ABC

from src.preprocessing.components.dataset import Sample


class PreprocessStep(ABC):
    """Abstract class that represents a Preprocessing Step."""

    @property
    def name(self) -> str:  # pragma: no cover
        return self.__class__.__name__

    def apply(self, sample: Sample) -> None:
        raise NotImplementedError()
