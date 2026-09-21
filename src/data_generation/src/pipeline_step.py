from abc import ABC, abstractmethod
from typing import Any, Callable


class PipelineStep(ABC):
    """
    Abstract base class for defining steps in a data pipeline.

    This class serves as a blueprint for creating specific steps in a data processing pipeline.
    Each step should implement the `process` method to process the input and output features,
    as well as the `cosmos_data`, and decide whether to call the next step in the pipeline.

    Attributes
    ----------
    None (abstract class)

    Methods
    -------
    ``process(g, next_step, **kwargs)``
        Processes the input and output features and decides whether to call the next step in the pipeline.
        This method should be implemented by subclasses to define specific processing logic.
    """
    @abstractmethod
    def process(self, g: Any, next_step: Callable[[Any], 'Any'], **kwargs):
        """
        Process the input and output features, along with cosmos data, and apply the next step in the pipeline.

        Parameters
        ----------
        g : Any
            The data or generator passed through the pipeline.
        next_step : Callable[[Any], 'Any']
            A callable that represents the next step in the pipeline. It should accept `g` and return the next value.

        Returns
        -------
        None
            This method does not return anything, as its purpose is to modify or pass data forward through the pipeline.
        """
        pass