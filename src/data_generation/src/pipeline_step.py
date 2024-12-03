from abc import ABC, abstractmethod

from src.data_generation.src.cosmos import CosmosData
from src.data_generation.src.feature import FeatureProcessor


class PipelineStep(ABC):
    @abstractmethod
    def process(self, input_feature: FeatureProcessor, output_feature: FeatureProcessor, cosmos_data: CosmosData):
        pass