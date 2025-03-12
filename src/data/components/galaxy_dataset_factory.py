from src.data.components.combined_dataset import CombinedDataset
from src.data.components.dianoga_dataset import DianogaDataset
from src.data.components.galaxy_dataset import GalaxyDataset
from src.data.components.illustris300_dataset import Illustris300Dataset


class GalaxyDatasetFactory:
    @staticmethod
    def get_dataset(dataset: str, **kwargs) -> GalaxyDataset:
        match dataset:
            case "dianoga":
                return DianogaDataset(**kwargs)
            case "illustris-300":
                return Illustris300Dataset(**kwargs)
            case "combined":
                return CombinedDataset(**kwargs)
        raise ValueError(f"Invalid feature name {dataset} provided")
