from src.data.components.combined_dataset import CombinedDataset
from src.data.components.dianoga_dataset import DianogaDataset
from src.data.components.galaxy_dataset import GalaxyDataset
from src.data.components.illustris300_dataset import Illustris300Dataset
from src.data.components.eagle_dataset import EagleDataset
from src.data.components.flamingo_dataset import FlamingoDataset
from src.data.components.obelisk_dataset import ObeliskDataset


class GalaxyDatasetFactory:
    @staticmethod
    def get_dataset(dataset: str, **kwargs) -> GalaxyDataset:
        match dataset:
            case "dianoga":
                return DianogaDataset(**kwargs)
            case "illustris-300":
                return Illustris300Dataset(**kwargs)
            case "eagle":
                return EagleDataset(**kwargs)
            case "combined" | "combined_vd_dm" | "combined_dm":
                return CombinedDataset(**kwargs)
            case "flamingo":
                return FlamingoDataset(**kwargs)
            case "obelisk":
                return ObeliskDataset(**kwargs)
        raise ValueError(f"Invalid feature name {dataset} provided")
