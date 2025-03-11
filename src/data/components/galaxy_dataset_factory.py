from src.data.components.combined_dataset import CombinedDataset
from src.data.components.dianoga_dataset import DianogaDataset
from src.data.components.galaxy_dataset import GalaxyDataset
from src.data.components.illustris300_dataset import Illustris300Dataset


class GalaxyDatasetFactory:
    @staticmethod
    def get_dataset(dataset: str, **kwargs) -> GalaxyDataset:
        datasets = {
            'dianoga': DianogaDataset(**kwargs),
            'illustris-300': Illustris300Dataset(**kwargs),
            #'combined': CombinedDataset(**kwargs),
        }

        if dataset not in datasets:
            raise ValueError(f"Invalid feature name {dataset} provided")
        return datasets[dataset]
