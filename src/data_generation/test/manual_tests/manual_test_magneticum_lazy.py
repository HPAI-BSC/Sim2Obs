import tracemalloc
import gc
#from rootutils import rootutils

from src.data_generation.src.data_fetcher import DataFetcherFactory

#rootutils.setup_root(__file__, pythonpath=True)

from hydra import initialize, compose


def magneticum_particle_test():
    with initialize(config_path="'../../../../../../configs"):
        cfg_magneticum = compose(config_name="data_generation.yaml")

        # Load the local config for paths and assign them
        local_config = compose(config_name="paths/gpfs.yaml")
        cfg_magneticum.paths = local_config.paths
        cfg_magneticum.catalog = compose("catalog/magneticum.yaml")['catalog']

    required_keys = ['G']
    data_loader_dict = next((step for step in cfg_magneticum.pipeline if step.name == 'data_loader'), None)
    attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
    fetcher_magneticum = DataFetcherFactory.get_data_fetcher(cfg_magneticum.catalog, **attr_dict)

    tracemalloc.start()
    halo_context = fetcher_magneticum.get_halos()
    subhalo_context = fetcher_magneticum.get_subhalos()
    tracemalloc.stop()


if __name__ == "__main__":
    # TODO DEBUG
    magneticum_particle_test()
