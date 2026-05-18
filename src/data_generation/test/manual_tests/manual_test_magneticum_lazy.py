import tracemalloc
import gc
from rootutils import rootutils

from src.data_generation.src.data_fetcher import DataFetcherFactory

rootutils.setup_root(__file__, pythonpath=True)

from hydra import initialize, compose


def magneticum_particle_test():
    with initialize(config_path="'../../../../../../configs"):
        cfg_magneticum = compose(config_name="data_generation.yaml")

        # Load the local config for paths and assign them
        local_config = compose(config_name="paths/local.yaml")
        cfg_magneticum.paths = local_config.paths
        cfg_magneticum.catalog = compose("catalog/magneticum.yaml")['catalog']

    required_keys = ['G']
    data_loader_dict = next((step for step in cfg_magneticum.pipeline if step.name == 'data_loader'), None)
    attr_dict = {key: data_loader_dict[key] for key in required_keys if key in data_loader_dict}
    fetcher_magneticum = DataFetcherFactory.get_data_fetcher(cfg_magneticum.catalog, **attr_dict)

    tracemalloc.start()

    halo_ids = [0, 1, 2, 3, 4]  # your test halo ids

    halo_context = fetcher_magneticum.get_halos()
    subhalo_context = fetcher_magneticum.get_subhalos()

    for halo_id in halo_ids:
        halo = halo_context.get_halo_by_id(halo_id)
        particles = fetcher_magneticum.get_p_halo(halo=halo)

        current, peak = tracemalloc.get_traced_memory()
        print(f"halo_id={halo_id} | stars={len(particles)} | "
              f"mem current={current/1e6:.1f} MB | peak={peak/1e6:.1f} MB")

        del particles
        gc.collect()

    tracemalloc.stop()


if __name__ == "__main__":
    # TODO DEBUG
    magneticum_particle_test()
