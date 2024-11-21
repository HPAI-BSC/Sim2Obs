import os
import warnings
import hydra
import numpy as np
import rootutils
from omegaconf import DictConfig

from src.data_generation.src.data_fetcher import DataFetcherFactory
from src.data_generation.src.map import create_maps_and_zooms
from src.data_generation.src.rotate import apply_rotation
from src.data_generation.src.utils import label_particles, get_icl_mask

warnings.filterwarnings("ignore")
rootutils.setup_root(__file__, pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../../../configs", config_name="data_generation.yaml")
def main(cfg: DictConfig) -> None:
    # Use Hydra configuration
    log.info(f'Simulation selected: {cfg.catalog}')
    fetcher = DataFetcherFactory.get_data_fetcher(cfg)

    halo_context = fetcher.get_halos()
    log.info(f'Number of halos in the snapshot: {len(halo_context.r200)}')

    subhalo_context = fetcher.get_subhalos()
    log.info(f"Number of subhalos in the snapshot: {len(subhalo_context.spos)}")
    log.debug(f'Number of unique halo ids inside subhalos data: {len(np.unique(subhalo_context.id_parents))}')

    # Change the coordinate system origin to center the subhalo with respect to the halo coordinate system.
    last_halo_id = halo_context.halo_id[-1]
    for parent_id in subhalo_context.id_parents:
        try:
            if parent_id <= last_halo_id:
                _, _, gpos, _ = halo_context.get_item(halo_id=parent_id)
                subhalo_context.spos = subhalo_context.spos - gpos
        except Exception as e:
            print(e)
            continue

    for id_halo in halo_context.halo_id:
        log.debug(" ")
        log.debug(f'Halo id: {id_halo}')
        _, _, gpos_halo, id_subhalo = halo_context.get_item(id_halo)
        spos_h, shmr_h, mstr_h = subhalo_context.get_items_by_halo(id_halo)

        if cfg.catalog.name == 'eagle' or (cfg.catalog.name == 'illustris-300' and
                                           id_subhalo != cfg.catalog.subhalo_error_id):
            print(id_subhalo)
            _, _, m_subhalo = subhalo_context.get_item(id_subhalo)
            # Same units 𝑀⊙
            if m_subhalo > cfg.subhalo_min_mass:
                try:
                    subhalo_particles = fetcher.get_p_subhalo(idsubhalo=id_subhalo)
                    log.debug(f'Primary subhalo {id_subhalo} has {len(subhalo_particles.m)} particles.')
                except Exception as e:
                    log.warning(e)
                    continue

                # Defining a new origin
                subhalo_particles.p -= gpos_halo
                _, df_icl = label_particles(subhalo_particles, halo_context, id_halo, cfg.G)

                if len(cfg.rotation_group) > 0:
                    apply_rotation(cfg.rotation_group, subhalo_particles, halo_context, df_icl, spos_h, shmr_h,
                                   id_halo, id_subhalo, cfg)
                else:
                    # Create a masking array for BCG and ICL (BCG==0, ICL==1)
                    type4 = get_icl_mask(df_icl, subhalo_particles.m)
                    create_maps_and_zooms(halo_context, subhalo_particles, id_halo, id_subhalo, spos_h, shmr_h,
                                          type4, cfg, rotation=None)


if __name__ == "__main__":
    # TODO just for local
    os.environ['PROJECT_ROOT'] = '../../..'
    main()

