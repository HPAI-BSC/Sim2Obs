import numpy as np
import pandas as pd
import warnings

from map import create_maps_and_zooms, generate_img_dataset
from src.data_generation.src.constants import PATH, HALO_MIN_MASS, H, SUBHALO_ERROR_ID, SUBHALO_MIN_MASS, SNAPNUM
from src.data_generation.src.halo import get_halos
from src.data_generation.src.rotate import apply_rotation
from src.data_generation.src.subhalo import get_subhalos, get_p_subhalo
from src.data_generation.src.utils import setup_logger, label_particles, get_icl_mask

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    rotation_groups = [[45, [0, 0, 1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]]]
                        #[120, [0, 0, 1], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                        #[90, [0, 0, 1], [[1, 0, 0]]]]
    logger = setup_logger()

    halo_context = get_halos(PATH)
    logger.info(f'Number of halos in the snapshot: {len(halo_context.r200)}')
    subhalo_context = get_subhalos(PATH)
    logger.info(f"Number of subhalos in the snapshot: {len(subhalo_context.spos)}")
    subhalo_context.spos = subhalo_context.spos - halo_context.gpos[subhalo_context.id_parents]
    logger.debug(f'Number of unique halo ids inside subhalos data: {len(np.unique(subhalo_context.id_parents))}')

    for id_halo in range(len(halo_context.r200)):
        logger.debug(" ")
        logger.debug(f'Halo id: {id_halo}')
        if halo_context.m200[id_halo] > HALO_MIN_MASS * H:

            id_subhalo = halo_context.fsub[id_halo]
            id_parents = subhalo_context.id_parents

            spos_h = pd.DataFrame(subhalo_context.spos)[id_parents == id_halo]
            shmr_h = pd.DataFrame(subhalo_context.shmr)[id_parents == id_halo]

            # Get mass, positions and velocities of the stars within a certain subhalo
            if id_subhalo != SUBHALO_ERROR_ID:
                m_subhalo = subhalo_context.mstr[id_subhalo]
                if m_subhalo > SUBHALO_MIN_MASS:
                    subhalo_particles = None
                    try:
                        subhalo_particles = get_p_subhalo(PATH, isubhalo=id_subhalo)
                        logger.debug(f'Primary subhalo {id_subhalo} has {len(subhalo_particles.m)} particles.')
                    except Exception as e:
                        logger.warning(e)
                        continue

                    # Defining a new origin
                    subhalo_particles.p -= halo_context.gpos[id_halo]
                    _, df_icl = label_particles(subhalo_particles, halo_context, id_halo)

                    if len(rotation_groups) > 0:
                        apply_rotation(rotation_groups, subhalo_particles, halo_context, df_icl, spos_h, shmr_h,
                                       SNAPNUM, id_halo, id_subhalo)
                    else:
                        # Create a masking array for BCG and ICL (BCG==0, ICL==1)
                        type4 = get_icl_mask(df_icl, subhalo_particles.m)
                        create_maps_and_zooms(halo_context, subhalo_particles, id_halo, id_subhalo, spos_h, shmr_h,
                                              type4, SNAPNUM, rotation=None)
                else:
                    logger.warning('Primary subhalo does not reach minimum mass.')

    # Directory containing the .npz files
    input_dirs = ['../data/npz/velocity_dispersion', '../data/npz/labels']
    output_dirs = ['../data/images/velocity_dispersion', '../data/images/labels']
    generate_img_dataset(input_dirs, output_dirs)





