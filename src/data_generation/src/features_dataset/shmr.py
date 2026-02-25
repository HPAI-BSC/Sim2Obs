from typing import Dict

import numpy as np

from src.data_generation.src.feature_computer import FeatureComputer


@FeatureComputer.register('stellar_half_mass_radius')
def stellar_half_mass_radius(all_particles: Dict[int, dict], columns_required: Dict) -> np.ndarray:
    """
    Compute stellar half mass radius for all subhalos.

    Parameters
    ----------
    all_particles    : {subhalo_id: {'POS': array, 'MASS': array, ...}}
    columns_required : dict of required column config from catalog
    """
    return np.array([
        FeatureComputer._subhalo_shmr(particles)
        for particles in all_particles.values()
    ])