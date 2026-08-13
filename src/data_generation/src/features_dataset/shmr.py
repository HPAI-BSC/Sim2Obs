import numpy as np

from src.data_generation.src.feature_computer import FeatureComputer
from src.data_generation.src.lazy_subhalo import LazySubhaloParticles
from src.data_generation.src.subhalo import SubhaloParticles


@FeatureComputer.register('stellar_half_mass_radius')
def stellar_half_mass_radius(**kwargs) -> float:
    """Compute stellar half mass radius for a single subhalo."""
    particles = kwargs.get('particles', None)

    if particles is None:
        raise ValueError('Particles cannot be None.')

    if isinstance(particles, SubhaloParticles):
        pos = particles.star_p
        mass = particles.star_m

    elif isinstance(particles, LazySubhaloParticles):
        # Only read the two columns we need — avoids loading full table
        table = particles.dataset.to_table(
            columns=['star_p', 'star_m'],
            filter=particles._build_filter()
        )

        # Apply transforms (PBC correction, rotation) if registered
        for fn in particles.transforms:
            table = fn(table)

        if table.num_rows == 0:
            return np.nan

        pos  = np.array(table['star_p'].to_pylist())   # (N, 3)
        mass = np.array(table['star_m'].to_pylist())   # (N,)

    else:
        raise TypeError(f"Unsupported particle type: {type(particles)}")

    # Shared computation
    if pos is None or mass is None or len(mass) == 0:
        return np.nan

    pos = np.asarray(pos)
    mass = np.asarray(mass)

    center = np.average(pos, weights=mass, axis=0)
    r = np.linalg.norm(pos - center, axis=1)
    sort_idx = np.argsort(r)
    cumulative_mass = np.cumsum(mass[sort_idx])
    half_mass = cumulative_mass[-1] / 2.0
    idx = np.clip(np.searchsorted(cumulative_mass, half_mass), 0, len(r) - 1)

    return r[sort_idx[idx]]
