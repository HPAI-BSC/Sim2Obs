import numpy as np

from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloContext


class CosmosData:
    def __init__(self, halo: Halo, subhalos: SubhaloContext, subhalo_particle_positions: np.ndarray):
        self.halo = halo
        self.subhalos = subhalos
        self.subhalo_particle_positions = subhalo_particle_positions
        self.metadata = {
            'rotation': {
                'has_rotated': None,
                'angle': None
            }
        }

    def __repr__(self):
        return f"CosmosData(halo={self.halo}, subhalos={self.subhalos}, rotation={self.metadata['rotation']})"