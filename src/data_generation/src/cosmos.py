import numpy as np

from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles


class CosmosData:
    def __init__(self, halo: Halo, subhalos: SubhaloContext, subhalo_id: int, subhalo_particles: SubhaloParticles):
        self.halo = halo
        self.subhalos = subhalos
        # Subhalo particles
        self.subhalo_id = subhalo_id
        self.subhalo_particles = subhalo_particles
        self.metadata = {
            'rotation': {
                'has_rotated': None,
                'angle': None
            }
        }

    def __repr__(self):
        return f"CosmosData(halo={self.halo}, subhalos={self.subhalos}, rotation={self.metadata['rotation']})"