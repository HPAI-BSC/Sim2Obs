from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloContext, SubhaloParticles


class CosmosData:
    """
    Per-halo working state carried through the data-generation pipeline.

    Bundles a single halo, its subhalos, and the particles of one specific
    subhalo (`subhalo_id`) together with the mutable state that later
    pipeline steps (rotation, map generation, feature computation) attach
    to it: rotation metadata, generated maps, input/output feature arrays,
    and any errors recorded along the way.

    Attributes:
    -----------
    halo : Halo
        The halo this instance represents.
    subhalos : SubhaloContext
        Subhalos associated with `halo` (including, where relevant,
        subhalos from overlapping neighbouring halos).
    subhalo_id : int
        ID of the specific subhalo whose particles this instance tracks.
    subhalo_particles : SubhaloParticles
        Particle data for `subhalo_id`.
    non_overlapping : SubhaloParticles
        Particle data for `subhalo_id` with particles belonging to
        overlapping neighbouring subhalos removed.
    metadata : dict
        Mutable pipeline state: `rotation` (`has_rotated`, `angle`) and
        `map` (populated by map-generation steps).
    input_feature : Any
        Input feature array/tensor, populated by a later pipeline step.
    output_feature : Any
        Output feature array/tensor, populated by a later pipeline step.
    errors : dict
        Errors recorded against this halo while it moves through the
        pipeline, keyed by the step/context that raised them.
    """
    def __init__(self, halo: Halo, subhalos: SubhaloContext, subhalo_id: int, subhalo_particles: SubhaloParticles,
                 non_overlapping: SubhaloParticles):
        """
        Initialize a CosmosData instance for one halo/subhalo pair.

        Parameters:
        -----------
        halo : Halo
            The halo this instance represents.
        subhalos : SubhaloContext
            Subhalos associated with `halo`.
        subhalo_id : int
            ID of the specific subhalo whose particles this instance tracks.
        subhalo_particles : SubhaloParticles
            Particle data for `subhalo_id`.
        non_overlapping : SubhaloParticles
            Particle data for `subhalo_id` with particles belonging to
            overlapping neighbouring subhalos removed.
        """
        self.halo = halo
        self.subhalos = subhalos
        # Subhalo particles
        self.subhalo_id = subhalo_id
        self.non_overlapping = non_overlapping
        self.subhalo_particles = subhalo_particles
        self.metadata = {
            'rotation': {
                'has_rotated': None,
                'angle': None
            },
            'map': {}
        }
        self.input_feature = None
        self.output_feature = None
        self.errors = {}

    def __repr__(self):
        """Return a short string identifying the halo, subhalos, and current rotation state."""
        return f"CosmosData(halo={self.halo}, subhalos={self.subhalos}, rotation={self.metadata['rotation']})"