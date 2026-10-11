"""
SubhaloParticlesV2: a multi particle type aware version of `SubhaloParticles`.

`SubhaloParticles` (subhalo.py) keeps working exactly as before. This class subclasses it, so the
constructor, `__len__`, `is_empty` and `__repr__` are inherited, and overrides only what assumed a
single (star) particle type:

  * the overlap boundary is recorded per particle type, so stars and dark matter, which have different
    particle counts, are each cut at their own boundary by `non_overlapping`
  * `__add__` keeps a particle type that only one of the two operands has
  * `preprocess` recentres every `<type>_p` position array (star_p, dm_p, ...) on the halo
  * the constructor logs a warning when the catalog declares a field that the loaded data lacks

NOT done yet: a multi type `rotate`. The inherited version only rotates `star_p`, so `rotate` raises
NotImplementedError (for a rotation other than 0 degrees) when there are other position arrays, instead of
silently misaligning them. Note for that step: `utils.rotate` draws a new random rotation on every call,
so all position arrays must share one rotation matrix.

Compatibility notes, so both classes can be merged later:
  * `overlapping_particles_id` is still set (to the star boundary, or the first type's) and still read
    as "there is an overlap" when it is not None. Resetting it to None, as the fetchers do after
    merging the subhalos of a halo, still means "no overlapping particles".
  * `a + b` still records the boundary on `a` itself the first time (a side effect the data loader
    relies on), now as `a.overlapping_boundaries = {particle_type: count}` as well.
  * `overlapping_boundaries` is a plain attribute that is never routed through the constructor,
    because a nested dict would make the constructor misread the data as nested by particle type.
"""
import copy

import numpy as np

from src.data_generation.src.halo import Halo
from src.data_generation.src.subhalo import SubhaloParticles, convert_to_dict
from src.data_generation.src.utils import dict_depth, setup_logger

logger = setup_logger("Subhalo-Particles-V2")

# Same particle type aliases as SubhaloParticles.__init__
FIX_NAMES = {'stars': 'star', 'dark_matter': 'dm'}


class SubhaloParticlesV2(SubhaloParticles):
    """
    Particles of a subhalo (or halo) for several particle types, e.g. `star_*` and `dm_*` attributes.

    Attributes:
    -----------
    overlapping_particles_id : int or None
        Same meaning as in SubhaloParticles: None means no overlapping particles.
    overlapping_boundaries : dict or None
        Number of the object's own particles of each type, `{particle_type: count}`, recorded the first
        time another object is added. Particles of that type after the boundary are overlapping ones.
    """
    # Class level default: an object without the instance attribute (e.g. unpickled from a state saved
    # before it existed) behaves as "no per-type boundaries recorded"
    overlapping_boundaries = None

    def __init__(self, data: dict, column_map) -> None:
        """
        Build the particles exactly as SubhaloParticles does, then warn about declared but missing fields.

        Parameters:
        -----------
        data : dict
            Particle data, flat or nested by particle type (see SubhaloParticles).
        column_map : DictConfig or dict or None
            Mapping of the catalog's particle fields to attribute names.
        """
        super().__init__(data, column_map)
        self.__dict__.setdefault('overlapping_boundaries', None)
        self._warn_about_missing_fields(data, column_map)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _particle_type(attribute: str) -> str:
        """Particle type prefix of a `<type>_<name>` attribute (e.g. 'star' for 'star_p')."""
        return attribute.partition('_')[0]

    @staticmethod
    def _entry_name(entry):
        """Attribute name of a column map entry, which is a plain string or a config with a `name` key."""
        return entry.name if hasattr(entry, 'name') else entry

    def _arrays(self) -> dict:
        """Every numpy array attribute of the object, in definition order."""
        return {key: value for key, value in vars(self).items() if isinstance(value, np.ndarray)}

    @property
    def particle_types(self) -> list:
        """list: particle types that have at least one array, in definition order (e.g. ['star', 'dm'])."""
        types = []
        for key in self._arrays():
            particle_type = self._particle_type(key)
            if particle_type not in types:
                types.append(particle_type)
        return types

    @property
    def position_attributes(self) -> list:
        """list: names of the position arrays, one `<type>_p` per particle type that has them."""
        return [key for key in self._arrays() if key == f'{self._particle_type(key)}_p']

    @staticmethod
    def _stack_if_object(values: np.ndarray) -> np.ndarray:
        """Stack an object array of per-particle vectors (parquet lists) into a proper (N, 3) array."""
        if values.dtype == object and len(values) > 0 and isinstance(values[0], (list, np.ndarray)):
            return np.stack(values)
        return values

    def _warn_about_missing_fields(self, data, column_map) -> None:
        """
        Log a warning for every field the catalog declares for a particle type present in `data`
        that did not become an attribute (the parent constructor drops them silently).
        """
        if column_map is None or not data:
            return
        data = convert_to_dict(data)
        missing = []
        if dict_depth(data) == 1:
            for variable in column_map.keys():
                attribute = f'star_{self._entry_name(column_map[variable])}'
                if not hasattr(self, attribute):
                    missing.append(attribute)
        elif dict_depth(data) == 2:
            for particle_type, columns in data.items():
                if not columns or particle_type not in column_map:
                    continue
                prefix = FIX_NAMES.get(particle_type, particle_type)
                for variable in column_map[particle_type]:
                    attribute = f'{prefix}_{self._entry_name(column_map[particle_type][variable])}'
                    if not hasattr(self, attribute):
                        missing.append(attribute)
        if missing:
            logger.warning(f'Fields declared in the catalog are missing from the loaded particle data: {missing}. '
                           f'Check the field names of the catalog YAML.')

    # ------------------------------------------------------------ overlap logic

    def _record_boundaries(self, own: dict, other: dict) -> None:
        """
        Record how many own particles of each type there are, before another object is concatenated.

        Types that only the other object has get boundary 0. Also sets `overlapping_particles_id` to the
        star boundary (or the first type's), which is what existing code reads.
        """
        boundaries = {}
        for key, values in own.items():
            boundaries.setdefault(self._particle_type(key), len(values))
        for key in other:
            boundaries.setdefault(self._particle_type(key), 0)
        self.overlapping_boundaries = boundaries
        if boundaries:
            self.overlapping_particles_id = boundaries.get('star', next(iter(boundaries.values())))

    def _non_overlapping_for(self, particle_type: str = None):
        """
        Return the particles that belong to the original (non-overlapping) subhalo.

        Every type is cut at its own recorded boundary. If there is no overlap
        (`overlapping_particles_id` is None) the object itself is returned. An object that only has the
        legacy integer boundary (no per-type boundaries) is cut at that integer for every type, as
        SubhaloParticles does.

        Parameters
        ----------
        particle_type : str, optional
            Unused, kept for compatibility with SubhaloParticles.
        """
        boundaries = self.overlapping_boundaries
        if self.overlapping_particles_id is None or (boundaries is None and self.overlapping_particles_id == 0):
            logger.debug('There are no overlapping particles, returning self.')
            return self

        result = copy.copy(self)
        for key, values in self._arrays().items():
            boundary = self.overlapping_particles_id if boundaries is None else boundaries.get(self._particle_type(key))
            if boundary is not None:
                setattr(result, key, values[:boundary])
        return result

    def __add__(self, other):
        """
        Concatenate `other`'s particle arrays onto this object's, type by type.

        The first time, the boundary between own and added particles is recorded per particle type on
        `self` (a side effect existing code relies on, see the module docstring). Arrays that only one
        of the two objects has are kept as they are, so a neighbour that carries only stars adds no dark
        matter, and a type that only the neighbour has is added with an own boundary of 0.

        Parameters:
        -----------
        other : SubhaloParticles
            Another SubhaloParticles or SubhaloParticlesV2 to append.

        Returns:
        --------
        SubhaloParticlesV2
            A new object with the concatenated arrays and the recorded boundaries.
        """
        if not isinstance(other, SubhaloParticles):
            return NotImplemented

        own = self._arrays()
        added = {key: value for key, value in vars(other).items() if isinstance(value, np.ndarray)}
        if self.overlapping_particles_id is None:
            self._record_boundaries(own, added)

        result = copy.copy(self)
        for key in list(own) + [key for key in added if key not in own]:
            if key in own and key in added:
                setattr(result, key, np.concatenate((own[key], added[key]), axis=0))
            else:
                setattr(result, key, own[key] if key in own else added[key])

        # A type that appears for the first time through `other` has no own particles. Legacy objects that
        # only have the integer boundary keep it as their single boundary, as SubhaloParticles does
        if result.overlapping_boundaries is not None:
            for key in added:
                result.overlapping_boundaries.setdefault(self._particle_type(key), 0)
        return result

    # ------------------------------------------------------ position processing

    @staticmethod
    def _apply_pbc(delta: np.ndarray, lbox: float, label: str) -> np.ndarray:
        """Fold offsets larger than half the box back to the nearest periodic image (minimum image convention)."""
        corrected = delta.copy()
        over_half = corrected > (lbox / 2)
        under_half = corrected < (-lbox / 2)
        corrected[over_half] -= lbox
        corrected[under_half] += lbox
        per_axis = np.sum(over_half | under_half, axis=0)
        if np.sum(per_axis) > 0:
            logger.debug(f'{label} corrected due to PBC (x/y/z): {per_axis} / {len(delta)}')
        return corrected

    def preprocess(self, halo: Halo, lbox: float = None):
        """
        Make every position array (star_p, dm_p, ...) relative to the halo centre.

        With `lbox`, offsets are corrected for periodic boundaries (minimum image convention).

        Parameters
        ----------
        halo : Halo
            Provides the centre (`gpos`) the positions are made relative to.
        lbox : float, optional
            Size of the simulation box, same units as the positions. If None, no correction is applied.
        """
        attributes = self.position_attributes
        if not attributes:
            logger.warning('No position arrays (<type>_p) to preprocess.')
            return
        logger.debug(f'halo gpos: {np.shape(halo.gpos)}, preprocessing {attributes}')
        for attribute in attributes:
            positions = self._stack_if_object(getattr(self, attribute))
            delta = positions - halo.gpos
            if lbox is not None:
                delta = self._apply_pbc(delta, lbox, f'{attribute} particles (halo {getattr(halo, "halo_id", "?")})')
            setattr(self, attribute, delta)

    def rotate(self, z, degree):
        """
        Rotate the particle positions, only while that is safe.

        The inherited rotation only handles `star_p`. With any other position array (`dm_p`, ...) it would
        leave it unrotated while the stars turn, silently misaligning the particle types, so this raises
        instead. A multi type rotation is not implemented yet (it must apply one rotation matrix to every
        position array, because `utils.rotate` draws a new random rotation on every call). A rotation of
        0 degrees changes nothing and is always allowed.

        Parameters:
        -----------
        z : np.ndarray
            Reference axis, as in SubhaloParticles.rotate.
        degree : float
            Angle in degrees.

        Raises:
        -------
        NotImplementedError
            If `degree` is not 0 and there are position arrays other than `star_p`.
        """
        if degree != 0:
            unsupported = [name for name in self.position_attributes if name != 'star_p']
            if unsupported:
                raise NotImplementedError(
                    f'SubhaloParticlesV2.rotate does not support {unsupported} yet, rotating only star_p would '
                    f'misalign the particle types. Run without rotate_step for now.')
        super().rotate(z, degree)
