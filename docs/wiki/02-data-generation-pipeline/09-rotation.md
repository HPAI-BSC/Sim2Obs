# Rotation / viewing-angle augmentation

`RotateStep.process` (`rotate_step.py:71-140`), calling `utils.rotate`
(`utils.py:865-889`).

## Not currently applied to any dataset

Worth stating first, since it's easy to assume otherwise from the code
alone: this step exists and is implemented, but **no checked-in config
file currently supplies the `rotation_directions`/`rotation_angle`
parameters it needs to run.** It was originally introduced as a data
augmentation strategy; it isn't applied to the current dataset because the
sample of halos across simulations already provides enough examples
without it. Without those parameters, `RotateStep` cannot run at all —
`_build_configuration` raises `ValueError`.

## How it works, when it does run

1. Config supplies a fixed set of reference directions (`rotation_directions`) and a step angle (`rotation_angle`, e.g. 45°).
2. For each reference direction, `iterations = 360 // rotation_angle` views are generated. The very first view overall (`i==0` of the first direction) uses `degree=0` — the unrotated, canonical view. Every other view, for every direction and every iteration, calls `.rotate(direction, rotation_angle)` with the *same fixed* `rotation_angle` value, not `i × rotation_angle`.
3. `utils.rotate(z, degree, X)`: if `degree==0`, no-op. Otherwise, `random_vector_at_angle(z, angle_deg=degree)` samples a vector at a **fixed polar angle** (`degree`) from `z`, with a **uniformly random azimuthal angle** around `z` — i.e. a random point on a cone around `z`, not a uniformly random point on the whole sphere. `get_transform_matrix(new_z, z)` then builds the 3×3 rotation matrix taking `z` onto `new_z` via a quaternion construction (mathematically equivalent to Rodrigues' rotation formula). `X_rot = Q @ X` is applied to positions and velocities.
4. `.rotate()` mutates `cosmos_data.subhalos`, `cosmos_data.subhalo_particles`, and `cosmos_data.input_feature` **in place**, so each successive call rotates whatever the *previous* call already produced.

**Consequence of point 4**: the sequence of views for one reference
direction is a **cumulative random walk on a cone**, not a set of fixed
absolute angles measured from the original orientation. If rotation
augmentation is ever reintroduced and described anywhere (a paper, a
config comment), this is the specific scheme to describe — not a simple
fixed-grid or fully-random rotation.

## Dead code nearby

- `RotateStep.__init__` unpacks a `self.z` attribute from config, but it's never referenced anywhere in `process()` — a dead attribute.
- A separate, explicit Rodrigues-formula implementation (`rotate_vector`, `utils.py:201-236`) exists alongside the active quaternion-based `rotate()`, but is only called by the disused `rotate_old` — not the active path. `rotate_old`/`rotate_vector`/`get_transform_matrix_old` are each explicitly marked "DO NOT USE"/"OLD"/"NOT ROBUST ENOUGH" in their own docstrings — version history left in place, not part of the active computation.

## See also

- [Known issues](../18-known-issues.md).

---
*Sourced from the SPACE-COE Technical Map §16(d), extended with the "not currently applied" fact and the config's own "no rotation_group anywhere in configs/" confirmation, both established directly in this project's own conversation history (not solely the technical map).*
