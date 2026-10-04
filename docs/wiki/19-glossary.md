# Glossary of fields & units

Every catalogue calls the same physical quantity by a different raw field
name. This page is the cross-reference — useful any time you're reading
a `configs/catalog/*.yaml` file and need to know "what is this called in
the other five catalogues."

All field names below verified directly against the current
`configs/catalog/*.yaml` files this session, not carried over from the
technical map's original snapshot.

## Halo properties, by catalogue

| Concept → pipeline name | Illustris-300 / TNG-Cluster | Millennium-TNG | Magneticum | FLAMINGO | EAGLE | Obelisk |
|---|---|---|---|---|---|---|
| Halo mass (M200) → `m200` | `Group_M_Crit200` | `Group_M_Crit200` | `MCRI` | `spherical_overdensity_200_crit.total_mass` | `Group_M_Crit200` | `mvir` |
| Halo radius (R200) → `r200` | `Group_R_Crit200` | `Group_R_Crit200` | `RCRI` | `spherical_overdensity_200_crit.soradius` | `Group_R_Crit200` | `rvir` |
| Halo position → `gpos` | `GroupPos` | `GroupPos` | `GPOS` | `input_halos.halo_centre` | `GroupCentreOfPotential_x/y/z` | `x`/`y`/`z` (box units) |
| "First subhalo" index → `fsub` | `GroupFirstSub` (real field) | `GroupFirstSub` (real field) | `FSUB` (real field) | *none — synthesized placeholder, see [FLAMINGO page](02-data-generation-pipeline/05-fetcher-flamingo.md)* | *not configured* | *none — synthesized placeholder, see [Obelisk page](02-data-generation-pipeline/06-fetcher-obelisk.md)* |

**`fsub` is the single most dangerous name in this codebase to assume
consistency for.** It's a real, meaningful catalogue field for
Illustris-family and Magneticum, but a purely synthetic `np.arange(N)`
placeholder — satisfying a structural requirement, meaning nothing — for
FLAMINGO and Obelisk. See each fetcher's page for what it's actually used
for downstream (spoiler: only an output filename, never a lookup).

## Subhalo properties, by catalogue

| Concept → pipeline name | Illustris-300 / TNG-Cluster | Millennium-TNG | Magneticum | FLAMINGO | EAGLE | Obelisk |
|---|---|---|---|---|---|---|
| Subhalo position → `spos` | `SubhaloPos` | `SubhaloPos` | `SPOS` | `input_halos.halo_centre` (same field as halo `gpos`) | `CentreOfPotential_x/y/z` | `x`/`y`/`z` (same field as halo `gpos`) |
| Subhalo stellar mass → `mstr` | `SubhaloMassType` (index 4) | `SubhaloMassType` (index 4) | `SMST` (index 4) | `bound_subhalo.stellar_mass` | `MassType_Star` | `mstar` |
| Subhalo half-mass radius → `shmr` | `SubhaloHalfmassRadType` (index 4) | `SubhaloHalfmassRadType` (index 4) | `RHMS` | `bound_subhalo.half_mass_radius_stars` | `HalfMassRad_Star` | `r50_kpc` |
| Parent-halo linkage → `id_parents` | `SubhaloGrNr` | `SubhaloGroupNr` (**different name from Illustris-300**, dispatched at runtime by a path-substring check — see [Illustris family page](02-data-generation-pipeline/02-fetcher-illustris-family.md)) | `GRNR` | `soap.host_halo_index` (row-position space — see [FLAMINGO page](02-data-generation-pipeline/05-fetcher-flamingo.md) for why this ID space matters) | `SubGroupNumber` + `GroupNumber` together (see below) | `host` (derived: self-`ID` if `is_central`, else `host` — see [Obelisk page](02-data-generation-pipeline/06-fetcher-obelisk.md)) |

**EAGLE needs two fields together, not one.** `SubGroupNumber` alone
isn't globally unique across groups, so EAGLE subhalo lookups query by
`(GroupNumber, SubGroupNumber)` as a pair — `subhalo.py`'s `get_item` has
an EAGLE-specific branch for this.

**Obelisk's `id_parents` isn't a raw field at all** — it's computed
per-row from `is_central`, `ID`, and `host` (see the fix documented on
the Obelisk page). Don't go looking for an `id_parents` entry in
`obelisk.yaml`'s `fields:` block; there isn't one.

## Pipeline-internal terms

| Term | Meaning |
|---|---|
| `HaloContext` / `SubhaloContext` | Wrapper around the filtered, unit-converted, renamed halo/subhalo table. See [Core abstractions](01-overview-architecture/04-core-abstractions.md). |
| `SubhaloParticles` | Wrapper around per-particle position/velocity/mass arrays for one subhalo (or a combined set, after `__add__`). |
| `id_parents` | The column every `SubhaloContext` uses to answer "which subhalos belong to halo X" — see the table above for how it's populated per catalogue. |
| `non_overlapping` | A `SubhaloParticles`' own particles, as opposed to particles pulled in from a neighbouring halo. See [ICL/BCG classification](02-data-generation-pipeline/10-icl-bcg-classification.md) — only non-overlapping particles get classified. |
| Interloper | A particle belonging to a *different*, nearby halo, included in a map because it falls within the field of view — flagged, not excluded, in a separate mask channel. See [Neighbour search](02-data-generation-pipeline/07-neighbour-search.md) and [Map generation](02-data-generation-pipeline/12-map-generation.md). |
| Transition radius | The radial scale where cumulative ICL mass starts sustainedly exceeding cumulative BCG mass — sets each map's field of view. See [ICL/BCG classification](02-data-generation-pipeline/10-icl-bcg-classification.md). |
| `PipelineStep` | The continuation-passing base class every data-generation stage implements. See [Core abstractions](01-overview-architecture/04-core-abstractions.md). |

## Units

- **Comoving vs. physical (proper)**: comoving coordinates don't shrink with the universe's expansion; physical/proper coordinates do. The scale factor `a` (`a = 1/(1+z)`) converts between them — comoving × `a` = physical. This is the single most common `scaling` string across every catalogue config (`"x * a"`).
- **`/h` suffix**: a quantity expressed in units that still carry a factor of the (dimensionless) Hubble parameter `h` (where `H0 = 100h` km/s/Mpc), not yet divided out. `Msun/h`, `kpc/h`. Multiplying or dividing by `h` (the catalogue's own configured value, *not* always the same across catalogues — see each fetcher's page) converts to/from the "plain" unit.
- **`ckpc`/`pkpc`**: comoving kpc / proper (physical) kpc — an explicit alternative to the `/h` convention above, seen in some catalogue configs' `input`/`output` field labels.
- **Box units**: some catalogues (Obelisk) store positions as `[0,1]` fractions of the box size directly, rather than in physical/comoving length units — converted via `× Lbox`.

## See also

- Part II's per-fetcher pages for full config tables (cosmology, box size, filters) per catalogue.
- [Data fetchers: shared mechanics](02-data-generation-pipeline/01-shared-mechanics.md) for how the `scaling` string mechanism actually works.

---
*Field names compiled directly from the current `configs/catalog/*.yaml` files this session (not carried over from the technical map, which didn't include a glossary of this kind).*
