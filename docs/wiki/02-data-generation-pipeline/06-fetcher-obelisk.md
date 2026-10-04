# Fetcher: Obelisk

`catalog.name: obelisk` → `ObeliskFetcher`. Reads the Obelisk main
catalogue (AdaptaHOP halo/galaxy properties, post-processed into one flat
HDF5 table per snapshot). Particle-level access reads star particles from
separate per-galaxy `GAL_{iout:05d}/gal_stars_{ID:07d}` Fortran binary
files — Obelisk's particle dumps contain only stars, no DM or gas, unlike
the other catalogues this pipeline was originally built around.

Note: the original technical map described this fetcher as "PROVISIONAL —
filled in with placeholders, not yet validated against real data." That's
now out of date — it has since been run end-to-end against real cluster
data (snapnum 201, 65 halos via `is_central`) and confirmed correct.

## Halo/subhalo split: `is_central`, not `level`

This was a real bug, found and fixed in this project. AdaptaHOP's `level`
field (1 = top-level, >1 = substructure) looks like the obvious way to
define "halo," and the fetcher originally used `level == 1`. It's wrong:
`level` only reflects whether an object sits within another structure's
AdaptaHOP *radius* — not whether it's gravitationally bound to it or
should be considered a satellite. Confirmed directly with the Obelisk team
(Maxime Trebitsch): *"being a substructure does not necessarily imply that
you would be called a satellite: a level 2 structure can be outside the
virial radius of the level 1 structure etc."* The recommendation was to
use `is_central` for halos, and `host`/`hostsub` to identify parents.

Checked against real data: every `level == 1` row is also `is_central`,
but ~36% of `is_central` rows have `level > 1`. Switching from `level==1`
to `is_central` recovered 23 real objects (up to 1.7×10¹² M☉, close to the
most massive halo in the box) that were previously being folded into
another halo's `subhalo_ids` instead of counted as their own halo.

```python
# ObeliskFetcher._get_halos / _get_subhalos
central_mask = raw['is_central']
...
raw['id_parents_computed'] = np.where(raw['is_central'], raw['ID'], raw['host'])
```

A central object's `id_parents` is its own `ID` (self-parented — it *is*
the halo); a non-central object's `id_parents` is `host` (its top-level
parent's `ID`). Obelisk needs no row-position/true-id conversion the way
FLAMINGO does — `ID`, `host`, and `hostsub` already share one namespace.

## `fsub` is not what links subhalos to parents

Easy to confuse with the above, so worth stating plainly: `fsub` has
**nothing to do with parent linkage**. `id_parents` (above) is the real
mechanism. `fsub` is a structural placeholder — `assign_id` just does
`np.arange(N)` over the row order at that point in the pipeline, because
the `Halo` class requires an `fsub` attribute to exist and Obelisk's
catalogue has no equivalent concept (confirmed absent from the Obelisk
team's own README). Its only downstream consumer is the output map
filename (`halo_{halo_id}_subhalo_{fsub}`) — never used to look up or
select any actual data, and never colliding across halos within a single
run since `_get_halos()`/`_get_subhalos()` are computed once per pipeline
run (not re-derived per batch — see `pipeline.py:57-58`), and filename
uniqueness is protected by `halo_id` regardless.

## Config values

| | Value |
|---|---|
| Snapshot (z) | 201 (z=3.56 — the lowest redshift available) |
| `a` | 0.21947 |
| `H` | 0.704 (WMAP-7 ΛCDM, same as parent Horizon-AGN; sourced and cross-checked against three Trebitsch et al. papers, cited in the config) |
| `Lbox` (kpc/h) | 100,000 (full Horizon-AGN parent box — Obelisk's own zoom region is smaller, but positions are still given relative to the full box, the standard RAMSES convention) |
| Halo mass filter | `> 1e11 * 0.704` |
| Subhalo (stellar) mass filter | `> 1e9` |

## Open items (email sent to the Obelisk team, awaiting reply on these)

- Are `rvir`/`r50_kpc` comoving or physical kpc? The README states plain "kpc" without saying — currently assumed physical.
- Is star-particle `velocity` already physical/peculiar, or does it need a correction analogous to the one confirmed for `position` (`Lbox/2 + (x/a) * h * 1000`)?
- AdaptaHOP's velocity dispersion (`sigma`) is flagged by the Obelisk team's own README (2026-02-18 note) as systematically overestimating for all objects, unfixed as of that catalogue version — no timeline known.
- Confirmed absent, but worth a final check with the Obelisk team: no `fsub`-equivalent field exists in the catalogue.

## See also

- [Shared mechanics](01-shared-mechanics.md).
- [Known issues](../18-known-issues.md) for the full, consolidated list.

---
*Written from this session's own direct investigation, fix, and verification (real-data testing, the Obelisk team's own reply) — supersedes the original technical map's "provisional, not yet validated" note on this fetcher.*
