# Fetcher: Magneticum

`catalog.name: magneticum_v2` → `MagneticumFetcherV2`
(`data_fetcher.py`), the only Magneticum fetcher. The current
`configs/catalog/magneticum.yaml` is already set to `magneticum_v2`.

The original interval-based `MagneticumFetcher` ("V1") has been **removed**
(2026-10-03). `MagneticumFetcherV2` used to subclass it; the shared
methods were folded into the one remaining class. `catalog.name:
magneticum` is no longer recognised by `DataFetcherFactory` and raises
`ValueError` — use `magneticum_v2`. See the [change log](../20-change-log.md).

## Format

Gadget-family group-catalogue and snapshot binaries, read via `g3read`
(`g3read.read_new`). Unlike Illustris-family catalogues, there is no
stored particle→halo column in the raw data at all — membership has to be
reconstructed by matching particle IDs (PIDs) against the group
catalogue's own particle listings.

## History: why the V1 approach was replaced

This is a real, measured bug, preserved in `MagneticumFetcherV2`'s own
class docstring as the rationale for the current design:

**The removed V1 approximated** "does particle X belong to halo H?" with a
`[min(PID of H), max(PID of H)]` range test. That's invalid — particle IDs
aren't ordered by spatial/halo membership, so a halo's ID interval ends up
spanning almost the entire ID space. Measured on real Magneticum
group-catalogue data: **17×–103× over-inclusion** (a halo with ~900K true
member particles matches ~90M IDs against its own interval). It also
re-read each halo's/subhalo's interval from disk on every single call
with no caching, which compounds into repeatedly reading multi-GB blocks
for millions of halos/subhalos.

**V2's approach** isn't a straightforward "build a correct index" fix —
that was tried first and reverted. An earlier corrected version built one
global sorted `(PID → halo_id, subhalo_id)` index across the whole group
catalogue, held in RAM. That's correct and reads each group file only
once, but its peak memory is `O(N)` in the *total* group-catalogue
particle count — it OOM-killed a real production run, and the cluster
queue this runs on can't grant more memory. Shaving the constant factor
(dropping a stable sort, freeing intermediates eagerly) only bought ~13%
(2.80 GB → 2.44 GB measured on a 90M-entry local fixture) — nowhere near
enough, since the approach still scales with total `N` regardless of
tuning.

**What V2 actually does**: restructure the algorithm rather than tune it.
Instead of one global index, it streams the group catalogue's PID block
once *per snapshot-file chunk* (i.e. once per iteration of
`_store_particles()`'s existing loop), holding only one group file's chunk
in memory at a time, matched directly against that chunk's much smaller
particle-ID set via a sorted search. Peak memory becomes
`max(one group file's chunk, one snapshot file's particle count)` —
independent of total catalogue size — at the cost of re-reading the group
catalogue once per snapshot file instead of once total. Since snapshot
files are typically far fewer than halos (tens, not millions), that
trade-off is favourable under the hard memory cap.

One implementation detail worth knowing if you're reading this code: V2's
`_assign_halo_ids(ids)` and `_assign_subhalo_ids(ids)` are called back to
back with the *same* `ids` array per snapshot file. The first call does
the full streaming pass and computes both results, caching them (keyed by
the `ids` array's identity) so the second call doesn't re-stream the group
catalogue a second time for no reason.

## Config values

| | Value |
|---|---|
| Snapshot (z) | 120 (z≈0.2125) |
| `a` | 0.8247 (from the snapshot header, per the config's own comment) |
| `H` | 0.704 |
| `Lbox` (kpc/h) | 290,294.4 — **flagged, see below** |
| Transition radius (kpc) | 499.39, labeled "Recomputed" in the config |
| Spatial resolution (kpc) | 2.425 — marked `TODO review` in the config |
| Halo mass filter | `> 1e13 * 0.6774` — uses `0.6774`, not Magneticum's own `H = 0.704`; looks copied from another catalogue's config rather than derived from this one |
| Subhalo (stellar) mass filter | `> 1e10` |
| `ignore_halos` | 13 halo IDs (`67, 255, 315, 1050, 1066, 1163, 1390, 2712, 3245, 3554, 4206, 8718, 15657`). |

**The `Lbox` flag**: the config's own adjacent comment states the raw box
size from the snapshot header is `352000.0` (ckpc/h), but `Lbox` is set to
`290294.4`. `352000 × 0.8247` (the file's own `a`) = `290294.4` exactly —
so `Lbox` looks like it already has the `a` (comoving→physical) scaling
pre-applied. That's inconsistent with every other field in this same file
(`RCRI`, `GPOS`, `SPOS`, `RHMS`), which expect the raw comoving value and
apply `x * a` themselves. Not yet resolved — a collaborator has been asked
whether `Lbox` should be `352000` instead.

## Open items

- `Lbox`'s apparent double-scaling (above) — awaiting collaborator confirmation.
- `spatial_resolution` marked `TODO review` in the file.
- The halo mass filter's `0.6774` literal doesn't match this catalogue's own `H = 0.704` — awaiting collaborator confirmation on whether the filter or the `H` value is the one that needs fixing.
- General `TODO review filters` comment above the whole `fields:` block in the config — the mass thresholds haven't been explicitly confirmed as final.

## See also

- [Shared mechanics](01-shared-mechanics.md).
- [Known issues](../18-known-issues.md).

---
*The V1→V2 fix description is quoted/paraphrased directly from `MagneticumFetcherV2`'s own class docstring, read directly from the current code this session. Config values verified directly against the current `configs/catalog/magneticum.yaml` this session (this page does not rely on the technical map's original numbers for Magneticum, which differed from the current file for at least one field).*
