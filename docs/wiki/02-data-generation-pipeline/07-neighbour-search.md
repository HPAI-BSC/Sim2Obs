# Neighbour search & overlapping-halo handling

Catalogue-agnostic — runs identically regardless of which fetcher supplied
the data. `DataLoader.get_overlapping_halos` (`data_loader_step.py:521-624`).

## What it does

1. Builds **one `scipy.spatial.cKDTree`** over every subhalo's position (`spos`) in the whole catalogue — constructed with **no `boxsize` argument**, i.e. a plain Euclidean tree, not periodic-boundary-aware.
2. For each halo, `tree.query_ball_point(halo_position, r=2×R200)` — finds every subhalo within a Euclidean ball of radius `2×R200` of that halo's centre.
3. Keeps only subhalos whose `id_parents` differs from the current halo's own id (genuinely foreign/neighbouring subhalos).
4. Persists the resulting `{halo_id: [(parent_id, subhalo_id), ...]}` map to a parquet file for reuse/debugging.

## The non-PBC-aware search radius

Point 1 above is a real, code-visible gap: a subhalo near a periodic-box
face, whose *wrapped* (nearest-image) distance to a halo is within
`2×R200` but whose raw stored coordinate is on the opposite side of the
box, would not be found by this search — even though, once a subhalo *is*
selected, its particle positions do get periodic-boundary corrected (see
[Periodic boundary correction](08-periodic-boundary-correction.md)). This
is a scale/consistency gap between the search step and the later
per-particle correction, not a contradiction in either step individually.
Whether it materially affects results depends on how many halos sit
within `2×R200` of a box face relative to each catalogue's `Lbox` — not
checked against real catalogue geometry.

## See also

- [Periodic boundary correction](08-periodic-boundary-correction.md) — what happens to a selected neighbour's particle positions afterward, including a related, more consequential coordinate-frame issue.
- [Known issues](../18-known-issues.md).

---
*Sourced from the SPACE-COE Technical Map §16(a), cross-checked against this session's own reading of the same code while writing the paper's methodology section.*
