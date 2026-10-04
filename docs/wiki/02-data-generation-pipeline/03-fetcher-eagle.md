# Fetcher: EAGLE

`catalog.name: eagle` → `EagleFetcher` (`data_fetcher.py:1071`). No V2
variant exists for this fetcher.

## Format

Halo/subhalo catalogue: the public EAGLE SQL database, queried via
`eagleSqlTools.execute_query` against a hand-written `SELECT` query.
Particles: a local snapshot HDF5 file, read with `h5py` and filtered
locally by an exact `(GroupNumber, SubGroupNumber)` match — membership is
already stored per-particle in the snapshot, so no ID reconstruction is
needed the way Magneticum requires.

`halo_id` is taken directly from the database's own `GroupID` field
(`fields.halo.GroupID` → `name: "halo_id"`, no scaling), unlike most other
fetchers where `assign_id` synthesizes it. `subhalo_id` similarly comes
directly from `SubGroupNumber`, and a separate `group_number` field
(`GroupNumber`) is kept alongside it — `subhalo.py`'s `get_item` lookup
has an EAGLE-specific branch that queries by `(group_number, subhalo_id)`
together rather than by a single index, since `SubGroupNumber` alone isn't
globally unique across groups.

## Config values

| | Value |
|---|---|
| Snapshot (z) | 26 (z=0.18) |
| `a` | 0.84551589459619 |
| `H` | 0.6774 |
| `Lbox` (kpc/h) | 73,811.63 — derived as `50 * 1000 / 0.6774` (a stated physical 50 Mpc box divided by `h`), a distinct derivation style from the other catalogues' Lbox literals |
| Transition radius (kpc) | 501.135 |
| Spatial resolution (kpc) | 3.3186 |
| Halo mass filter | `> 1e13 * 0.6774` (comment: `"Final unit measures TODO update"`) |
| Subhalo (stellar) mass filter | `> 1e10` |
| `ignore_halos` | `[26000000000009]` — no comment explaining why this one halo is excluded |

The active box is `RefL0050N0752`. Two alternative variants
(`AGNdT9L0050N0752`, `RefL0100N1504`) are present in the config as
commented-out blocks, each with their own `transition_radius`/`Lbox`/
`ignore_halos` — a record of alternatives considered, not dead
configuration to clean up.

## Open items

- The halo mass filter's own comment (`"Final unit measures TODO update"`) flags it as not yet finalized — worth checking before trusting `EAGLE`'s sample selection as-is.
- `ignore_halos: [26000000000009]` has no explanation anywhere in the file for why this specific halo is excluded.
- `H = 0.6774` vs. the paper's own cosmology table listing `67.77` for EAGLE — see [Known issues](../18-known-issues.md).

---
*Config values verified directly against the current `configs/catalog/eagle.yaml` this session. Architecture description (SQL database + local snapshot matching) sourced from the SPACE-COE Technical Map §13–14, not independently re-traced through `EagleFetcher`'s full body this session — flag if this needs deeper verification too.*
