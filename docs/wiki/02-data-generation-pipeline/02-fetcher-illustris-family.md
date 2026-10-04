# Fetcher: Illustris-TNG / TNG-Cluster / Millennium-TNG

All three of these route through the **same class**, `Illustris300Fetcher`
(`catalog.name: illustris-300`) — see
[shared mechanics](01-shared-mechanics.md) for why. This page covers what's
specific to each.

## Access library and format

Group-catalogue HDF5, read via `il.groupcat.loadHalos`/`loadSubhalos`.
Particle membership is resolved by `il.snapshot.loadSubhalo` — the
`illustris_python` toolkit handles particle-ID matching internally, so
there's no manual matching code in this repo for these three.
(`data_fetcher.py:866,933,985`)

## Distinguishing Millennium-TNG at runtime

There is no Millennium-TNG-specific code path. `Illustris300Fetcher` only
checks whether `'millenium'` appears as a substring in `self.catalog.path`
(`data_fetcher.py:911-919`), and if so reads the subhalo-parent field as
`SubhaloGroupNr` instead of `SubhaloGrNr` — both are real TNG-format field
names, this is just which one that particular data release used.

## Per-catalogue config values

| | Illustris-300 (TNG300) | TNG-Cluster | Millennium-TNG (MTNG-L500-4320-A) |
|---|---|---|---|
| Config file | `illustris300.yaml` | `tngcluster.yaml` | `millenium.yaml` |
| Snapshot (z) | 84 (z=0.2) | 84 (z=0.2) | 237 (z=0.25) |
| `a` | 0.8333 | 0.8333 | 0.8 |
| `H` | 0.6774 | 0.6774 | 0.6774 |
| `Lbox` (kpc/h) | 205,000 | 680,000 | 500,000 |
| Transition radius (kpc) | 637.031 | 711.387 | 703.578 |
| Spatial resolution (kpc) | 2.4602 ("reviewed by Ilaria in the 2nd iteration") | 2.4602 | 3.125 (`TODO BE REVIEWED` in the file) |
| Halo mass filter | `> 1e13*0.6774` | (same fields) | `> 1e13*0.6774` |

`millenium.yaml` also carries three smaller/lower-resolution MTNG box
variants (`L31.25-270-A`, `L125-1080-A`, `L62.5-540-A`) commented out
alongside the active `L500-4320-A` — a record of alternatives considered,
not dead code to clean up.

## Open items

- **Millennium-TNG's `H = 0.6774` doesn't match the paper's own cosmology table (`H0 = 73.00`, i.e. `h = 0.73`).** This is a real, unresolved discrepancy — see [Known issues](../18-known-issues.md).
- **Millennium-TNG's `spatial_resolution` is explicitly marked `TODO BE REVIEWED`** in the config file itself.
- A related, malformed config exists at `configs/catalog/bigillustris300.yaml` (`name: illustris-300`, not in the checked-in defaults cycle): `spatial_resolution: 64**3` — a bare unevaluated arithmetic expression, not a number, with its own comment `"kpc ? TODO verify with Sebastian"`. Self-flagged by whoever wrote it, not yet resolved.
- DIANOGA is often discussed alongside these three simulations in the paper, but has no fetcher — see [shared mechanics](01-shared-mechanics.md).

---
*Sourced from the SPACE-COE Technical Map (§13–14, §6 parameters table), cross-checked against the current repo state.*
