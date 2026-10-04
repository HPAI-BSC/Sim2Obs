# Data fetchers: shared mechanics

Every simulation the pipeline supports has its own catalogue format, its
own access library, and its own on-disk layout. `DataFetcher` (an abstract
base class in `src/data_generation/src/data_fetcher.py`) and its
subclasses exist to hide all of that behind one interface, so everything
downstream (rotation, map generation, ICL classification) runs unmodified
regardless of which simulation the data came from.

## Which fetcher a config actually routes to

`DataFetcherFactory.get_data_fetcher` (`data_fetcher.py:666-698`) dispatches
purely on `catalog.name` — a string field inside each `configs/catalog/*.yaml`
file. It recognises exactly these values and raises `ValueError` for
anything else:

| `catalog.name` | Fetcher class |
|---|---|
| `illustris-300` | `Illustris300Fetcher` |
| `eagle` | `EagleFetcher` |
| `flamingo` | `FlamingoFetcher` (V1) |
| `flamingo_v2` | `FlamingoFetcherV2` |
| `magneticum_v2` | `MagneticumFetcherV2` |
| `obelisk` | `ObeliskFetcher` |

Two things worth knowing before you go looking for a "Millennium-TNG
fetcher" or a "TNG-Cluster fetcher": **neither exists.** Both
`configs/catalog/millenium.yaml` and `configs/catalog/tngcluster.yaml` set
`name: illustris-300`, and both are routed through `Illustris300Fetcher` —
this is intentional (all three share the same AREPO/TNG-format group
catalogue schema), not a copy-paste bug. `Illustris300Fetcher`
distinguishes Millennium-TNG at runtime only by checking for the substring
`'millenium'` in `self.catalog.path`, which switches which subhalo-parent
field it reads (`SubhaloGroupNr` vs. `SubhaloGrNr` — two different real
field names TNG-family releases use) — see `data_fetcher.py:911-919`.

**DIANOGA has no fetcher at all.** It only appears in
`src/data/components/dianoga_dataset.py`, a training-time `Dataset` that
loads pre-existing `.npz` files — there is no halo/subhalo/particle
ingestion code for it anywhere in this repository. If a paper draft
describes DIANOGA's maps as having gone through this pipeline, that claim
isn't supported by anything in `src/data_generation/`.

## The shared per-field pipeline

Regardless of catalogue, every fetcher's `_get_halos()`/`_get_subhalos()`
runs its raw table through the same six generic functions, chained via
`run_pipeline`:

1. `remove_useless_columns` — drop anything not declared in the config's `fields` block.
2. `assign_id` — assign `halo_id`/`subhalo_id`/`fsub`/etc. Where the catalogue has no real concept for a required field (e.g. Obelisk has no "first subhalo" index), this just does `np.arange(N)` — an arbitrary row-position placeholder, not a derived value. See the per-fetcher pages for where this matters and where it's harmless.
3. `validate_and_process_fields` — per-field type/shape checks.
4. `convert_to_dataframe`
5. `apply_unit_conversion` — evaluates each field's config `scaling` string (e.g. `"x * a"`, `"x * 1e10 / h"`) via `apply_conversion` (`utils.py:788-797`), which `eval`s it with `x`, `a`, `h`, `Lbox` bound. This is the mechanism behind every comoving→proper, `1e10 Msun/h`→`Msun`, etc. conversion across every catalogue.
6. `apply_filters` — applies each field's configured mass/etc. cut.

**What differs by catalogue is only how particle→halo/subhalo membership
gets established** — that's forced by each catalogue's own group-finder
format and can't be made generic:

- **Illustris-300 / TNG-Cluster / "Millennium-TNG"**: resolved entirely by the `illustris_python` library (`loadSubhalo(..., subhalo_id, part_type)` returns exactly that subhalo's particles) — no manual ID matching in this repo at all.
- **Magneticum**: no stored particle→halo column exists in the raw data. `MagneticumFetcherV2` reconstructs it by exact particle-ID matching against the group catalogue's offset/length arrays.
- **FLAMINGO**: SOAP exposes two non-interchangeable ID spaces for the same rows — row-position and true catalogue index. `FlamingoFetcher` (V1) conflates them in its particle-matching code, silently assigning particles to the wrong halo; `FlamingoFetcherV2` fixes this. Full detail on [the FLAMINGO page](05-fetcher-flamingo.md) — this is not a minor edge case, it's been verified to affect essentially every halo in the catalogue.
- **EAGLE**: membership is already stored per-particle in the snapshot (`GroupNumber`, `SubGroupNumber`) — an exact boolean-mask match, no reconstruction needed.
- **Obelisk**: `ID`, `host`, and `hostsub` are all expressed in the same namespace already (the catalogue's own `ID` column) — no row-position/true-index split to reconcile, unlike FLAMINGO. See [the Obelisk page](06-fetcher-obelisk.md) for how halo/subhalo classification itself works here.

## The common output shape

No matter which catalogue, every fetcher converges on the same three
output types — `HaloContext`, `SubhaloContext` (or `LazySubhaloContext`),
`SubhaloParticles` (or `LazySubhaloParticles`) — consumed identically by
`DataLoader` downstream. That uniform interface, despite five structurally
unrelated upstream formats (Gadget binary, two different HDF5
group-catalogue schemas, a SQL database, and a SWIFT/SOAP HDF5 schema), is
the actual point of the `DataFetcher` abstraction.

## A structural assumption worth knowing

Every fetcher treats its catalogue's own group-finder output (FoF/SUBFIND
for the Illustris family and Magneticum, SOAP for FLAMINGO, AdaptaHOP for
Obelisk, EAGLE's own database pipeline) as authoritative ground truth for
halo/subhalo membership. None of this code re-runs or independently
verifies a group-finding algorithm — that's out of scope by design.

## See also

- Per-simulation fetcher pages, for what's specific to each catalogue.
- [Known issues](../18-known-issues.md) for the open config-value questions (cosmology mismatches, unconfirmed units) that cut across catalogues.

---
*Sourced from the SPACE-COE Technical Map (§13–14), cross-checked against the current repo state. Where this conversation's own verification (FLAMINGO V2, Obelisk `is_central`) is more current than the technical map's original snapshot, this page uses the newer facts.*
