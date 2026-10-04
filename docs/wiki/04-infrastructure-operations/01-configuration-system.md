# Configuration system

Everything is Hydra: config groups under `configs/`, composed via
`defaults:` lists, with CLI overrides available at any level (e.g.
`catalog=eagle`).

## Config groups

| Group | Purpose |
|---|---|
| `configs/catalog/*.yaml` | Per-simulation cosmology, box size, field mappings, filters — one file per catalogue. See Part II's per-fetcher pages for the actual values. |
| `configs/data/*.yaml` | Per-training-run datamodule config — which `.npz` directories to load as `data_dir`/`target_maskdir`, batch size, split, `mask_interlopers`. |
| `configs/model/*.yaml` | Per-architecture hyperparameters (optimizer, scheduler, network config). |
| `configs/trainer/*.yaml` | PyTorch-Lightning `Trainer` settings — `cpu`/`gpu`/`mps`/`ddp`/`ddp_sim`/`default`. |
| `configs/paths/*.yaml` | Where data/logs/output live — differs between `local` (development) and `gpfs` (the MN5 cluster). |
| `configs/data_generation.yaml`, `configs/train.yaml`, `configs/eval.yaml` | The three top-level configs each entry script composes from. See [Entry points](../01-overview-architecture/02-entry-points.md). |

## How a catalogue config actually gets interpreted

Every `configs/catalog/*.yaml`'s field entries declare `input`/`output`
units and a `scaling` string, e.g. `"x * a"` or `"x * 1e10 / h"`.
`apply_conversion` (`utils.py:788-797`) evaluates that string via `eval`,
with `x`, `a`, `h`, `Lbox` bound to their config/runtime values. This one
mechanism is what every comoving→proper, mass-unit, and box-relative
conversion across every catalogue actually runs through — see
[Data fetchers: shared mechanics](../02-data-generation-pipeline/01-shared-mechanics.md)
for the full pipeline this sits inside, and each fetcher's own page for
what its specific scaling strings evaluate to.

Two config-level facts worth knowing that cut across catalogues:

- **The gravitational constant `G = 4.30091e-6`** (kpc·Msun⁻¹·(km/s)²) is set once in `configs/data_generation.yaml`'s `pipeline.data_loader.G` — but the *same* value is also separately hardcoded in a debug helper (`utils.py:1110`). Two copies of the same constant means they could silently diverge if either is ever changed without touching the other.
- **Map field-of-view/bin-count**: `bin_count = int(fov / (spatial_resolution × bin_spatial_resolution_scale))` (`utils.muse_bins_r`), where `fov = 2 × transition_radius`. E.g. for a config with `spatial_resolution=1.480`, `bin_spatial_resolution_scale=1.5`: `bin_size = 1.480 × 1.5 = 2.22 kpc`. See [Map generation](../02-data-generation-pipeline/12-map-generation.md) for the fuller picture, including the open question of whether field of view is actually computed per-halo or held fixed per-catalogue in practice.

## Constants that exist but haven't been derived or cited anywhere

Worth resolving before quoting any of these in a paper: `_muse_pixelization`'s
`muse_n_bins=300` default; the `1000` sentinel value for low-particle-count
MUSE bins (see [MUSE instrument simulation](../02-data-generation-pipeline/11-muse-instrument-simulation.md));
`utils.py:892-899`'s `r500_to_r200 = 0.65` and `m500_to_m200 = 1.54` scaling
constants (stated as NFW-based in a comment, but no derivation or citation
given); the `±20`-bin random crop window (`map_step.py:278-297`).

## See also

- Part II's per-fetcher pages for actual per-catalogue values.
- [Known issues](../18-known-issues.md) for the specific config-value inconsistencies already found (Magneticum's `Lbox`, Millennium-TNG's `H`, etc.).

---
*Sourced from the SPACE-COE Technical Map §6, cross-referenced against this session's own direct config reading throughout Part II.*
