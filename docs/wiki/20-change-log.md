# Change log

Dated entries for fixes and decisions worth remembering the *why* of, not
a full commit history (see `git log` for that).

## 2026-10-04

- **ICL classification now runs in batches for very large halos.** Above 5,000,000 particles `_compute_icl_mask` classifies in batches of 2,000,000 (via the new `_icl_mask_for_slice`); `v_cm` is still computed once over the whole halo, so results are unchanged. Motivation: an ~89M-particle Obelisk halo OOM'd a single `_predict()` call in a 128G job. See [ICL/BCG classification](02-data-generation-pipeline/10-icl-bcg-classification.md).
- **Config/run settings changed:** `configs/data_generation.yaml` now activates `catalog: flamingo` and sets `skip_existing_halos: true`; `generate_dataset.py` processes all halos. The per-halo context CSV export in `pipeline.py` was commented out. See [Entry points](01-overview-architecture/02-entry-points.md).
- **Fixed the percentile outlier filter** (`lower_bound = 100 - upper_bound` mixed a radius with a percentage); the duplicate in `data_loader_step.py` was removed.
- Added a sustainability-plan document (`docs/space-coe-sustainability-plan.docx`, SPACE WP3 template).

## 2026-10-03

- **Magneticum: fixed `ValueError: setting an array element with a sequence` during map generation.** Parquet list columns (`POS`/`VEL`) came back as 1-D object arrays of 3-vectors. `star_p` was repaired later in `SubhaloParticles.preprocess`, but `VelocityDispersionProcessor.process` stored the un-repaired `star_v`, which the ICL step then rebound to a stacked copy, leaving `VelocityDispersionProcessor.value` stale. Fixed at the source with `MagneticumFetcherV2._column_to_numpy`, used by both particle loaders. Only Magneticum code is touched. Regression tests: `test_velocity_dispersion_object_star_v.py`. See [Magneticum page](02-data-generation-pipeline/04-fetcher-magneticum.md).
- **Magneticum: removed the V1 `MagneticumFetcher` class.** `MagneticumFetcherV2` subclassed it; the shared methods (`_get_halos`, `_store_particles`, the particle loaders, ...) were folded into the one remaining class, and V1's interval-based assignment, its `_get_pid_for_entity`/`_get_subhalo_pid_for_entity` helpers and the `catalog.name: magneticum` factory branch were deleted. `catalog.name` must now be `magneticum_v2` (the checked-in config already is). The tests that exercised only the removed V1 code were deleted from `test_fetcher_magneticum.py`; the `_store_particles` tests were ported to the streaming assignment.

## 2026-09-28

- **FLAMINGO: switched active config from `flamingo` (V1) to `flamingo_v2`.** `FlamingoFetcherV2` fixes a row-position vs. true-catalogue-index particle-assignment bug in V1 that affected essentially every halo in the L1000N0900 catalogue (verified: 2,000/2,000 scanned halos diverging). See [FLAMINGO page](02-data-generation-pipeline/05-fetcher-flamingo.md).
- Added `test_fetcher_flamingo_v2.py` (12 mock + 2 real-data tests) and `manual_test_flamingo_v2.py` (standalone smoke test, mirrors `manual_test_obelisk.py`'s pattern, with an added V1-vs-V2 comparison specific to this fetcher).

## 2026-09-27

- **Obelisk: switched halo/subhalo classification from `level == 1` to `is_central`.** Confirmed with the Obelisk team (Maxime Trebitsch) that `level` reflects AdaptaHOP spatial nesting, not gravitational boundedness or "satellite" status. Recovered 23 real halos (up to 1.7×10¹² M☉) that were previously being folded into another halo's `subhalo_ids`. See [Obelisk page](02-data-generation-pipeline/06-fetcher-obelisk.md).
- Updated `test_fetcher_obelisk.py`'s mock fixtures to include `is_central`; all 18 tests pass.

---
*Add entries here as things change — newest at the top.*
