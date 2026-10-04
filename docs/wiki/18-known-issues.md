# Known issues & open questions

A living, consolidated list. Anything here that gets resolved should move
to the relevant page's own docs (and to [the change log](20-change-log.md))
rather than just being deleted from here silently.

Status legend: 🔴 open, 🟡 answered but not yet acted on, 🟢 resolved.

## Cosmology mismatches

- 🔴 **Millennium-TNG**: config `H = 0.6774`, but the paper's own cosmology table states `H0 = 73.00` (`h = 0.73`). Real discrepancy, not rounding. See [Illustris family page](02-data-generation-pipeline/02-fetcher-illustris-family.md).
- 🔴 **EAGLE**: config `H = 0.6774` vs. the paper's table's `67.77` — small but real.
- 🔴 **FLAMINGO**: config's own comment already flags `H = 0.6774` as `"Different from theirs"` (FLAMINGO's own published cosmology). See [FLAMINGO page](02-data-generation-pipeline/05-fetcher-flamingo.md).
- 🔴 **Magneticum**: the halo mass filter literal (`> 1e13*0.6774`) uses the *other* catalogues' `H`, not Magneticum's own (`0.704`) — looks copied from elsewhere rather than derived correctly.

## Config-value inconsistencies

- 🔴 **Magneticum `Lbox`**: set to `290294.4`, but the config's own adjacent comment says the raw box size from the snapshot header is `352000.0` (ckpc/h). `352000 × 0.8247` (the file's own `a`) = `290294.4` exactly — `Lbox` looks like it already has the `a` scaling pre-applied, while every other field in the same file (`RCRI`, `GPOS`, `SPOS`, etc.) expects the raw comoving value and applies `x * a` itself. Email sent to a collaborator asking whether `Lbox` should be `352000` instead.
- 🔴 **Magneticum & Millennium-TNG `spatial_resolution`**: both explicitly marked `TODO review`/`TODO BE REVIEWED` in their own config files.
- 🔴 **`bigillustris300.yaml`'s `spatial_resolution: 64**3`**: a bare unevaluated arithmetic expression, not a number. Self-flagged by its author: `"kpc ? TODO verify with Sebastian"`.
- 🔴 **FLAMINGO's transition radius** (429,352.02 kpc) is 2–3 orders of magnitude larger than every other catalogue's (400–700 kpc range). Not yet explained.
- 🔴 **`eagle.yaml` excludes one halo by ID** (`ignore_halos: [26000000000009]`) with no comment explaining why.
- 🟡 **Magneticum's transition radius** (499.39 kpc) is labeled "Recomputed" in the config, implying it's been validated; Millennium-TNG's (703.578 kpc) carries no such note — worth confirming it's had the same treatment, not just a first-pass number.

## Obelisk (awaiting the Obelisk team's reply)

See the [Obelisk page](02-data-generation-pipeline/06-fetcher-obelisk.md) for full detail. Open:
- Comoving vs. physical convention for `rvir`/`r50_kpc`.
- Whether star-particle `velocity` needs a correction the way `position` did.
- Timeline (if any) for the Obelisk team's own flagged velocity-dispersion bug.

🟢 **Resolved**: `level` vs. `is_central` for halo definition — confirmed with the Obelisk team, switched, verified against real data (recovered 23 previously-miscounted halos).

## Structural gaps

- 🔴 **DIANOGA has no data-generation code at all.** It's loaded pre-generated as `.npz` files at training time; how those maps were originally produced isn't established by anything in this repository. If the paper describes DIANOGA as going through this pipeline, that claim needs a different source or a caveat.
- 🔴 **"Gray area" (from the technical map, not yet re-verified this session)**: whether `StellarMassDensityProcessor`/`DMMassDensityProcessor` are used by the pipeline at all — neither appears in the checked-in `feature_names` config, though `generate_map` implements support for them.

## Magneticum — resolved

🟢 **Object-dtype `star_v` crashed map generation** (`ValueError: setting an array element with a sequence`, e.g. halo 19980). Fixed in `MagneticumFetcherV2._column_to_numpy`; see the [change log](20-change-log.md).

🟢 **V1 `MagneticumFetcher` removed** — only `magneticum_v2` exists now.

🟢 **Percentile outlier filter used `100 - upper_bound`** (a radius in kpc subtracted from 100, so the lower cut did nothing). The lower bound is now `np.percentile(x, 100 - percentile)` in `test_utils.py`, and the standalone diagnostic that duplicated the bug was removed from `data_loader_step.py`.

## FLAMINGO — resolved this session

🟢 **V1's row-position vs. true-catalogue-index particle-assignment bug.**
`FlamingoFetcherV2` fixes it, fully tested (12 mock + 2 real-data tests),
verified at production scale on the real MN5 cluster: 100% of a 2,000-halo
scan had row-position ≠ true catalogue index, meaning V1 was silently
wrong for essentially the entire catalogue. `configs/catalog/flamingo.yaml`
now points at `flamingo_v2`. See the [FLAMINGO page](02-data-generation-pipeline/05-fetcher-flamingo.md)
and the [change log](20-change-log.md).

---
*This page is meant to be edited directly as things get resolved — it's not generated from anything.*
