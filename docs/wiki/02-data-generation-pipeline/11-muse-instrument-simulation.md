# MUSE instrument simulation

`FeatureProcessor._apply_muse_filter` (`feature.py:318-488`), gated by
`self.simulate_muse`. **Off by default** — the checked-in
`configs/data_generation.yaml` doesn't set `simulate_muse`, so this whole
code path is inactive unless explicitly enabled.

## What it does, per spatial bin

After `_muse_pixelization` rebins the high-resolution grid down to
`muse_n_bins` (default 300):

- **Bin has more than `n_particles_threshold` (default 100) velocity samples**: build a 1D histogram (`⌈√n⌉` bins), pass it through `convolve_and_pixelize_from_bins_1d` with `FWHM=2.5` — literally commented `# ToDo: Adjust to MUSE` in the code — to simulate the Line Spread Function (a Gaussian convolution), re-pixelize, then fit Gaussian moments for a mean/std. The `FWHM` is a self-flagged placeholder, not a value derived from MUSE's actual instrument spec.
- **Bin has fewer than `n_particles_threshold` samples**: output a hardcoded sentinel value, `1000` — not a physically meaningful velocity dispersion, a flag value.
- **Bin has *exactly* `n_particles_threshold` particles**: matches neither the `>` nor the `<` branch, and is left at its initialised value of `0` — a directly observable edge-case gap in the threshold logic.

## Dead code in this path

- A substantial alternative "grow region" scheme — merging neighbouring under-populated bins via a BFS queue before histogramming, so sparse bins get *some* usable statistic instead of the `1000` sentinel — exists in the file but is entirely wrapped in a triple-quoted string and **never executes**. Confirmed dead by direct reading, not merely commented-out lines.
- `avg_pool_2x2` (a final 2× downsampling back toward original resolution) is called out in a comment but is itself commented out.

## See also

- [Map generation](12-map-generation.md) — how field of view and pixel scale are set for the maps this step operates on.
- [Known issues](../18-known-issues.md).

---
*Sourced from the SPACE-COE Technical Map §16(e), cross-checked against this session's own reading of the same code while writing the paper's methodology section.*
