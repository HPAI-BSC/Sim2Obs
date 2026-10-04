# ICL/BCG classification & transition radius

All logic lives in `ICLProcessor` (`feature.py:708`), a
`FeatureProcessor` subclass with `name='IntraClusterLight'`. It performs
two distinct, sequential computations: (a) per-particle binary ICL/BCG
classification, and (b) a transition-radius calculation built on top of
that classification. Neither method, nor any other part of `feature.py`,
contains a citation to Dolag et al. 2010, Marini et al. 2022, or any
paper/arXiv/DOI reference — confirmed by a repo-wide grep. The paper's own
attribution of this method to those works isn't independently verifiable
from this repository alone; it's the collaboration's own domain knowledge,
not something encoded in the code.

## The classifier itself

A pre-trained model, loaded once per pipeline run from
`src/data_generation/model/Best_Model-Random-3par.pkl` via `joblib`.
Directly loading and inspecting this exact pickle file (not just reading
code that references it) confirms: `sklearn.ensemble.RandomForestClassifier`,
`n_estimators=500`, `criterion='gini'`, `max_features='auto'`,
`bootstrap=True`, `random_state=42`, `n_features_in_=3`,
`classes_=[0., 1.]`, and — notably — **no `feature_names_in_`**, meaning it
was fit on a plain array, not a labelled DataFrame. Nothing in the pickle
itself records what the 3 input columns mean or what order they must be
in.

**No training script, training dataset, or hyperparameter-search code
exists anywhere in this repository.** Only the frozen, pre-trained model
file is present — its provenance isn't independently verifiable from this
repo.

## (a) Per-particle classification

`_compute_icl_mask` (`feature.py:861`) computes the centre-of-mass velocity
once, then delegates the actual classification to `_icl_mask_for_slice`
(`feature.py:821`), which calls `_predict` (`feature.py:761`):

- `v_cm = np.average(particle_velocity, axis=0)` (in `_compute_icl_mask`, computed once over the whole halo) — the *unweighted* mean of the halo's non-overlapping star-particle velocities. Called "center-of-mass velocity" in the code's own comment despite not being mass-weighted.
- `distance = ||star_p||` (in `_icl_mask_for_slice`; Euclidean norm of the already-centred position).
- `v_delta = ||star_v - v_cm||` (also in `_icl_mask_for_slice`) — a scalar speed, not a signed/directional quantity.
- Exactly three features are built, in this fixed order — `distance`, `velocity`, `m200` — and passed to `self.model.predict(...)`. Since the pickle carries no `feature_names_in_`, **this fixed order in `_predict` is the only place in the whole system that fixes what the 3 numbers mean to the classifier.** Changing it would silently produce nonsense predictions with no error raised.

**Batching for very large halos** (added 2026-10): above
`icl_batch_threshold = 5_000_000` particles, `_compute_icl_mask` runs the
classifier in batches of `icl_batch_size = 2_000_000` instead of one pass
(both overridable via the feature config). The reason is memory: an
Obelisk halo with ~89M star particles OOM'd a single `_predict()` call even
in a 128G job. `v_cm` is still computed once over the full halo, so
batching changes memory use and wall time, not what gets computed.

**A latent indexing risk** (still present; with batching it now applies
within each slice rather than across the whole halo): rows with `inf`/`NaN` (only possible if
`distance == 0` exactly) are dropped via `.replace(...).dropna()` before
calling `.predict()`. `final_predictions` is therefore potentially shorter
than the original particle count — but `_compute_icl_mask` builds
`icl_mask` at the *original* particle count and indexes it with
`np.where(final_predictions == 0)[0]` as if those were still positions in
the original array. If `dropna()` ever removes a row, every subsequent
particle's ICL/BCG label would be silently shifted by one position. Not
verified whether `distance == 0` ever actually occurs on real data —
needs a targeted check before fully trusting per-particle labels near
halo centres.

**Class-convention flip**: the RF's own convention is `0 = ICL, 1 = BCG`
(per `_predict`'s docstring). `_compute_icl_mask` then *inverts* this for
its own return value: the returned `icl_mask` has `1 = ICL, 0 = BCG`. Two
opposite-looking but internally consistent conventions are in play across
these two methods — easy to misstate in prose without checking both.

Overlapping/interloper particles are **never classified at all** —
`process()` runs classification only on a halo's own non-overlapping star
particles; particles pulled in from overlapping neighbour subhalos are
appended to the mask as a block of zeros, i.e. treated as non-ICL by
construction, not by prediction.

If the RF predicts zero ICL particles for a halo, `_compute_icl_mask`
raises `ValueError`, and that halo is skipped upstream.

## (b) Transition radius

`_compute_transition_radius` (`feature.py:937`), only runs when
`self.transition_radius_compute` is true (the checked-in default config
sets this).

- Particles are sorted by `distance` — **not** divided by `R200`, despite the docstring saying positions are "normalize[d]... by the halo's `R200` radius." `R200`-normalization only appears in a debug log line, and is never returned or stored. The stored `self.transition_radius` is in the same physical (unnormalized) units as `star_p`.
- Two cumulative sums are built along that sorted order: `icl_cumulative = cumsum(icl_mask · mass)`, `bcg_cumulative = cumsum((1-icl_mask) · mass)` — **mass-weighted**, despite the docstring saying "cumulative *count* of ICL particles."
- The transition radius is picked not at the *first* crossing where `icl_cumulative > bcg_cumulative`, but at the position immediately after the **single largest gap** between consecutive crossing-indices — the start of the last, most sustained run of ICL-dominance, explicitly skipping any earlier, smaller, possibly-transient crossing. This specific rule isn't mentioned in any docstring or comment; it's only recoverable by reading the loop itself.
- If ICL mass never exceeds BCG mass at any radius, an `Exception` is raised and the halo is skipped.

## Equations

- `V200 = √(G·M200/R200)` — standard virial/circular-velocity relation. Unit check (from the per-catalogue config conventions): `G` is in kpc·Msun⁻¹·(km/s)², `M200` in `Msun/h`, `R200` in `kpc/h` — the `1/h` factors cancel exactly, leaving `V200` in plain km/s with no residual `h`-dependence.
- RF input features: `distance_feat = log10(r/R200)`, `velocity_feat = v_delta/V200` (linear, not logged), `m200_feat = log10(M200 · 1e-14)`. By the same cancellation, `r/R200` is also exactly `h`-independent.

## See also

- [Known issues](../18-known-issues.md).
- `appendix_impl_notes.tex` in the paper repo, notes N9–N11, N19–N20 — the same findings, written up for the paper's own implementation cross-check appendix.

---
*Sourced from the SPACE-COE Technical Map §15, which itself directly loaded and inspected the pickled model file (not just read code referencing it) — the strongest form of verification available for a frozen artifact like this.*
