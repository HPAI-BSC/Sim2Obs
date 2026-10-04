# Periodic boundary correction

## The correction itself

`apply_pbc_correction` — implemented as near-identical closures inside
both `SubhaloContext.preprocess` and `SubhaloParticles.preprocess`
(`subhalo.py:127-205` and `653-702`; duplicated code, not shared — see
"software engineering, not scientific" note below). For a raw offset
`delta = position - centre`:

```
if |delta| > Lbox/2:  delta' = delta - Lbox * sign(delta)
else:                 delta' = delta
```

The standard minimum-image convention, applied independently per
coordinate axis.

## A real, unresolved coordinate-frame issue this doesn't cover

This is a genuine gap, found while tracing how a halo's particles get
combined with its neighbours' particles for interloper handling (see
[neighbour search](07-neighbour-search.md)):

- The **central halo's own particles** are centred via `halo_particles_no_overlapping.preprocess(halo, lbox)` — i.e. relative to *the halo currently being processed*'s own `gpos`.
- **Each overlapping subhalo's particles** are centred inside `_load_particles_overlapping_subhalos` (`data_loader_step.py:442-519`) by resolving `halo_id` = that overlapping subhalo's *own parent halo's* id (a *different* halo — that's what makes it "overlapping"), fetching that halo, and calling `subhalo_particles_i.preprocess(halo, lbox)` — i.e. centred on **that other halo's own `gpos`**, not the central halo's.
- These two sets are then combined via `SubhaloParticles.__add__` (`subhalo.py:592-651`), which **only concatenates the raw position arrays** — no coordinate shift is applied to reconcile the two different centres.

**Net effect**: the combined particle cloud used for interloper
masking/ICL classification mixes positions expressed relative to two (or
more, if several distinct neighbouring halos contribute) different
physical origins, without reconciling them into one common frame. This
looks like a coordinate-frame inconsistency on its face. It hasn't been
run against real data to observe its practical size, and there may be an
intentional design reason not visible from the code alone — this needs
author confirmation before treating it as either a bug or a deliberate
choice. (Tracked as N13 in the paper's implementation cross-check
appendix.)

## Software engineering, not scientific

The PBC-correction closure is written independently in
`SubhaloContext.preprocess` and `SubhaloParticles.preprocess` rather than
factored into one shared function — a code-organisation detail, not a
correctness concern (both copies implement the same logic).

## See also

- [Neighbour search](07-neighbour-search.md) for how a subhalo gets selected as a neighbour in the first place.
- [Known issues](../18-known-issues.md).

---
*Sourced from the SPACE-COE Technical Map §16(b)–(c), cross-checked against this session's own reading of the same code while writing the paper's methodology section.*
