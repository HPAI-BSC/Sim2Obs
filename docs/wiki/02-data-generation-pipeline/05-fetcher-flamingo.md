# Fetcher: FLAMINGO

`catalog.name: flamingo` → `FlamingoFetcher` (V1, buggy — see below).
`catalog.name: flamingo_v2` → `FlamingoFetcherV2` (fixed). As of this
writing, `configs/catalog/flamingo.yaml` is set to `flamingo_v2`.

## Format

FLAMINGO's catalogue is SOAP (SWIFT's halo/subhalo finder). Halos and
subhalos are **the same underlying table**, just filtered differently:
`soap.host_halo_index == -1` selects primaries (halos), all rows are
subhalos. Loaded eagerly via `swiftsimio` (`SWIFTDataset`) in `__init__`,
unlike Magneticum's lazy, cache-flag-gated loading.

## The V1 bug, and why V2 exists

SOAP exposes two non-interchangeable ID spaces for the same rows:

- **Row-position space**: a row's index within `self.fof_subhalos` as loaded. V1's `assign_id()` assigns exactly this (`np.arange(N)`) as `halo_id`/`subhalo_id`. `soap.host_halo_index` (satellite↔parent linkage) is *also* expressed in this space — confirmed on real data: interpreting `HostHaloIndex` as a direct row index gives physically sane satellite-to-parent distances (sub-Mpc); any other interpretation gives distances of hundreds of Mpc. So host/satellite linkage under V1's row-position convention is correct.
- **True catalogue-index space**: `input_halos.halo_catalogue_index` (halo table) and the per-particle `halo_catalogue_index` fields use a separate, shuffled numbering unrelated to row position — confirmed on real data: it is not `np.arange(N)`, and a halo's row-position id generally refers to an entirely different physical halo than whichever one's *true* catalogue index happens to equal that number.

**V1's bug**: `_get_p_halo`'s primary-particle mask
(`particles_of_type.halo_catalogue_index == halo.halo_id`) compares a
row-position `halo_id` directly against a true-catalogue-index particle
field, with no conversion — silently matching star particles to an
essentially unrelated halo elsewhere in the box. The satellite-gathering
code three lines below it in the same method already performs the correct
conversion; it just was never applied to the primary-particle line.

**V2's fix** (surgical, not a rewrite): convert the row-position id to its
true catalogue index at exactly the two points that compare against
particle data (`_get_p_halo`, `_get_p_subhalo`). Everything else
(`assign_id`, `_get_halos()`, `_get_subhalos()`, satellite-gathering) is
left untouched — it's internally consistent under the row-position
convention already.

### How much this actually matters

Verified with `src/data_generation/test/manual_tests/manual_test_flamingo_v2.py`
against the real L1000N0900 catalogue (snapshot 73, both locally and on
the MN5 cluster — identical results in both places): **all 2,000 scanned
halos** had row-position ≠ true catalogue index. This isn't an edge case;
V1 would misassign particles for essentially the entire catalogue. Sample
halos showed V1 and V2 disagreeing on star-particle counts by an order of
magnitude (e.g. halo 3: V1 counted 828, V2 counted 62).

## Test coverage

`test_fetcher_flamingo_v2.py` — 12 mock tests plus 2 real-data integration
tests (skipped automatically if the real catalogue isn't reachable; on
this machine it is). Includes a direct side-by-side
(`test_v1_reproduces_the_false_positive_v2_fixes`) that runs V1's
unmodified logic against a deliberately-diverging synthetic dataset to
prove the contrast is real, not just asserted.

## Config values

| | Value |
|---|---|
| Snapshot (z) | 73 (z=0.2) |
| `a` | 0.8333333333333335 |
| `H` | 0.6774 — the config's own comment flags this as `"Different from theirs"`, i.e. known to differ from FLAMINGO's own published cosmology |
| `Lbox` (kpc/h) | 677,400 |
| Transition radius (kpc) | 429,352.02 — 2–3 orders of magnitude larger than every other catalogue's (400–700 kpc range); not yet explained, worth double-checking before trusting FLAMINGO's field-of-view sizing |
| Spatial resolution (kpc) | 22.3 ("from the documentation 'comoving' spatial resolution") |
| Halo mass filter | `> 1e13*0.6774` |

## Open items

- The `H = 0.6774` vs. FLAMINGO's own published cosmology mismatch, flagged in the config's own comment, is unresolved.
- The transition radius's order-of-magnitude outlier value hasn't been explained or re-derived.
- FLAMINGO's SOAP catalogue is read as authoritative for VELOCIraptor-family structure identification; confirm whether "VELOCIraptor" or "SOAP" is the right term to use when the paper describes this.

## See also

- [Shared mechanics](01-shared-mechanics.md) for the generic per-field pipeline this fetcher also runs through.
- [Known issues](../18-known-issues.md).

---
*Written from this session's own direct investigation and verification (real-data testing, both locally and on MN5), not solely the original technical map — the map's own note on this fetcher was "I did not trace FlamingoFetcherV2's diff in the same depth as Magneticum's V1→V2 change," which this page supersedes.*
