# Map generation

## Field of view and pixel scale

The field of view is set from the transition radius (see
[ICL/BCG classification](10-icl-bcg-classification.md)): computed as
`2 × transition_radius`, in physical kpc, converted to a pixel count using
a physical (kpc) spatial-resolution value. **No angular (arcsec/arcmin)
conversion exists anywhere in this code path** — this pipeline does not
replicate an instrument's angular field of view the way, for example,
Marini et al. 2024 do (fixed angular FOV converted to physical units at a
single reference redshift). Here, both quantities are set directly in
physical units, independently per simulation: the pixel scale is matched
to that simulation's own particle/gravitational-softening resolution (see
each fetcher's config table, e.g. [Illustris family](02-fetcher-illustris-family.md)),
and the field of view tracks each halo's own transition radius.

A separate, more specific nuance: at least one catalogue's config marks
its field-of-view value with its own "to be recomputed" note, and as
checked in, the field of view used for each map is currently a **single,
fixed, per-catalogue config value**, not computed per halo at generation
time the way the `2 × transition_radius` design above implies it should
be. Whether that's a temporary simplification or the actual intended
behaviour hasn't been confirmed.

## Random zoom

Applied to introduce spatial variability: a random offset of **up to ±20
pixels/bins along each axis** — a *fixed bin count*, not a percentage of
image size. Since bin count varies by halo and catalogue, this
corresponds to a different percentage of image size in each case. (Worth
knowing if this is ever described in a paper as "a random offset within
N% of the image size" — that's not what the code does.)

## Output channels

- **ICL fraction map**: the fraction of intracluster stars per spatial bin, computed from the binary ICL/BCG classification mask (see [ICL/BCG classification](10-icl-bcg-classification.md)).
- **Velocity dispersion map**: the standard deviation of projected velocities within each pixel.
- **Interloper mask channel**: flags pixels containing particles from overlapping/neighbouring halos (see [neighbour search](07-neighbour-search.md)). At training time, this channel is used to zero out flagged pixels in both the input and label maps.

## See also

- [ICL/BCG classification & transition radius](10-icl-bcg-classification.md)
- [MUSE instrument simulation](11-muse-instrument-simulation.md)
- [Neighbour search](07-neighbour-search.md)
- [Known issues](../18-known-issues.md)

---
*Sourced from the SPACE-COE Technical Map §16, cross-checked against this session's own work verifying the same facts while writing the paper's methodology section (where this exact "no angular conversion" finding led to a correction in the paper's own draft text).*
