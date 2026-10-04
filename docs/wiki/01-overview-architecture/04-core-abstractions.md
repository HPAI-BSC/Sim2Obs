# Core abstractions

Three patterns show up everywhere in this codebase. Understanding them
first makes everything else easier to read.

## `DataFetcher` (`src/data_generation/src/data_fetcher.py`)

An abstract base class (`class DataFetcher(ABC)`, `data_fetcher.py:422`)
declaring one interface that every simulation-specific subclass
implements:

```python
get_halos(...)      -> HaloContext
get_subhalos(...)   -> SubhaloContext
get_p_subhalo(...)  -> SubhaloParticles
get_p_halo(...)     -> SubhaloParticles
```

Five structurally unrelated upstream formats (Gadget binary, two
different HDF5 group-catalogue schemas, a SQL database, a SWIFT/SOAP HDF5
schema, an AdaptaHOP HDF5 table) converge on this one interface. That's
the actual point of the abstraction — it's what lets rotation, map
generation, and ICL classification run completely unmodified regardless
of which simulation the data came from. See
[Data fetchers: shared mechanics](../02-data-generation-pipeline/01-shared-mechanics.md)
for the dispatch table and the generic per-field pipeline every subclass
also shares, and Part II's per-simulation pages for what's specific to
each.

`DataFetcherFactory.get_data_fetcher` is the single place that decides
which subclass a given config actually instantiates, based on the
config's own `catalog.name` string — not the YAML filename.

## `HaloContext` / `SubhaloContext` (`halo.py`, `subhalo.py`)

Thin wrappers around a `pandas.DataFrame` holding the filtered,
unit-converted halo or subhalo table. Two things they consistently do:

- **Field renaming**: each catalogue's native field names get mapped to this pipeline's common names via that catalogue's own `column_map` (e.g. `Group_R_Crit200` → `r200`, `GroupFirstSub` → `fsub`). The renaming itself is generic; what a given name actually *means* can still differ subtly by catalogue — see each fetcher's page for specifics (e.g. `fsub` is a real field for Magneticum/Millennium-TNG but a meaningless placeholder for Obelisk/FLAMINGO).
- **Parent linkage via `id_parents`**: every `SubhaloContext` carries an `id_parents` column used to look up "all subhalos belonging to halo X." How that column gets populated is genuinely catalogue-specific and is one of the more bug-prone parts of this codebase in practice — see [Obelisk's `is_central` fix](../02-data-generation-pipeline/06-fetcher-obelisk.md) and [FLAMINGO's row-position bug](../02-data-generation-pipeline/05-fetcher-flamingo.md) for two real examples of this going wrong and getting fixed.

`SubhaloParticles` (also in `subhalo.py`) is the equivalent wrapper for
per-particle arrays (position, velocity, mass), and is what
`preprocess()` centres on a halo and periodic-boundary-corrects — see
[Periodic boundary correction](../02-data-generation-pipeline/08-periodic-boundary-correction.md).

## The `PipelineStep` chain (`pipeline_step.py`, `pipeline.py`)

An abstract base class with one method:

```python
class PipelineStep(ABC):
    @abstractmethod
    def process(self, g: Any, next_step: Callable[[Any], Any], **kwargs):
        ...
```

This is continuation-passing style, not a plain linear list: each step
receives `next_step` as a callable and decides itself when (and how many
times) to call it. That's what lets `RotateStep` turn one halo into
several rotated views by calling `next_step` once per rotation, and what
lets each step accumulate its own errors independently rather than one
step's failure aborting the whole chain. `Pipeline.build_pipeline`
(`pipeline.py:144-196`) wires the steps listed in a config's `pipeline:`
list together this way, and `Pipeline.run` appends `LastStep`
automatically at the end regardless of what's configured. See
[Entry points](02-entry-points.md) for the actual step list and what's
currently active.

## See also

- [Entry points](02-entry-points.md)
- [End-to-end data flow](03-data-flow.md)
- [Data fetchers: shared mechanics](../02-data-generation-pipeline/01-shared-mechanics.md)

---
*Class signatures (`DataFetcher`, `PipelineStep`) read directly from the current code this session. Narrative framing draws on the SPACE-COE Technical Map §1–2 and this session's own Part II work.*
