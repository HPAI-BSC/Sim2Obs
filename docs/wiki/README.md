# space-coe wiki

A narrative guide to the `space-coe` codebase: how data flows from raw
simulation catalogues to trained ICL-prediction models, why the pipeline is
shaped the way it is, and where the known rough edges are.

This is different from `docs/` (the Sphinx API reference generated from
docstrings) — that's the *what does this function take/return*. This wiki
is the *why does this exist, how do the pieces fit together, and what do
we still not know for sure*.

Status: **complete.** All parts (I–V) are written.

## Part I — Overview & Architecture
1. [How to use this wiki](01-overview-architecture/01-how-to-use-this-wiki.md)
2. [Entry points](01-overview-architecture/02-entry-points.md)
3. [End-to-end data flow](01-overview-architecture/03-data-flow.md)
4. [Core abstractions](01-overview-architecture/04-core-abstractions.md)

## Part II — Data Generation Pipeline
5. [Data fetchers: shared mechanics](02-data-generation-pipeline/01-shared-mechanics.md)
6. Per-simulation fetcher reference
   - [Illustris-TNG / TNG-Cluster / Millennium-TNG](02-data-generation-pipeline/02-fetcher-illustris-family.md)
   - [EAGLE](02-data-generation-pipeline/03-fetcher-eagle.md)
   - [Magneticum](02-data-generation-pipeline/04-fetcher-magneticum.md)
   - [FLAMINGO](02-data-generation-pipeline/05-fetcher-flamingo.md)
   - [Obelisk](02-data-generation-pipeline/06-fetcher-obelisk.md)
7. [Neighbour search & overlapping-halo handling](02-data-generation-pipeline/07-neighbour-search.md)
8. [Periodic boundary correction](02-data-generation-pipeline/08-periodic-boundary-correction.md)
9. [Rotation / viewing-angle augmentation](02-data-generation-pipeline/09-rotation.md)
10. [ICL/BCG classification & transition radius](02-data-generation-pipeline/10-icl-bcg-classification.md)
11. [MUSE instrument simulation](02-data-generation-pipeline/11-muse-instrument-simulation.md)
12. [Map generation](02-data-generation-pipeline/12-map-generation.md)

## Part III — Model Training
13. [Model architecture catalog](03-model-training/01-model-architecture-catalog.md)
14. [Training pipeline & the `LightningModule` pattern](03-model-training/02-training-pipeline.md)
15. [Loss functions & evaluation metrics](03-model-training/03-loss-functions-and-metrics.md)

## Part IV — Infrastructure & Operations
16. [Configuration system](04-infrastructure-operations/01-configuration-system.md)
17. [HPC / cluster operations](04-infrastructure-operations/02-hpc-cluster-operations.md)
18. [Known issues & open questions](18-known-issues.md)

## Part V — Appendix
19. [Glossary of fields & units](19-glossary.md)
20. [Change log](20-change-log.md)

## Sources

Sections 1–17 above are built from the SPACE-COE Technical Map (a
code-verified deep-dive produced earlier), cross-checked and extended
against the current state of the repo. Sections 18–21 of that original
map were draft paper text for `main.tex`, not codebase documentation —
that content has since been written directly into the actual paper
(`introduction.tex`/`methodology.tex`/`appendix_impl_notes.tex`) and isn't
part of this wiki.
