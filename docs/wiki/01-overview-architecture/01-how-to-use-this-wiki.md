# How to use this wiki

## What this is, and isn't

This wiki explains *why* the `space-coe` codebase is shaped the way it
is — the data flow, the design decisions, and the places where the code
doesn't do quite what you'd assume from reading a paper or a variable
name. It is not:

- **API reference.** For exact function signatures and docstrings, see `docs/` (a separate, auto-generated Sphinx build from the source docstrings — `docs/api/`).
- **A tutorial.** It assumes you can read Python and Hydra configs; it doesn't walk through "how to run your first job."
- **A finished, self-consistent document.** Several pages explicitly flag things that are unconfirmed, inconsistent, or actively being asked about with collaborators. That's deliberate — the goal is to be honest about what's actually known versus assumed, not to read as more polished than the codebase itself currently is.

## How pages are written

Every page ends with a short source note explaining where its claims come
from — direct code reading in a specific session, the original SPACE-COE
Technical Map, or a collaborator's confirmation. Where a page's own
verification found the technical map's original numbers to be stale (this
has happened more than once — see the [FLAMINGO](../02-data-generation-pipeline/05-fetcher-flamingo.md)
and [Obelisk](../02-data-generation-pipeline/06-fetcher-obelisk.md) pages,
and [Magneticum's `Lbox`](../02-data-generation-pipeline/04-fetcher-magneticum.md)),
the page says so explicitly and uses the newer facts.

If you find something here that's since changed in the code, please
update the page rather than working around the stale claim silently —
the [known issues](../18-known-issues.md) and [change log](../20-change-log.md)
pages exist specifically to track that kind of drift.

## Where to start

- New to the codebase? Read [Entry points](02-entry-points.md), then [End-to-end data flow](03-data-flow.md).
- Working on a specific simulation's ingestion? Go straight to [Part II](../02-data-generation-pipeline/01-shared-mechanics.md)'s per-simulation pages.
- Wondering "is this a known problem?" — check [Known issues](../18-known-issues.md) first.

---
*Not sourced from the technical map — written directly for this wiki.*
