# Entry points

Three independent Hydra (`@hydra.main`) scripts, each its own pipeline
stage — there's no single "run everything" command.

| Script | Config | Launcher | Role |
|---|---|---|---|
| `src/generate_dataset.py` | `configs/data_generation.yaml` | `launch_data_generation.sh` | Builds the `PipelineStep` chain from `cfg.pipeline`, runs `Pipeline.run()` |
| `src/train.py` | `configs/train.yaml` | `launch_train.sh` | Instantiates `cfg.data`/`cfg.model` via Hydra, runs PyTorch-Lightning `Trainer.fit`/`.test` |
| `src/eval.py` | `configs/eval.yaml` | `launch_eval.sh` | Loads a checkpoint, runs inference/prediction saving |

## Data-generation step chain

`Pipeline`/`PipelineStep` (`src/data_generation/src/pipeline.py`,
`pipeline_step.py`) is a continuation-passing chain:
`PipelineStep.process(gen, next_step, **kwargs)`. `Pipeline.build_pipeline`
(`pipeline.py:144-196`) recursively wires `steps[i].process(gen,
next_step_callback, ...)`, giving each step control over fan-out (e.g. a
rotation step could call `next_step` once per rotated view) and its own
error accumulation.

Steps are selected by name in `configs/data_generation.yaml`'s
`pipeline:` list:

- **`data_loader`** (`DataLoader`, always first) — fetches halo/subhalo catalogues and particle data.
- **`rotate_step`** (`RotateStep`, optional) — applies rotations, multiplying each halo into several rotated views. See [Rotation](../02-data-generation-pipeline/09-rotation.md) — not currently configured anywhere, so this step doesn't run in practice right now.
- **`map_step`** (`MapStep`, optional) — bins particles into 2D maps and writes `.npz` files.
- **`LastStep`** (`src/data_generation/src/last_step.py`, always added automatically by `Pipeline.run`, not itself listed in `cfg.pipeline`) — aggregates errors and finalizes output.

**Currently active, as checked in**: `configs/data_generation.yaml`'s
`pipeline:` list has only `data_loader` uncommented (`map_step` is present
but commented out; `rotate_step` isn't in the list at all). As configured,
a run performs catalogue loading and feature computation but not map
generation, unless overridden at invocation time.

**Other current settings** (`configs/data_generation.yaml`, as of 2026-10-04):
the active catalog is `flamingo` (it has been switched between
`magneticum`, `flamingo` and `obelisk` as runs were launched, so always
check before assuming); `skip_existing_halos: true`, so already-generated
halos are skipped on re-runs; and `src/generate_dataset.py` processes *all*
halos (`starting_idx`/`ending_idx` are `None`).

## Training entry

`src/train.py`: the standard Lightning-Hydra template —
`hydra.utils.instantiate(cfg.data)` → `LightningDataModule`,
`hydra.utils.instantiate(cfg.model)` → `LightningModule`, then
`trainer.fit(...)`/`trainer.test(...)`. `configs/train.yaml` defaults to
`data: combined_dataset`, `model: mamba_unet`.

## See also

- [End-to-end data flow](03-data-flow.md) for what actually happens inside each pipeline step.

---
*Sourced from the SPACE-COE Technical Map §1, with the active `pipeline:` list re-verified directly against the current `configs/data_generation.yaml` this session.*
