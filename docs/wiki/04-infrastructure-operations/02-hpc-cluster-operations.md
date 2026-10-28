# HPC / cluster operations

Runs on MN5 (a Barcelona Supercomputing Center cluster — `account bsc70`,
paths under `/path/to), via three SLURM launch
scripts at the repo root.

## SLURM resources per job

Verified directly against the current launch scripts this session (one
figure below has changed since the original technical map was written —
noted explicitly):

| Job | Script | SLURM resources | Container |
|---|---|---|---|
| Data generation | `launch_data_generation.sh` | 1 node, **64 cores**, no GPU, 10h walltime, QoS `gp_bsccs` | `space_data_generation_v2.sif` |
| Training | `launch_train.sh` | 1 node, 32 cores, 1 GPU, 2h walltime, QoS `acc_debug` | `space_v33.sif` |
| Eval | `launch_eval.sh` | 1 node, 32 cores, 1 GPU, 15min walltime, QoS `acc_debug` | `space_v33.sif` |

**The data-generation core count is 64, not 32** — this has been bumped
since the original technical map was written, and the script's own
comment explains exactly why: *"MN5 has no `--mem` flag; memory is
auto-set per core (GPP: 2G/core), so 64 cores → 128G total. Obelisk halos
can have tens of millions of star particles."* Worth knowing if you're
about to launch a data-generation job and wondering why it asks for so
many cores on a job with no parallel compute step — it's a memory
workaround, not a throughput one, and it's specifically there because of
Obelisk.

The same constraint is why the ICL classifier now runs in batches for very
large halos (above 5M particles; see
[ICL/BCG classification](../02-data-generation-pipeline/10-icl-bcg-classification.md)):
a single pass over an ~89M-particle halo exceeded even the 128G this job
gets.

## Distributed training

`configs/trainer/`: `ddp.yaml` configures single-node 4-GPU DDP
(`sync_batchnorm: True`); `ddp_sim.yaml` simulates DDP on 2 CPU processes
for debugging. **The actually-launched job only requests 1 GPU**
(`launch_train.sh`'s `--gres=gpu:1`) — multi-GPU DDP is available in
config but isn't what the default launcher uses. Worth checking which
trainer config was actually selected for any specific reported results.

**Precision**: no `precision` value is set anywhere in the trainer
configs (only present, commented out, in `default.yaml`) — training runs
at PyTorch Lightning's default precision unless overridden at the command
line.

## Memory-bounded streaming design (Magneticum)

`MagneticumFetcherV2` deliberately avoids building an in-RAM global
particle-to-halo index — an earlier version did exactly that and was
OOM-killed in production under this cluster's memory cap. See
[Magneticum's fetcher page](../02-data-generation-pipeline/04-fetcher-magneticum.md)
for the full story (including the discarded intermediate approach and why
it wasn't viable either). This is a wall-clock-cost tradeoff, not a
scientific-results one, unless a paper specifically discusses
computational scalability.

## Containers

Two separate build systems, for two different purposes:

- **`singularity/*.def`** (7 files: `space_data_generation.def`, `space_data_generation_v3_mocks.def`, `space_v2.def` through `space_v6_mocks.def`) — the images actually used by the SLURM launch scripts above. Training containers pin `torch==2.1.2`, `lightning==2.2.1`, `hydra-core==1.3.2` (verified directly against `space_v5.def`/`space_v6_mocks.def`).
- **`containers/Dockerfile-local` and `containers/Dockerfile-amd`** — a primary NVIDIA/CUDA path and an AMD-ROCm variant, each with their own `requirements-*.txt`.

If a paper needs to state exact software versions for reproducibility,
the Singularity `.def` files (not the Dockerfiles) are what's actually
running the reported jobs.

---
*Sourced from the SPACE-COE Technical Map §8. SLURM resource requests, container file locations/count, and version pins were all re-verified directly against the current launch scripts and `singularity/*.def` files this session — the 32→64 core change for data generation was caught this way.*
