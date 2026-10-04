# Training pipeline & the `LightningModule` pattern

Covers `src/train.py` and the six `LightningModule` variants in
`src/models/` (`bce_module.py`, `mse_module.py`, `mae_module.py`,
`vae_module.py`, `gaussian_interp_module.py`, `mamba_module.py`).

## Input

Pairs `(x, y)` from a `LightningDataModule`: `x` = a velocity-dispersion
map, `y` = the corresponding ICL target map, both pre-generated `.npz`
files from the data-generation pipeline (see
[End-to-end data flow](../01-overview-architecture/03-data-flow.md)).

### The active default training set is smaller than you might expect

`configs/train.yaml`'s `data: combined_dataset` selects `CombinedDataset`
(`src/data/components/combined_dataset.py`), which concatenates
sub-datasets by matching each configured path against the substrings
`'dianoga'`/`'ilarias'`, `'illustris'`, or `'eagle'` — **any other path
silently yields `dataset = None` and is dropped, with no warning.**

The actually-configured list, in `configs/data/combined_dataset.yaml`, is:

- `eagle/RefL0050N0752/rotation_45_s1.5` (EAGLE, reference sub-grid physics)
- `eagle/AGNdT9L0050N0752/rotation_45_s1.5` (EAGLE, AGN sub-grid-physics variant)
- `illustris300/rotation_45_s1.5_v4` (Illustris-300, rotated)
- `illustris300/no_rotation_s1.5` (Illustris-300, unrotated)

...and a fifth line, `ilarias_dataset_v0` (DIANOGA), is present but
**commented out**. As checked into this repository, **the default
training set spans exactly 2 of the six simulations covered by this
project's own fetchers (EAGLE and Illustris-300), each via 2 variants** —
DIANOGA disabled, Magneticum/FLAMINGO/Millennium-TNG/TNG-Cluster absent
from the list entirely. If a paper reports training across more
simulations than this, that's either a different config than what's
checked in, or worth double-checking.

Also from this config: `resize_shape: [128, 128]`, `batch_size: 32`,
`train_val_test_split: [.85, .05, .10]`, `random_seed: 0`,
`mask_interlopers: True`.

### A real plumbing gap: `mask_interlopers` doesn't do what the config implies

`CombinedDataset.__init__(self, image_dir, labels_dir, transform=None,
fraction_data=1, **kwargs)` — verified directly. `mask_interlopers` isn't
a named parameter, so the config's `mask_interlopers: True` lands in
`**kwargs` and is **never forwarded** to any child dataset (`DianogaDataset`,
`Illustris300Dataset`, `EagleDataset` are all constructed with only
`transform=transform`). The masking that *does* happen comes entirely
from `Illustris300Dataset`/`EagleDataset`'s own internal default of
`mask_interlopers=True` in their own constructors — which happens to
match the config's value today. **Setting `mask_interlopers: False` in
this config would silently fail to disable masking** for the combined
dataset.

## Interloper masking at load time

`Illustris300Dataset.__getitem__` (and the equivalent in `EagleDataset`):
each saved map has at least 2 channels — channel 0 the physical feature,
channel 1 an interloper-flag map. When enabled:
`image = image[...,0] * (1 - image[...,1])`, applied identically to both
input and label — zeroing any pixel flagged as an interloper in *both*
the velocity-dispersion input and the ICL target. The mask itself is
baked into channel 1 of the saved `.npz` files by the data-generation
pipeline (see [Map generation](../02-data-generation-pipeline/12-map-generation.md)),
not computed here.

## The `LightningModule` pattern

All six module files follow the same structure — the standard
lightning-hydra-template shape: `training_step`/`validation_step`/
`test_step` each call a per-module `model_step(batch)` that runs the
forward pass and computes one scalar loss, then update/log a fixed set of
`torchmetrics` objects.

**What actually differs between modules is `model_step` itself** — see
[Loss functions & evaluation metrics](03-loss-functions-and-metrics.md)
for the full per-module breakdown, including the non-obvious fact that
the active default (Mamba architecture) is trained with a BCE-style loss,
not a direct regression loss.

## Checkpoint selection

Best-checkpoint selection is by `val/loss` — whichever loss the active
module computes (BCE for the default), *not* by `val/RASE` or
`val/masked_mae`, even though those are also tracked and arguably closer
to a physically meaningful evaluation metric. Worth confirming this is
the intended selection criterion if RASE/MaskedMAE are what get reported.

## Outputs

- Per-batch/epoch logged metrics (train/val/test × loss, RASE, rmse, mae, masked_mae, plus `*_best` running-minimum trackers), logged via `logger: mlflow`.
- Model checkpoints written by Lightning's checkpoint callback.
- At test time, the same metric suite computed on the held-out 10% split.

## See also

- [Model architecture catalog](01-model-architecture-catalog.md)
- [Loss functions & evaluation metrics](03-loss-functions-and-metrics.md)

---
*Sourced from the SPACE-COE Technical Map §17. The `mask_interlopers` plumbing gap and the active model's BCE-on-sigmoid loss were both re-verified directly against the current code this session (`CombinedDataset.__init__`'s actual signature, `mamba_module.py`'s `model_step`).*
