# Loss functions & evaluation metrics

## `model_step` differs by module — this is where the real methodological choice lives

All six `LightningModule`s share the same training-loop scaffolding (see
[Training pipeline](02-training-pipeline.md)); what differs is each
module's own `model_step`:

| Module | `model_step` computation | Sigmoid applied? |
|---|---|---|
| `BCELitModule` | `loss = BCE(sigmoid(preds), sigmoid(y))` | Yes, to both preds and target |
| `MSELitModule` | `loss = MSELoss()(sigmoid(preds), sigmoid(y))` | Yes, to both — even though plain MSE doesn't require a bounded domain |
| `MAELitModule` | `loss = MaskedMAE()(preds, y)` (the same custom metric class also used for evaluation, below) | No |
| `VAELitModule` | `loss = BCE(sigmoid(x_hat), sigmoid(x)) + KLD(mean, log_var)` | Yes, on the reconstruction term only |
| `GaussianInterpLitModule` | `loss = self.net.train_step(x, y)` — delegated entirely into the Gaussian-Interp network component, not traced further | Not established |
| `MambaLitModule` (**active default**, `configs/train.yaml`: `model: mamba_unet`) | `loss = BCE(sigmoid(preds), sigmoid(y))` — identical to `BCELitModule` | Yes, to both |

**Consequence**: the loss function is fully decoupled from the
architecture — any `net` can in principle be paired with any of these
`LightningModule`s via Hydra config. **The active default combination
(Mamba architecture + BCE-style loss) trains by treating both the
predicted and target pixel maps as if they were sigmoid-transformed
probabilities, minimising binary cross-entropy between them — not a
direct regression loss (MSE/MAE) on the physical map values — despite the
task being framed as predicting a continuous-valued map.** This is a
specific, non-obvious, directly-verified fact (confirmed this session
against `mamba_module.py`'s actual `model_step`): worth knowing before
describing the training objective in a paper.

## Equations

**BCE-after-sigmoid** (active default). For prediction `p` and target `t`
(raw network/map values), with `p' = sigmoid(p)`, `t' = sigmoid(t)`:

```
L = -mean[ t'·log(p') + (1-t')·log(1-p') ]
```

**VAE loss (ELBO)** (`vae_module.py:30-34`) — standard
reconstruction-plus-KL-divergence-to-a-unit-Gaussian-prior:

```
L = BCE(σ(x̂), σ(x)) - (1/2)·Σ(1 + log(σ²) - μ² - σ²)
```

**MaskedMAE** (used both as a metric across every module, and as
`MAELitModule`'s own training loss) — verified directly against
`src/models/utils.py`:

```
MaskedMAE(x, y) = Σᵢ |xᵢ-yᵢ|·𝟙[|xᵢ-yᵢ| ≥ α]  /  Σᵢ 𝟙[|xᵢ-yᵢ| ≥ α]
```

with `α = 0.005`, and returns `0.0` (with gradient) if every position is
masked out. **A real, minor discrepancy worth being precise about if
quoting this**: the class's own docstring says it's "ignoring positions
where the difference is exactly zero" / measuring "ICL diff only on
non-zero regions" — but the code actually masks on the *residual's
magnitude* (`|x-y| ≥ α`), not on whether the target itself is zero. A
pixel where the prediction is already very close to a nonzero target gets
excluded too, which the docstring's phrasing doesn't quite convey.

**RASE** (Relative Average Spectral Error) and the sliding-window RMSE:
both `torchmetrics.image` classes, `window_size=8`, used as imported —
standard library implementations, not project-authored, so their internal
formulas aren't reproduced here.

## Parameters (active default: `configs/model/mamba_unet.yaml`)

| Parameter | Value | Source |
|---|---|---|
| Optimizer | Adam, `lr=0.0005`, `weight_decay=0.0` | `mamba_unet.yaml:3-7` |
| LR scheduler | `ReduceLROnPlateau(mode='min', factor=0.1, patience=10)`, monitored on `val/loss` | `mamba_unet.yaml:9-14` + `configure_optimizers`, identical pattern in every module |
| Architecture config | `patch_size=1`, `in_chains=1`, `embed_dim=1024`, `depths=[2,1,1,1]`, `drop_rate=0.01`, `drop_path_rate=0.01`, `patch_norm=True`, `final_upsample='expand_first'`, `num_classes=1` | `mamba_unet.yaml:16-30` — a 4-stage Mamba/VMamba encoder-decoder; internal state-space equations not re-derived here |
| MaskedMAE threshold `α` | 0.005 | Hardcoded, `src/models/utils.py:15` |
| RASE / sliding-window-RMSE window | 8 | Hardcoded in every module's `__init__` |
| Input map size | 128×128, 1 channel | `resize_shape: [128,128]` (data config) + `in_chains: 1`/`num_classes: 1` (model config) — consistent across both |
| Batch size | 32 | `combined_dataset.yaml:16` |
| Train/val/test split | 85% / 5% / 10% | `combined_dataset.yaml:17` |
| `compile` | `false` | `mamba_unet.yaml:34` — `torch.compile` available but disabled by default |

## Assumptions worth knowing

- **Sigmoid-squashing both predictions and targets before computing loss/metrics** is assumed consistently across the BCE/MSE/VAE modules, with no comment beyond a one-line, commented-out note ("BCEwithlogits for better num stability"). The rationale for this transform over, say, direct min-max normalisation isn't stated anywhere.
- **Checkpoint selection is by `val/loss`** — whichever loss the active module computes (BCE for the default) — not by `val/RASE` or `val/masked_mae`, even though those are tracked too. See [Training pipeline](02-training-pipeline.md).
- A `net` module is architecturally interchangeable across all `LightningModule` loss variants (Hydra composes `model.net` and `model._target_` independently) — but whether every architecture was actually paired with every loss variant *in practice* isn't established from config alone.

## See also

- [Model architecture catalog](01-model-architecture-catalog.md)
- [Training pipeline & the LightningModule pattern](02-training-pipeline.md)

---
*Sourced from the SPACE-COE Technical Map §17. The equations, the `model_step` table, and the MaskedMAE docstring discrepancy were all re-verified directly against the current code this session (`mamba_module.py`, `src/models/utils.py`).*
