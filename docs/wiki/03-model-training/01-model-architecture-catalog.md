# Model architecture catalog

All architectures live in `src/models/components/*.py` and share the same
contract: a 128×128, single-channel velocity-dispersion map in, a
predicted 128×128 single-channel ICL map out. Which architecture gets
used is a Hydra config choice (`configs/model/*.yaml`), independent of
which loss function trains it — see
[Training pipeline](02-training-pipeline.md) for why that decoupling
matters.

## Architectures present

Directly listed from `src/models/components/`:

- U-Net (`UNet.py`) and a 3D variant (`UNet3D.py`)
- Attention U-Net (`Attention_UNet.py`)
- TransUNet (`TransUNet.py`) and a ResNet-backbone variant (`TransUNet_resnet.py`)
- Swin-UNet (`SwinUNet.py`)
- Mamba-UNet (`Mamba_UNet.py`) — the active default (`configs/train.yaml`'s `model: mamba_unet`)
- TransEncDec (`TransEncDec.py`)
- VAE (`VAE.py`)
- GaussianInterpolationFlow (`Gaussian_Interp.py`)

## Provenance, quoted from the code's own docstrings/comments (not inferred)

- **Attention U-Net** cites Oktay et al. 2018.
- **Swin-UNet** cites the Swin Transformer paper (`arxiv.org/pdf/2103.14030`).
- **Mamba-UNet** is adapted from `github.com/ziyangwang007/Mamba-UNet`.
- **TransUNet** is adapted from `github.com/Beckschen/TransUNet`.
- **GaussianInterpolationFlow**'s stochastic-interpolant formulation (`I_t = (1-t)x + ty + t(1-t)eps`) is documented in its own module docstring as a from-scratch implementation — no external citation given for this one.

## The active default

`configs/train.yaml` selects `model: mamba_unet` (a 4-stage Mamba/VMamba
encoder-decoder — its internal state-space equations aren't re-derived on
this page; see the model's own config for hyperparameters, covered in
[Loss functions & evaluation metrics](03-loss-functions-and-metrics.md)'s
parameters table).

**Worth knowing if you're about to describe this architecture in a
paper**: which architecture is active says nothing about which *loss
function* trains it — see [Training pipeline](02-training-pipeline.md).
The default Mamba-UNet is currently paired with a BCE-on-sigmoid loss, not
a direct regression loss, despite the task being framed as predicting a
continuous-valued map.

## See also

- [Training pipeline & the LightningModule pattern](02-training-pipeline.md)
- [Loss functions & evaluation metrics](03-loss-functions-and-metrics.md)

---
*Sourced from the SPACE-COE Technical Map §4. Architecture list cross-checked directly against `src/models/components/` this session.*
