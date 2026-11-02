# One-shot, Dual Signals: Less Appearance, More Anatomy for 3D Medical Image Segmentation

This repository contains the research code for **One-shot, Dual Signals: Less Appearance, More Anatomy for 3D Medical Image Segmentation**. The framework jointly learns registration and segmentation from labeled atlases and unlabeled target images.

## Overview

<p align="center"><img src="fig/1.png" width="90%"></p>

ODSIS combines a shared 3D feature extractor with task-specific registration and segmentation branches. MedSAM2 provides feature-alignment supervision, while degradation-aware training encourages the model to preserve anatomical predictions under image perturbations. The optional cached gate uses two reference teachers to weight the alignment signal spatially. The current training entry point expects five labeled atlases.

The paper citation will be added when the final public metadata is available.

## Code structure

- `train.py`: five-atlas training and validation.
- `test_and_save_results.py`: evaluation with trained registration and segmentation checkpoints.
- `models.py`, `medsam2_align.py`, and `degrade_aware.py`: the main network and auxiliary training components.
- `data/`, `utils/`, and `sam2/`: data processing and model dependencies.

Datasets and model weights are not included in this repository.

## Dependencies

The code has been smoke-tested with Python 3.11, PyTorch 2.5.1 + CUDA 12.1, and torchvision 0.20.1. Install a PyTorch build compatible with your GPU by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install numpy==1.26.4 wheel setuptools
python -m pip install --no-build-isolation -r requirements.txt
```

The `--no-build-isolation` option is needed when building `GeodisTK` from source. A GPU with more than 8 GB of free memory is recommended for the full 128³ training configuration. The required memory depends on the gate mode and the number of loaded teachers.

## Dataset preparation

Training expects exactly five labeled atlas image/label pairs, unlabeled target images, and validation images with labels. NIfTI files (`.nii` or `.nii.gz`) are supported. A typical layout is:

```text
<DATA_ROOT>/
  atlas/
    vol/
      case_01_vol.nii.gz
      ...
      case_05_vol.nii.gz
    seg/
      case_01_seg.nii.gz
      ...
      case_05_seg.nii.gz
  train/
    vol/
      case_101_vol.nii.gz
      ...
  test/
    vol/
      case_201_vol.nii.gz
      ...
    seg/
      case_201_seg.nii.gz
      ...
```

The five atlas image and label filenames must have matching stems before `_vol` and `_seg`. Validation labels are found in the sibling `seg` directory. Images are resampled to 128 × 128 × 128 within the training loader. The current label conversion in `data/datasets_mixed.py` is configured for the 14-class brain setting; adapt it before using another label scheme.

## Pretrained checkpoints and gate cache

Three teacher checkpoints are required for the default cached-gate training path:

| Checkpoint | Role | Argument |
| --- | --- | --- |
| MedSAM2 checkpoint (tiny configuration) | Main alignment teacher | `--medsam2_ckpt` |
| SAM2.1 Hiera Large checkpoint | Gate reference teacher | `--gate_sam2_ckpt` |
| MedSAM2 pretraining checkpoint (tiny configuration) | Gate reference teacher | `--gate_medsam2_ckpt` |

Use checkpoints that match the configurations in `sam2/configs/`. The gate cache is derived from these checkpoints and the local training/atlas images. With `GATE_BUILD['auto'] = True` in `train.py`, the default training path creates or checks the cache in `--align_gate_dir`. It contains per-case `.npy` maps, `gate_meta.json`, and `gate_coefs.pt`. Rebuild it when the input data or teacher checkpoints change. At least two unlabeled training volumes are needed to fit the cache.

The repository does not supply download links for these checkpoints yet. In particular, the provenance and distribution method for the main MedSAM2 checkpoint must be specified before claiming full reproducibility.

## Training

Replace the placeholders with paths on your machine. Quote wildcard patterns so the Python loader receives them unchanged.

```bash
python train.py \
  --train_path "<DATA_ROOT>/train/vol/*" \
  --test_path "<DATA_ROOT>/test/vol/*" \
  --atlas_dir "<DATA_ROOT>/atlas" \
  --medsam2_ckpt "<WEIGHTS>/MedSAM2_latest.pt" \
  --gate_sam2_ckpt "<WEIGHTS>/sam2.1_hiera_large.pt" \
  --gate_medsam2_ckpt "<WEIGHTS>/MedSAM2_pretrain.pth" \
  --align_gate_dir "<OUTPUT>/gate_cache" \
  --save_path "<OUTPUT>/train" \
  --epochs 200 \
  --gpu 0
```

For a smoke test without gate weighting, pass `--align_gate_dir ""` and omit the two gate checkpoint arguments. This checks the main teacher path but does not reproduce the full gated method. Training writes logs, TensorBoard events, and checkpoints under `--save_path`.

## Evaluation

Evaluation currently takes one atlas image/label pair and separately selected registration and segmentation checkpoints:

```bash
python test_and_save_results.py \
  --seg_ckpt "<OUTPUT>/train/best_seg_<EPOCH>.ckpt" \
  --reg_ckpt "<OUTPUT>/train/best_reg_<EPOCH>.ckpt" \
  --test_path "<DATA_ROOT>/test/vol/*" \
  --src_path "<DATA_ROOT>/atlas/vol/case_01_vol.nii.gz" \
  --sseg_path "<DATA_ROOT>/atlas/seg/case_01_seg.nii.gz" \
  --save_path "<OUTPUT>/evaluation" \
  --gpu 0
```

Select checkpoint files that actually exist after training. The evaluation entry point has not yet been verified end to end on a complete public dataset.

## Validation status

All source files passed Python syntax parsing. The main network passed a 128³ GPU forward pass and a 96³ GPU forward/backward pass. The MedSAM2 tiny teacher checkpoint loaded and produced encoder features on GPU. A five-atlas training run reached and completed its first real-data iteration with the gate cache disabled. Full cached-gate training and end-to-end evaluation remain to be validated.

## Citation

The BibTeX entry will be added when the manuscript is publicly available. Until then, please cite the repository URL if you use this code.
