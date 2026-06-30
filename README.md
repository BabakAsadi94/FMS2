# FMS²: Unified Flow Matching for Segmentation and Synthesis of Thin Structures

Official implementation of the ECCV paper:

> **FMS²: Unified Flow Matching for Segmentation and Synthesis of Thin Structures**  
> *B. Asadi, P. Wu, M. Golparvar-Fard, V. Shah, R. Hajj*  
> European Conference on Computer Vision (ECCV), 2026

---

## Overview

FMS addresses the data scarcity problem in crack segmentation by pairing two symmetric flow-matching components — one for synthesis, one for segmentation — into an end-to-end data-augmentation pipeline:

| Component | Direction | Description |
|-----------|-----------|-------------|
| **SynFlow** | mask → image | Mask-guided crack image synthesis with classifier-free guidance (CFG). Includes a **Mask Generator** sub-component (class → mask) for producing diverse training masks. |
| **SegFlow** | image → mask | Direct flow-matching segmentation trained on real + synthetic image–mask pairs. |

**Full pipeline:**
1. Train the **Mask Generator** (part of SynFlow) to synthesize diverse crack masks conditioned on terrain class.
2. Run **SynFlow** to generate photorealistic crack images from those synthetic masks.
3. Train **SegFlow** on the combined real + synthetic pairs to improve segmentation.

All components use [Conditional Flow Matching (CFM)](https://arxiv.org/abs/2302.00482) as the training objective and support multiple ODE solvers (Euler, dopri5, RK4, midpoint, Heun) at inference.

---

## Repository Structure

```
FMS/
├── torchcfm/                        # CFM library (adapted from conditional-flow-matching)
│   ├── conditional_flow_matching.py
│   ├── optimal_transport.py
│   ├── models/
│   │   ├── models.py
│   │   └── unet/
│   │       ├── unet.py              # Base UNet (Mask Generator)
│   │       ├── unet_residal.py      # Lightweight UNet for SegFlow
│   │       ├── unet_sdm_CrackSDM_v3_encoder.py  # SPADE UNet for SynFlow
│   │       ├── nn.py
│   │       └── fp16_util.py
│
├── FMS/
│   ├── synflow/                     # SynFlow pipeline (mask → image)
│   │   ├── train.py                 #   image synthesis training
│   │   ├── sample.py                #   image synthesis inference
│   │   ├── train_mask.py            #   Mask Generator training  (class → mask)
│   │   ├── sample_mask.py           #   Mask Generator inference
│   │   ├── image_datasets.py        #   image + annotation loader
│   │   ├── mask_dataset.py          #   mask + class-label loader
│   │   └── utils.py
│   └── segflow/                     # SegFlow pipeline (image → mask)
│       ├── train.py
│       └── image_datasets.py
│
├── scripts/
│   ├── synflow_train_mse_cfg_unet_encoder.sh   # SynFlow — image synthesis training
│   ├── synflow_sample.sh                       # SynFlow — image synthesis inference
│   ├── synflow_train_mask_mse_cfg.sh           # Mask Generator training
│   ├── synflow_sample_mask.sh                  # Mask Generator inference
│   └── segflow_train.sh                        # SegFlow training
│
├── data/
│   └── CRACK500O/
│       ├── training/{images,annotations}/
│       ├── validation/{images,annotations}/
│       ├── testing/{images,annotations}/
│       └── CRACK500O.csv            # class labels for Mask Generator
│
├── requirements.txt
└── setup.py
```

---

## Environment Setup

### Prerequisites

- Python 3.9 or 3.10
- CUDA-capable GPU (tested on NVIDIA A100 / V100)
- conda (recommended)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/BabakAsadi94/FMS2.git
cd FMS2

# 2. Create and activate a conda environment
conda create -n fms python=3.10
conda activate fms

# 3. Install PyTorch (adjust the CUDA version to match your system)
#    See https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install the torchcfm package in editable mode
pip install -e .
```

Verify the installation:

```bash
python -c "import torchcfm; print(torchcfm.__version__)"
```

---

## Data Preparation

All scripts are run from the **repository root**. Data should be placed at `data/CRACK500O/` with the following layout:

```
data/
└── CRACK500O/
    ├── training/
    │   ├── images/        # RGB crack images  (.png / .jpg)
    │   └── annotations/   # Binary masks, same filenames as images
    ├── validation/
    │   ├── images/
    │   └── annotations/
    ├── testing/
    │   ├── images/
    │   └── annotations/
    └── CRACK500O.csv      # Class labels for the Mask Generator
                           # Columns: image_name, class
```

`image_name` in the CSV should be a path relative to the directory pointed to by `--data_dir` (e.g. `CRACK500O/training/annotations/foo.png`).  
Use `data/annotation_csv_generator.ipynb` to generate this CSV automatically.

---

## Training

All scripts use `PYTHONPATH=$(pwd)` so absolute imports resolve correctly. Run every command from the repository root.

---

### 1. SynFlow — mask → image synthesis

SynFlow synthesizes photorealistic crack images conditioned on binary masks using a SPADE-conditioned UNet with classifier-free guidance (CFG). It relies on a **Mask Generator** sub-component to first produce diverse synthetic masks.

#### 1.1 Mask Generator — class → mask (prerequisite)

Trains a class-conditional CFM model to synthesize binary crack masks from integer terrain-class labels. Run this before training SynFlow if you need synthetic masks.

```bash
bash scripts/synflow_train_mask_mse_cfg.sh
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `icfm` | CFM variant (`icfm` / `otcfm` / `fm` / `si`) |
| `--data_dir` | `data` | Root of the `data/` directory |
| `--dataset_csv` | `data/CRACK500O/CRACK500O.csv` | Class-label CSV (`image_name`, `class`) |
| `--batch_size` | `16` | Batch size |
| `--total_steps` | `80001` | Training steps |
| `--omega` | `0.4` | CFG guidance strength |
| `--output_dir` | `outputs/mask_generator` | Checkpoint directory |

#### 1.2 Image Synthesis — mask → image

Trains the SPADE UNet to synthesize crack images from binary masks.

```bash
bash scripts/synflow_train_mse_cfg_unet_encoder.sh
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `icfm` | CFM variant |
| `--data_dir` | `data/CRACK500O` | Dataset root (expects `{training,validation,testing}/{images,annotations}/`) |
| `--data_type` | `fms` | Dataset mode |
| `--img_size` | `256` | Spatial resolution |
| `--batch_size` | `4` | Batch size |
| `--total_steps` | `150000` | Training steps |
| `--omega` | `0.4` | CFG guidance strength |
| `--drop_rate` | `0.1` | Mask dropout probability for CFG training |
| `--output_dir` | `outputs/synflow_crack500_256` | Checkpoint directory |

---

### 2. SegFlow — image → mask segmentation

SegFlow directly maps crack images to segmentation masks using a lightweight flow-matching model. Train it on the combination of real and SynFlow-synthesized image–mask pairs.

```bash
bash scripts/segflow_train.sh
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `icfm` | CFM variant |
| `--image_dirs_training` | *(required)* | Comma-separated training image dirs |
| `--class_dirs_training` | *(required)* | Comma-separated training annotation dirs |
| `--image_dirs_validation` | *(required)* | Validation image dirs |
| `--class_dirs_validation` | *(required)* | Validation annotation dirs |
| `--image_dirs_testing` | *(required)* | Testing image dirs |
| `--class_dirs_testing` | *(required)* | Testing annotation dirs |
| `--num_inference_steps` | `10` | ODE steps during evaluation |
| `--integration_method` | `euler` | ODE solver (`euler` / `dopri5`) |
| `--save_step` | `2000` | Checkpoint and eval frequency |
| `--output_dir` | `outputs/segflow_crack500` | Checkpoint directory |

---

## Inference

---

### SynFlow — synthesize crack images from masks

#### Step 1: Generate masks (Mask Generator)

```bash
bash scripts/synflow_sample_mask.sh
```

Reads class labels from `--dataset_csv` and writes one binary mask PNG per entry to `--save_dir`.

Key flags: `--ckpt_step`, `--omega`, `--num_steps`, `--integration_method` (`euler` / `dopri5` / `rk4` / `midpoint`), `--surfix`.

#### Step 2: Generate images from masks

```bash
bash scripts/synflow_sample.sh
```

Reads masks from `--data_dir` and writes one synthesized PNG per mask to `--save_dir`, named by the annotation filename stem.

Key flags: `--ckpt_step`, `--omega`, `--num_steps`, `--integration_method` (`euler` / `dopri5` / `rk4` / `midpoint` / `heun`).

---

## Acknowledgements

This codebase is built upon and borrows heavily from the
[**conditional-flow-matching**](https://github.com/atong01/conditional-flow-matching)
repository by Alexander Tong and Kilian Fatras (MIT License).  
We gratefully acknowledge their work on the TorchCFM library, which provided the
CFM training objectives, UNet backbone, and OT-plan sampler used throughout this project.

The SPADE-conditioned UNet (`unet_sdm_CrackSDM_v3_encoder.py`) and the
classifier-free guidance training scheme for SynFlow are original contributions of this paper.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{asadi2026fms,
  title={FMS $\^{} 2$: Unified Flow Matching for Segmentation and Synthesis of Thin Structures},
  author={Asadi, Babak and Wu, Peiyang and Golparvar-Fard, Mani and Shah, Viraj and Hajj, Ramez},
  journal={arXiv preprint arXiv:2603.13659},
  year={2026}
}
```

Please also cite the TorchCFM library:

```bibtex
@article{tong2024improving,
  title   = {Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport},
  author  = {Alexander Tong and Kilian Fatras and Nikolay Malkin and Guillaume Huguet and
             Yanlei Zhang and Jarrid Rector-Brooks and Guy Wolf and Yoshua Bengio},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  url     = {https://openreview.net/forum?id=CD9Snc73AW}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
