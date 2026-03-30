# CCR-COD v1 (COCO_V1)

This repository contains a **PyTorch-based Camouflaged Object Detection (COD)** experimental codebase.  
It is organized around a **ResNet50 backbone + simple FPN + dual-branch reasoning (Branch A / Branch B) + gated fusion + refinement decoder** architecture, so that training, evaluation, visualization, and ablation experiments can all be reproduced within a single codebase.

## Project Overview

This project approaches COD by separating responsibilities as follows.

### Branch A: global / objectness branch
- coarse semantic localization
- fine semantic prediction
- objectness estimation
- uncertainty estimation
- boundary prior estimation

### Branch B: contour closure reasoning branch
- ROI gating
- boundary candidate extraction
- fragment tokenization
- graph reasoning
- closure prediction
- boundary refinement

### Fusion and final decoding
- combines Branch A and Branch B features through gating-based fusion
- predicts the final camouflage mask through a refinement decoder

## What You Can Do Directly in This Repository

- generate training splits
- run a model structure sanity check
- run dev training and evaluation
- run full-train experiments
- run the **A-only baseline**
- run **Branch A signal ablations**
  - `objectness_map`
  - `uncertainty_map`
  - `boundary_prior`
- save sample-level predictions and branch visualizations

## Current Release Scope

This README is written **based on the code currently included in the repository**.

- The main experimental flow is organized around the **A+B base model** and **Branch A signal ablations**.
- Although config files may contain paths related to `GT_Edge` and `GT_Instance`, the default training flow in the current release mainly operates on **image + mask** inputs.
- Boundary targets are basically generated internally from the mask.
- Hooks for affinity / topology are included in the code structure, but they are not the primary active experimental axes in the default public setup.

In other words, the most accurate way to understand this repository is as a **v1 experimental codebase**.

## Repository Structure

```text
.
├── configs/                 # experiment setting YAML files
├── data/
│   └── splits/              # outputs from prepare-splits
├── datasets/                # dataset / transforms / target generation
├── engine/                  # trainer / evaluator
├── losses/                  # segmentation / boundary / auxiliary losses
├── metrics/                 # COD metric computation
├── models/
│   ├── backbones/           # ResNet50 backbone
│   ├── necks/               # simple FPN neck
│   ├── branches/            # Branch A / Branch B
│   ├── fusion/              # gated / identity fusion
│   ├── decoders/            # refinement decoder
│   └── cod_model.py         # full assembled model
├── scripts/                 # PowerShell scripts for repeated experiments
├── tools/                   # environment checks / result collection
├── utils/                   # config / visualization / common utilities
├── Results/                 # output directory for training results
├── main.py                  # entry point
└── README.md
```

## Execution Entry Points

`main.py` provides the following four commands.

```bash
python main.py prepare-splits
python main.py sanity-model
python main.py train
python main.py eval
```

## Dataset Setup

The default config files use example paths for a Windows environment.  
Before running anything, you **must modify the dataset / results paths in each YAML file to match your own environment**.

The repository assumes the following datasets by default:

- COD10K
- CAMO
- CHAMELEON
- NC4K

Example config section:

```yaml
paths:
  datasets:
    cod10k:
      train_image_root: ...
      train_mask_root: ...
      test_image_root: ...
      test_mask_root: ...
    camo:
      train_image_root: ...
      test_image_root: ...
      mask_root: ...
    chameleon:
      image_root: ...
      mask_root: ...
    nc4k:
      image_root: ...
      mask_root: ...
  results:
    root: ...
```

Split files are generated under `data/splits/` after running `prepare-splits`.

## Installation

First, create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch separately according to your CUDA / OS environment, then install the remaining dependencies.

```powershell
pip install -r requirements.txt
```

Environment check:

```powershell
python .\tools\check_env.py
```

## Quick Start

### 1) Generate splits

```powershell
python .\main.py prepare-splits --config .\configs\model_v1_resnet50.yaml
```

### 2) Model sanity check

```powershell
python .\main.py sanity-model --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml
```

### 3) Dev training

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml --run-name debug_ab_noamp --skip-prepare
```

### 4) Dev evaluation

If `best_salpha.pth` has been created, you can use that checkpoint first. Otherwise, evaluate `last.pth`.

```powershell
python .\main.py eval --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml --run-name debug_ab_noamp_eval --checkpoint .\Results\checkpoints\debug_ab_noamp\last.pth --skip-prepare
```

## Main Experiments

### Base full-train

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_base.yaml --run-name fulltrain_base_v1 --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_base.yaml --run-name fulltrain_base_v1_eval --checkpoint .\Results\checkpoints\fulltrain_base_v1\last.pth --skip-prepare
```

### A-only baseline

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_a_only.yaml --run-name fulltrain_a_only --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_a_only.yaml --run-name fulltrain_a_only_eval --checkpoint .\Results\checkpoints\fulltrain_a_only\last.pth --skip-prepare
```

### Branch A signal ablation

#### Objectness off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_objectness_map.yaml --run-name ablate_objectness --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_objectness_map.yaml --run-name ablate_objectness_eval --checkpoint .\Results\checkpoints\ablate_objectness\last.pth --skip-prepare
```

#### Uncertainty off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_uncertainty_map.yaml --run-name ablate_uncertainty --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_uncertainty_map.yaml --run-name ablate_uncertainty_eval --checkpoint .\Results\checkpoints\ablate_uncertainty\last.pth --skip-prepare
```

#### Boundary prior off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_boundary_prior.yaml --run-name ablate_boundaryprior --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_boundary_prior.yaml --run-name ablate_boundaryprior_eval --checkpoint .\Results\checkpoints\ablate_boundaryprior\last.pth --skip-prepare
```

### Repeated experiment script

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_fulltrain_ablation_sequence.ps1
```

## Result Output Locations

```text
Results/
├── checkpoints/<run_name>/
├── logs/<run_name>/
├── metrics/<run_name>/
├── predictions/<run_name>/
├── vis/<run_name>/
└── debug/<run_name>/
```

## Visualization Outputs

When `eval` is executed, branch visualizations for each sample are saved.

<details>
<summary>View the main generated visualization files</summary>

```text
Results/vis/<run_name>/<dataset>/<sample_id>/
  00_input.jpg
  01_gt.png
  02_gt_overlay.jpg
  03_final_pred.png
  04_final_overlay.jpg
  05_error_fp_fn.png
  06_a_coarse_heatmap.png
  07_a_fine_heatmap.png
  08_a_objectness_raw_heatmap.png
  09_a_objectness_used_heatmap.png
  10_a_uncertainty_raw_heatmap.png
  11_a_uncertainty_used_heatmap.png
  12_a_boundary_prior_raw_heatmap.png
  13_a_boundary_prior_used_heatmap.png
  14_a_feature_energy_heatmap.png
  15_b_roi_mask.png
  16_b_boundary_candidate_heatmap.png
  17_b_boundary_refined_heatmap.png
  18_b_closure_heatmap.png
  19_b_feature_energy_heatmap.png
  20_fusion_gate_heatmap.png
  21_fused_feature_energy_heatmap.png
  22_affinity_graph.png
  23_branch_summary_board.jpg
```

</details>

This structure allows you to inspect the following together:

- whether Branch A signal heads are actually responding
- whether those signals are truly being used downstream
- where Branch B and fusion are contributing

## Log Interpretation

### When using a dev split
- `train_log.csv` records both training loss and validation metrics.

### When using full-train
- if `val_split: null`, validation metrics are not recorded per epoch, and logs are saved mainly around training loss
- final metric comparison is generally performed in the `eval` stage after training

## Recommended Comparison Order

The cleanest interpretation comes from comparing runs in the following order:

1. `fulltrain_a_only`
2. `fulltrain_base_v1`
3. `ablate_objectness`
4. `ablate_uncertainty`
5. `ablate_boundaryprior`

This lets you separate:

- the overall contribution of Branch B
- the contribution of each Branch A signal

## Result Aggregation

After evaluation is finished, you can collect the metric CSV files into one summary file.

```powershell
python .\tools\collect_ablation_results.py --metrics-root .\Results\metrics --output-csv .\Results\ablation_summary.csv
```

## Notes

- Even if `loss_affinity` and `loss_topology` are 0, that does **not** mean Branch B is entirely inactive. It may simply mean those specific losses are disabled in the current configuration.
- The heatmaps saved during `eval` are intended for visualization purposes. It is safest to interpret them as sample-wise normalized outputs.

---

This repository is a **v1 codebase for COD architecture experiments**, with a particular focus on the **A+B base pipeline and Branch A signal ablation analysis**.
