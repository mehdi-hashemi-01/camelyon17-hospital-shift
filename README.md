<div align="center">

# Camelyon17 Hospital Shift

**Accuracy and calibration of lightweight vision models across hospitals**

[Dataset](#dataset) · [Worked example](#worked-example) · [Baseline](#baseline) · [Reproduction](#reproduction)

</div>

## Overview

When a model encounters tissue prepared at another hospital, does its confidence still reflect how often it is right? This repository studies that question with Camelyon17-WILDS. A small worked example introduces the metrics. A frozen ResNet-18 provides the first image baseline.

## Dataset

Each input is a 96 × 96 RGB tissue patch. The binary label records whether the central 32 × 32 region contains any tumor tissue. The hospital defines the domain. Multiple patches can come from the same slide.

| Official split | Role in this project |
|---|---|
| train | Fit the classifier using patches from three source hospitals |
| id_val | Measure performance on held-out patches from source hospitals |
| val | Measure performance at a fourth hospital during development |
| test | Reserve the fifth hospital for the final study |

[WILDS benchmark paper](https://proceedings.mlr.press/v139/koh21a.html) · [Dataset archive, labeled v1.0, 10.7 GB](https://huggingface.co/datasets/pangwei/wilds-camelyon17/resolve/main/camelyon17_v1.0.tar.gz?download=true)

## Worked example

Eight synthetic patches illustrate confidence under hospital shift. Each classifier predicts class 1 when its probability reaches 0.5. Confidence is the probability assigned to the chosen class.

| Hospital | Correct (both models) | Accuracy | Model A confidence | Model B confidence |
|---|---:|---:|---:|---:|
| Source | 3/4 | 75% | 75% | 95% |
| Unseen | 2/4 | 50% | 75% | 95% |

[Input predictions](baseline/data/toy_predictions.csv) · [Calculated results](results/toy_summary.csv) · [Hand calculation](reports/01_two_hospitals.md)

## Baseline

The first training script fits a two-class linear head on ImageNet-pretrained ResNet-18 features. Images keep their original size. The backbone weights and batch normalization statistics remain fixed. The head trains for one pass over the official training split with Adam, learning rate 0.001, batch size 64 and seed 42.

The evaluation script records accuracy, mean top-class confidence and binary Brier score on both validation splits. Brier score is the mean squared difference between the class-1 probability and the binary label. Lower values indicate smaller probability errors. Measured Camelyon17 validation results are pending.

## Repository layout

```text
baseline/
    data/
        toy_predictions.csv
    settings.py
    01_compare_confidence.py
    02_inspect_dataset.py
    03_train_linear_probe.py
    04_evaluate_validation.py
    requirements.txt
experiments/
    fine_tuning/
    calibration/
    adapters/
models/
results/
    toy_summary.csv
    calibration/
    figures/
reports/
    01_two_hospitals.md
    literature/
    protocol/
    error_analysis/
README.md
.gitignore
```

The numbered scripts form the baseline workflow. The experiment directories are reserved for later comparisons. Store checkpoints in `models/`, numeric outputs and figures in `results/`, and written interpretations in `reports/`.

## Reproduction

Run from the repository root. The worked example uses the Python standard library.

```bash
python baseline/01_compare_confidence.py
```

For the image baseline, extract the labeled archive so that the metadata file is at `baseline/data/camelyon17_v1.0/metadata.csv` alongside the `patches/` directory. Install a matching PyTorch and torchvision build for your device. The first training run downloads the ImageNet weights if they are absent from the local cache.

```bash
python -m pip install -r baseline/requirements.txt
python baseline/02_inspect_dataset.py
python baseline/03_train_linear_probe.py
python baseline/04_evaluate_validation.py
```

Training writes `models/resnet18_linear_head.pt`. Evaluation writes `results/validation_metrics.csv`. Each rerun replaces the corresponding file.

## Evaluation scope

The worked example uses hand-written probabilities. The image baseline will supply the first empirical comparison. Mean confidence can hide differences between probability ranges, so the validation analysis will also need accuracy by confidence range. Results from a single held-out hospital describe that particular transfer.
