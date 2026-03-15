# Privacy-Preserving Concept Drift Detection in Streaming Audio

> **STUDD + Ensemble Disagreement** on **UrbanSound8K**

## Overview

This project detects **concept drift** in streaming environmental-sound data using a hybrid approach that combines:

| Signal | What it captures |
|--------|-----------------|
| **STUDD** (Student–Teacher Drift Detection) | Reconstruction error between a frozen teacher encoder and student models |
| **Ensemble Disagreement** | Variance of reconstruction losses across independently-trained student models |

```
drift_score = α × mean_reconstruction_loss + β × loss_variance
```

The score is monitored with an **ADWIN** online drift detector (from `river`).

## Privacy Design

1. Raw audio is **deleted** immediately after MFCC feature extraction.
2. The system operates **only on feature vectors** — never on raw waveforms.
3. Students are trained on **disjoint subsets** (naturally private ensemble).
4. Only **aggregated** statistics (mean, variance) are used for detection.
5. Optional **Gaussian noise injection** for differential-privacy flavour.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (downloads UrbanSound8K on first run)
python main.py

# 3. (Optional) Custom config
python main.py --config configs/config.yaml
```

## Project Structure

```
audio_drift_detection/
├── configs/config.yaml          # Hyperparameters
├── features/                    # Audio loading & MFCC extraction
├── models/                      # Teacher, Student, Ensemble
├── training/                    # Pre-train teacher, train students
├── streaming/                   # Stream simulator & drift scenarios
├── drift_detection/             # Drift score & ADWIN monitor
├── evaluation/                  # Metrics & analysis
├── experiments/                 # Experiment runner & ablation
├── visualization/               # Plotting utilities
├── scripts/                     # Helper scripts
├── utils/                       # Seed, logger, config helpers
├── main.py                      # ← Entry point
└── requirements.txt
```

## Drift Scenarios

| Scenario | Description |
|----------|-------------|
| **Abrupt** | Sudden class-distribution switch at a single point |
| **Gradual** | Slow class-mixing over a transition window |
| **Noise** | Gaussian noise added to features after drift point |

## Evaluation Metrics

- **Detection Delay** — steps between true drift and first detection
- **False Positive Rate** — fraction of spurious detections
- **Precision / Recall / F1**

## Ablation Modes

| Mode | α | β |
|------|---|---|
| STUDD-only | 1.0 | 0.0 |
| Ensemble-only | 0.0 | 1.0 |
| Combined | 0.7 | 0.3 |

## Outputs

After running `python main.py`, results are saved under `outputs/`:

```
outputs/
├── abrupt/          # drift_scores.png, loss_distributions.png, metrics.json
├── gradual/
├── noise/
├── ablation/
│   ├── studd_only/
│   ├── ensemble_only/
│   ├── combined/
│   └── ablation_results.json
└── pipeline.log
```

## Tech Stack

Python 3.10+ · PyTorch · librosa · scikit-learn · river · soundata · matplotlib · seaborn

## Dataset

**UrbanSound8K** — 8,732 labelled sound excerpts of urban sounds (10 classes).  
Downloaded automatically via `soundata` on first run.
