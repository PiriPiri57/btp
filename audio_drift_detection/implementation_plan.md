# Privacy-Preserving Concept Drift Detection — UrbanSound8K

## Overview

Implement a modular Python project that loads UrbanSound8K via `soundata`, extracts MFCC features, trains a teacher–student ensemble, and detects concept drift using hybrid STUDD + Ensemble Disagreement with ADWIN.

## Key Change from Previous Plan

> [!IMPORTANT]
> Dataset: UrbanSound8K loaded via `soundata` (`pip install soundata`).  
> `audio_loader.py` → uses `soundata.initialize('urbansound8k')` + `dataset.download()`.  
> Clips accessed via `dataset.clip_ids` / `dataset.clip(id)` → `clip.audio` (signal, sr), `clip.tags.labels`.

## Proposed Changes

### Scaffolding
| File | Purpose |
|------|---------|
| `requirements.txt` | All deps incl. `soundata` |
| `configs/config.yaml` | Hyperparams (α, β, num_students, etc.) |
| `README.md` | Quickstart & architecture |
| `utils/seed.py` | Global seed |
| `utils/logger.py` | Logging config |
| `utils/helpers.py` | Config loader, paths |

---

### Data & Features (`features/`)
| File | Purpose |
|------|---------|
| `audio_loader.py` | Load UrbanSound8K via `soundata`, return `(audio, sr, label)` per clip, auto-download |
| `feature_extractor.py` | MFCC + Δ + ΔΔ, normalise, delete raw audio reference (privacy) |

---

### Models (`models/`)
| File | Purpose |
|------|---------|
| `teacher_model.py` | MLP encoder → embedding; freeze after pretrain |
| `student_model.py` | MLP → predicted teacher embedding (MSE loss) |
| `ensemble_manager.py` | N students on disjoint subsets; collective inference + stats |

---

### Training (`training/`)
| File | Purpose |
|------|---------|
| `train_teacher.py` | Pretrain teacher (autoencoder reconstruction) |
| `train_students.py` | Train each student to mimic teacher |

---

### Streaming & Drift Scenarios (`streaming/`)
| File | Purpose |
|------|---------|
| `stream_simulator.py` | Yield `(timestamp, feature_vector, label)` |
| `drift_scenarios.py` | Abrupt / gradual / noise drift injection |

---

### Drift Detection (`drift_detection/`)
| File | Purpose |
|------|---------|
| `drift_score.py` | `α·mean_loss + β·loss_variance` |
| `drift_monitor.py` | ADWIN wrapper, report drift events |

---

### Evaluation & Experiments
| File | Purpose |
|------|---------|
| `evaluation/metrics.py` | Delay, FPR, precision, recall, F1 |
| `evaluation/drift_analysis.py` | Compare predicted vs actual drift |
| `experiments/run_experiments.py` | Run all 3 scenarios |
| `experiments/ablation_study.py` | STUDD-only / ensemble-only / combined |

---

### Visualization (`visualization/`)
| File | Purpose |
|------|---------|
| `plot_drift_scores.py` | Drift score vs time |
| `plot_loss_distributions.py` | Loss distributions & variance |

---

### Entry Points
| File | Purpose |
|------|---------|
| `main.py` | End-to-end pipeline |
| `scripts/run_pipeline.py` | Scripted pipeline |
| `scripts/run_experiment_suite.py` | All experiments |

## Verification Plan

```bash
cd audio_drift_detection
pip install -r requirements.txt
python main.py
```

Expected: downloads UrbanSound8K on first run, trains models, streams data, detects drift, prints events, saves plots.
