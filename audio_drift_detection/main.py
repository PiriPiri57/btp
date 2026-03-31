"""Main entry point — run the full privacy-preserving drift detection pipeline.

Usage:
    python main.py
    python main.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from utils.helpers import load_config, ensure_dir, get_device
from utils.logger import setup_logger
from utils.seed import set_global_seed
from features.audio_loader import AudioLoader
from features.feature_extractor import FeatureExtractor
from models.ensemble_manager import EnsembleManager
from training.train_teacher import train_teacher
from training.train_students import train_students
from experiments.run_experiments import run_all_experiments
from experiments.ablation_study import run_ablation


def main(config_path: str = r"C:\Users\Priyanshu.Priyanshu_PC\Desktop\BTP\audio_drift_detection\configs\config.yaml") -> None:
    """Run the complete pipeline end-to-end."""

    # ── Configuration ────────────────────────────────────────────────
    cfg = load_config(config_path)
    seed = cfg["experiment"]["seed"]
    set_global_seed(seed)
    device = get_device()

    output_dir = cfg["experiment"]["output_dir"]
    ensure_dir(output_dir)
    logger = setup_logger(log_file=str(Path(output_dir) / "pipeline.log"))
    logger.info("Device: %s", device)
    logger.info("Configuration loaded from %s", config_path)

    # ── 1. Download / Load UrbanSound8K ───────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 — Dataset download & loading")
    logger.info("=" * 60)

    loader = AudioLoader(data_home=cfg["dataset"]["data_home"])
    dataset_path = Path(cfg["dataset"]["data_home"]) / "UrbanSound8K"

    if not dataset_path.exists():
        logger.info("Dataset not found. Downloading...")
        loader.download()
    else:
        logger.info("Dataset folder exists. Checking index...")

    try:
        # This will fail if index is missing
        _ = loader.dataset.clip_ids
        logger.info("Index found. Skipping download.")
    except Exception:
        logger.info("Index missing. Downloading metadata only...")
        loader.download()

    clips = loader.load_all_clips(
        max_duration=cfg["dataset"]["max_duration"],
        sample_rate=cfg["dataset"]["sample_rate"],
    )

    # ── 2. Feature Extraction (privacy: raw audio deleted) ───────────
    logger.info("=" * 60)
    logger.info("STEP 2 — Feature extraction (MFCC + Δ + ΔΔ)")
    logger.info("=" * 60)

    fe = FeatureExtractor(
        n_mfcc=cfg["features"]["mfcc_features"],
        n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"],
        use_deltas=cfg["features"]["use_deltas"],
        normalize=cfg["features"]["normalize"],
    )

    processed_dir = cfg["dataset"]["processed_dir"]
    cache_file = Path(processed_dir) / "features.npz"

    if cache_file.exists():
        logger.info("Loading cached features from %s", cache_file)
        features, labels, label_names = FeatureExtractor.load_cached(cache_file)
    else:
        features, labels, label_names = fe.extract_batch(clips, save_path=processed_dir)

    input_dim = features.shape[1]
    logger.info("Feature dim: %d  |  Samples: %d  |  Classes: %d", input_dim, len(features), len(label_names))

    # ── 3. Train Teacher ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 — Pretrain teacher encoder")
    logger.info("=" * 60)

    teacher = train_teacher(
        features=features,
        input_dim=input_dim,
        hidden_dims=cfg["teacher"]["hidden_dims"],
        embedding_dim=cfg["teacher"]["embedding_dim"],
        epochs=cfg["teacher"]["epochs"],
        batch_size=cfg["teacher"]["batch_size"],
        learning_rate=cfg["teacher"]["learning_rate"],
        device=device,
    )

    # ── 4. Train Student Ensemble ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 — Train student ensemble (%d students)", cfg["students"]["num_students"])
    logger.info("=" * 60)

    ensemble = EnsembleManager(
        num_students=cfg["students"]["num_students"],
        input_dim=input_dim,
        hidden_dims=cfg["students"]["hidden_dims"],
        embedding_dim=cfg["students"]["embedding_dim"],
        device=device,
    )

    train_students(
        ensemble=ensemble,
        teacher=teacher,
        features=features,
        labels=labels,
        epochs=cfg["students"]["epochs"],
        batch_size=cfg["students"]["batch_size"],
        learning_rate=cfg["students"]["learning_rate"],
        device=device,
    )

    # ── 5. Run Drift Detection Experiments ────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 — Drift detection experiments")
    logger.info("=" * 60)

    run_all_experiments(
        features=features,
        labels=labels,
        teacher=teacher,
        ensemble=ensemble,
        scenarios=cfg["experiment"]["scenarios"],
        alpha=cfg["drift"]["alpha"],
        beta=cfg["drift"]["beta"],
        adwin_delta=cfg["drift"]["adwin_delta"],
        noise_sigma=cfg["drift"]["noise_sigma"],
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"],
        output_dir=output_dir,
        device=device,
    )

    # ── 6. Ablation Study ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 — Ablation study")
    logger.info("=" * 60)

    run_ablation(
        features=features,
        labels=labels,
        teacher=teacher,
        ensemble=ensemble,
        scenario="abrupt",
        adwin_delta=cfg["drift"]["adwin_delta"],
        noise_sigma=cfg["drift"]["noise_sigma"],
        drift_point_ratio=cfg["streaming"]["drift_point_ratio"],
        output_dir=output_dir,
        device=device,
        ablation_modes=cfg["experiment"]["ablation_modes"],
    )

    logger.info("✅  Pipeline complete. Results saved to %s/", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving Concept Drift Detection"
    )

    default_config = str(Path(__file__).parent / "configs/config.yaml")

    parser.add_argument("--config", default=default_config)

    args = parser.parse_args()
    main(args.config)