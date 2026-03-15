"""Run the full experiment suite across all drift scenarios."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from drift_detection.drift_monitor import DriftMonitor
from drift_detection.drift_score import compute_drift_score
from evaluation.drift_analysis import analyse_drift
from models.ensemble_manager import EnsembleManager
from models.teacher_model import TeacherEncoder
from streaming.drift_scenarios import create_drift_scenario
from visualization.plot_drift_scores import plot_drift_scores
from visualization.plot_loss_distributions import plot_loss_distributions

logger = logging.getLogger("audio_drift")


def run_single_experiment(
    scenario_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    teacher: TeacherEncoder,
    ensemble: EnsembleManager,
    alpha: float = 0.7,
    beta: float = 0.3,
    adwin_delta: float = 0.002,
    noise_sigma: float = 0.0,
    drift_point_ratio: float = 0.5,
    output_dir: str = "outputs",
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one drift-detection experiment for a given scenario.

    Returns:
        Summary dict with all metrics.
    """
    logger.info("━" * 60)
    logger.info("Running scenario: %s", scenario_name)
    logger.info("━" * 60)

    # Create drift scenario
    stream_feat, stream_lbl, drift_point, true_drifts = create_drift_scenario(
        scenario_name, features, labels, drift_point_ratio=drift_point_ratio,
    )

    # Drift monitor
    monitor = DriftMonitor(delta=adwin_delta)
    teacher.eval()

    total = len(stream_feat)
    for t in tqdm(range(total), desc=f"  {scenario_name}", leave=False):
        x = torch.tensor(stream_feat[t], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            h_t = teacher(x)

        losses = ensemble.compute_losses(x, h_t)
        mean_loss, loss_var = EnsembleManager.compute_statistics(losses)
        score = compute_drift_score(mean_loss, loss_var, alpha, beta, noise_sigma)
        monitor.update(t, score, mean_loss, loss_var)

    # Analyse
    result = analyse_drift(true_drifts, monitor.detected_timestamps, total)
    result["scenario"] = scenario_name

    # Save plots
    out = Path(output_dir) / scenario_name
    out.mkdir(parents=True, exist_ok=True)
    plot_drift_scores(monitor, true_drifts, save_path=str(out / "drift_scores.png"))
    plot_loss_distributions(monitor, true_drifts, save_path=str(out / "loss_distributions.png"))

    # Save metrics JSON
    with open(out / "metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def run_all_experiments(
    features: np.ndarray,
    labels: np.ndarray,
    teacher: TeacherEncoder,
    ensemble: EnsembleManager,
    scenarios: list[str] | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run experiments for all scenarios.

    Returns:
        List of per-scenario result dicts.
    """
    if scenarios is None:
        scenarios = ["abrupt", "gradual", "noise"]

    results: list[dict[str, Any]] = []
    for name in scenarios:
        r = run_single_experiment(name, features, labels, teacher, ensemble, **kwargs)
        results.append(r)

    # Summary table
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 70)
    header = f"{'Scenario':<12} {'Detections':>10} {'Delay':>8} {'FPR':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}"
    logger.info(header)
    logger.info("-" * 70)
    for r in results:
        logger.info(
            f"{r['scenario']:<12} {r['num_detections']:>10} "
            f"{r['detection_delay']:>8.1f} {r['false_positive_rate']:>8.3f} "
            f"{r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1']:>8.3f}"
        )
    logger.info("=" * 70)
    return results
