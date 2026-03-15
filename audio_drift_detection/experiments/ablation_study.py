"""Ablation study — compare STUDD-only, ensemble-only, and combined."""

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

logger = logging.getLogger("audio_drift")

# Ablation configurations: (alpha, beta, label)
ABLATION_CONFIGS = {
    "studd_only": (1.0, 0.0),
    "ensemble_only": (0.0, 1.0),
    "combined": (0.7, 0.3),
}


def run_ablation(
    features: np.ndarray,
    labels: np.ndarray,
    teacher: TeacherEncoder,
    ensemble: EnsembleManager,
    scenario: str = "abrupt",
    adwin_delta: float = 0.002,
    noise_sigma: float = 0.0,
    drift_point_ratio: float = 0.5,
    output_dir: str = "outputs",
    device: str = "cpu",
    ablation_modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run ablation study.

    Args:
        ablation_modes: Subset of modes to evaluate (default: all three).

    Returns:
        List of per-mode result dicts.
    """
    if ablation_modes is None:
        ablation_modes = list(ABLATION_CONFIGS)

    stream_feat, stream_lbl, drift_point, true_drifts = create_drift_scenario(
        scenario, features, labels, drift_point_ratio=drift_point_ratio,
    )

    results: list[dict[str, Any]] = []

    for mode_name in ablation_modes:
        alpha, beta = ABLATION_CONFIGS[mode_name]
        logger.info("Ablation [%s]: α=%.1f  β=%.1f  scenario=%s", mode_name, alpha, beta, scenario)

        monitor = DriftMonitor(delta=adwin_delta)
        teacher.eval()
        total = len(stream_feat)

        for t in tqdm(range(total), desc=f"  ablation:{mode_name}", leave=False):
            x = torch.tensor(stream_feat[t], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                h_t = teacher(x)
            losses = ensemble.compute_losses(x, h_t)
            mean_loss, loss_var = EnsembleManager.compute_statistics(losses)
            score = compute_drift_score(mean_loss, loss_var, alpha, beta, noise_sigma)
            monitor.update(t, score, mean_loss, loss_var)

        result = analyse_drift(true_drifts, monitor.detected_timestamps, total)
        result["mode"] = mode_name
        result["alpha"] = alpha
        result["beta"] = beta
        result["scenario"] = scenario
        results.append(result)

        out = Path(output_dir) / "ablation" / mode_name
        out.mkdir(parents=True, exist_ok=True)
        plot_drift_scores(monitor, true_drifts, save_path=str(out / "drift_scores.png"))

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY (scenario: %s)", scenario)
    logger.info("=" * 70)
    header = f"{'Mode':<18} {'α':>4} {'β':>4} {'Det':>5} {'Delay':>7} {'FPR':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    logger.info(header)
    logger.info("-" * 70)
    for r in results:
        logger.info(
            f"{r['mode']:<18} {r['alpha']:>4.1f} {r['beta']:>4.1f} "
            f"{r['num_detections']:>5} {r['detection_delay']:>7.1f} "
            f"{r['false_positive_rate']:>7.3f} {r['precision']:>7.3f} "
            f"{r['recall']:>7.3f} {r['f1']:>7.3f}"
        )
    logger.info("=" * 70)

    # Save
    out_path = Path(output_dir) / "ablation" / "ablation_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results
