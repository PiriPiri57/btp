"""Feature extractor — MFCC + Δ + ΔΔ with privacy enforcement.

Raw audio references are discarded after feature computation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("audio_drift")


class FeatureExtractor:
    """Extract MFCC-based features from raw audio waveforms."""

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        use_deltas: bool = True,
        normalize: bool = True,
    ) -> None:
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.use_deltas = use_deltas
        self.normalize = normalize
        self._scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # Single waveform → feature vector
    # ------------------------------------------------------------------
    def extract_single(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract a fixed-length feature vector from one waveform.

        Args:
            audio: 1-D waveform array.
            sr: Sample rate.

        Returns:
            Feature vector of shape ``(feature_dim,)``.
        """
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        features = [np.mean(mfcc, axis=1)]  # (n_mfcc,)

        if self.use_deltas:
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)
            features.append(np.mean(delta, axis=1))
            features.append(np.mean(delta2, axis=1))

        return np.concatenate(features)  # (n_mfcc * 3,) or (n_mfcc,)

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------
    def extract_batch(
        self,
        clips: list[dict[str, Any]],
        save_path: str | Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Extract features for all clips and **delete raw audio** (privacy).

        Args:
            clips: List of clip dicts from ``AudioLoader.load_all_clips()``.
            save_path: Optional directory to cache features as ``.npz``.

        Returns:
            Tuple of ``(features, labels_encoded, label_names)`` where
            ``features`` has shape ``(N, feature_dim)`` and ``labels_encoded``
            has shape ``(N,)`` with integer-encoded labels.
        """
        feature_list: list[np.ndarray] = []
        label_list: list[str] = []

        for clip in clips:
            feat = self.extract_single(clip["audio"], clip["sr"])
            feature_list.append(feat)
            label_list.append(clip["label"])

            # ---- Privacy: remove raw audio reference ----
            clip["audio"] = None

        features = np.stack(feature_list, axis=0).astype(np.float32)

        # Encode labels as integers
        unique_labels = sorted(set(label_list))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels_encoded = np.array([label_to_idx[l] for l in label_list], dtype=np.int64)

        # Optional normalization (fit once)
        if self.normalize:
            self._scaler = StandardScaler()
            features = self._scaler.fit_transform(features).astype(np.float32)

        logger.info(
            "Extracted features: shape=%s, labels=%d classes",
            features.shape,
            len(unique_labels),
        )

        # Optional caching
        if save_path is not None:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_path / "features.npz",
                features=features,
                labels=labels_encoded,
                label_names=np.array(unique_labels),
            )
            logger.info("Features saved to %s", save_path / "features.npz")

        return features, labels_encoded, unique_labels

    # ------------------------------------------------------------------
    # Load cached features
    # ------------------------------------------------------------------
    @staticmethod
    def load_cached(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Load previously cached features.

        Args:
            cache_path: Path to the ``.npz`` file.

        Returns:
            ``(features, labels_encoded, label_names)``
        """
        data = np.load(cache_path, allow_pickle=True)
        return (
            data["features"],
            data["labels"],
            data["label_names"].tolist(),
        )

    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        multiplier = 3 if self.use_deltas else 1
        return self.n_mfcc * multiplier
