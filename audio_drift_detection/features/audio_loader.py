"""Audio loader — UrbanSound8K via soundata.

Downloads the dataset once to `data_home`; subsequent runs skip the download.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundata

logger = logging.getLogger("audio_drift")


class AudioLoader:
    """Load UrbanSound8K clips through the soundata API."""

    def __init__(self, data_home: str = "data/raw_audio") -> None:
        """Initialise the loader.

        Args:
            data_home: Directory where the dataset is stored / will be downloaded.
        """
        self.data_home = str(Path(data_home).resolve())
        self.dataset = soundata.initialize("urbansound8k", data_home=self.data_home)

    # ------------------------------------------------------------------
    # Download (idempotent — skips if data already present)
    # ------------------------------------------------------------------
    def download(self) -> None:
        """Download UrbanSound8K if not already present."""
        logger.info("Ensuring UrbanSound8K is downloaded to %s …", self.data_home)
        self.dataset.download()
        logger.info("Dataset ready.")

    # ------------------------------------------------------------------
    # Load all clips
    # ------------------------------------------------------------------
    def load_all_clips(
        self,
        max_duration: float = 4.0,
        sample_rate: int = 22050,
    ) -> list[dict[str, Any]]:
        """Load all clips and return a list of metadata dicts.

        Each dict contains:
            - ``clip_id``   : str
            - ``audio``     : np.ndarray  (mono waveform)
            - ``sr``        : int
            - ``label``     : str         (e.g. 'dog_bark')
            - ``fold``      : int         (UrbanSound8K fold number)

        Args:
            max_duration: Maximum clip duration in seconds (pad/truncate).
            sample_rate: Target sample rate.

        Returns:
            List of clip dicts.
        """
        clip_ids = self.dataset.clip_ids
        clips_data: list[dict[str, Any]] = []

        logger.info("Loading %d clips …", len(clip_ids))
        for cid in clip_ids:
            try:
                clip = self.dataset.clip(cid)

                # Audio ---------------------------------------------------
                audio_signal, sr = clip.audio
                if audio_signal is None:
                    continue

                # Convert to mono if stereo
                if audio_signal.ndim > 1:
                    audio_signal = np.mean(audio_signal, axis=0)

                # Pad / truncate to fixed length
                target_len = int(max_duration * sr)
                if len(audio_signal) < target_len:
                    audio_signal = np.pad(
                        audio_signal, (0, target_len - len(audio_signal))
                    )
                else:
                    audio_signal = audio_signal[:target_len]

                # Label ----------------------------------------------------
                label = clip.tags.labels[0] if clip.tags and clip.tags.labels else "unknown"

                # Fold (for cross-val / disjoint splits)
                fold = int(clip.fold) if hasattr(clip, "fold") and clip.fold is not None else 0

                clips_data.append(
                    {
                        "clip_id": cid,
                        "audio": audio_signal,
                        "sr": sr,
                        "label": label,
                        "fold": fold,
                    }
                )
            except Exception as exc:
                logger.warning("Skipping clip %s: %s", cid, exc)

        logger.info("Successfully loaded %d clips.", len(clips_data))
        return clips_data
