from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frozen_data import SourceData, sha256_file

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

LANGUAGE_RUN = PACKAGE_ROOT / "runs" / "20260728_g17_language_conditioned_source_features_v1"
LANGUAGE_NPZ = LANGUAGE_RUN / "data" / "source_language_conditioned_state_action_fields.npz"
LANGUAGE_REPORT = LANGUAGE_RUN / "gate_report.json"

ALIGNED_KEYS = (
    "task_ids",
    "direction_indices",
    "radius_indices",
    "episode_indices",
    "step_indices",
    "split_codes",
)


@dataclass
class LanguageFeatures:

    state_features: np.ndarray
    frozen_extractor_features: np.ndarray
    available: np.ndarray
    sha256: str
    prompt_swap_relative_error: float
    replay_drift_vs_frozen_substrate: float

    @property
    def state_dim(self) -> int:
        return int(self.state_features.shape[1])

    def available_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        return rows[self.available[rows]]


def language_substrate_available() -> bool:
    return LANGUAGE_NPZ.exists() and LANGUAGE_REPORT.exists()


def load_language_features(data: SourceData, verify_hash: bool = True) -> LanguageFeatures:
    import json

    if not LANGUAGE_NPZ.exists():
        raise FileNotFoundError(
            f"language-conditioned substrate missing: {LANGUAGE_NPZ}; run "
            "scripts/build_g17_language_source_features.py"
        )
    report = json.loads(LANGUAGE_REPORT.read_text(encoding="ascii"))
    digest = sha256_file(LANGUAGE_NPZ) if verify_hash else ""
    if verify_hash and digest != report["dataset_sha256"]:
        raise RuntimeError(
            f"language substrate hash {digest} does not match its own gate report "
            f"{report['dataset_sha256']}"
        )
    error = report["prompt_swap_relative_error"]
    if not (error and error > 0.0):
        raise RuntimeError(
            f"language substrate reports prompt_swap_relative_error={error!r}; a value of 0.0 means "
            "the feature never saw the instruction and no E15.7 or E17.3 arm on it is interpretable"
        )

    payload = np.load(LANGUAGE_NPZ)
    features = payload["state_features"]
    blind = payload["frozen_extractor_features"]
    available = payload["feature_available"].astype(bool)
    for name, block in (("state_features", features), ("frozen_extractor_features", blind)):
        if block.shape[0] != data.state_features.shape[0]:
            raise RuntimeError(
                f"language substrate {name} has {block.shape[0]} rows, frozen substrate has "
                f"{data.state_features.shape[0]}"
            )
        if block.shape[1] != data.state_features.shape[1]:
            raise RuntimeError(
                f"language substrate {name} is {block.shape[1]}-d, frozen is "
                f"{data.state_features.shape[1]}-d; section 10d requires the identical layout so the "
                "comparison varies exactly one factor"
            )
    if available.shape != (data.state_features.shape[0],):
        raise RuntimeError(f"feature_available has shape {available.shape}, expected one flag per row")
    if not available.any():
        raise RuntimeError("no row in the language substrate was rebuilt")
    for key in ALIGNED_KEYS:
        frozen_column = getattr(data, key)
        if not np.array_equal(payload[key].astype(np.int64), frozen_column):
            raise RuntimeError(
                f"language substrate column {key!r} diverges from the frozen substrate; the g6 "
                "phase labels are positional and would no longer describe these rows"
            )
    return LanguageFeatures(
        state_features=features,
        frozen_extractor_features=blind,
        available=available,
        sha256=digest,
        prompt_swap_relative_error=float(error),
        replay_drift_vs_frozen_substrate=float(report["replay_drift_vs_frozen_substrate"]),
    )
