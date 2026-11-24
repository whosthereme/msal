from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

PARENT_ROOT = Path(__file__).resolve().parents[2]
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from . import rlds_reader as rlds

AGENTVIEW_KEY = "agentview_image"
WRIST_KEY = "robot0_eye_in_hand_image"


def _invert(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[::-1, ::-1])


def rlds_policy_observation(episode: rlds.Episode, step: int) -> dict[str, Any]:
    import torch

    from smolvla_state_contract import raw_policy_observation

    if not 0 <= step < len(episode):
        raise IndexError(f"step {step} out of range for a {len(episode)}-step episode")
    agentview = _invert(rlds.decode_jpeg(episode.images[step]))
    wrist = _invert(rlds.decode_jpeg(episode.wrist_images[step]))
    state = np.asarray(episode.states[step], dtype=np.float32)
    if state.shape != (rlds.STATE_DIM,):
        raise ValueError(f"expected {rlds.STATE_DIM}-d RLDS state, got {state.shape}")

    observation = raw_policy_observation(
        {
            AGENTVIEW_KEY: agentview,
            WRIST_KEY: wrist,
            "robot0_eef_pos": state[:3],
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": state[6:8],
        },
        episode.instruction,
    )
    observation["observation.state"] = torch.from_numpy(state)
    return observation


def assert_flip_cancels(episode: rlds.Episode, step: int = 0) -> dict[str, float]:
    import torch

    from smolvla_state_contract import raw_policy_observation

    reference = rlds.decode_jpeg(episode.images[step]).astype(np.float32) / 255.0
    reference_tensor = torch.from_numpy(reference).permute(2, 0, 1)

    corrected = rlds_policy_observation(episode, step)["observation.images.image"]
    naive = raw_policy_observation(
        {
            AGENTVIEW_KEY: rlds.decode_jpeg(episode.images[step]),
            WRIST_KEY: rlds.decode_jpeg(episode.wrist_images[step]),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        },
        episode.instruction,
    )["observation.images.image"]

    return {
        "upright_residual": float((corrected - reference_tensor).abs().max()),
        "naive_residual": float((naive - reference_tensor).abs().max()),
    }
