from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frozen_data import sha256_file
from .v5_data import TASK_SPLIT_SOURCE

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

LANGUAGE_RUN_V5 = PACKAGE_ROOT.parent / "runs" / "20260813_g33_v5_language_substrate_v1"
LANGUAGE_NPZ_V5 = LANGUAGE_RUN_V5 / "data" / "v5_language_conditioned_state_action_fields.npz"
LANGUAGE_REPORT_V5 = LANGUAGE_RUN_V5 / "gate_report.json"

ALIGNED_KEYS_V5 = (
    "task_ids",
    "direction_indices",
    "radius_indices",
    "episode_indices",
    "step_indices",
    "split_codes",
)

EXPECTED_SOURCE_TASKS = 41


class SourceRowView:

    __slots__ = ("_block", "_index", "_absolute_rows")

    def __init__(self, block: np.ndarray, index: np.ndarray, absolute_rows: int) -> None:
        self._block = block
        self._index = index
        self._absolute_rows = absolute_rows

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._absolute_rows,) + self._block.shape[1:]

    @property
    def dtype(self) -> np.dtype:
        return self._block.dtype

    @property
    def compacted(self) -> np.ndarray:
        return self._block

    def __len__(self) -> int:
        return self._absolute_rows

    def __array__(self, dtype: object = None) -> np.ndarray:
        raise TypeError(
            "refusing to materialize the v5 language features as a dense absolute-row array: it would "
            "need a zero block for the 9,629 sealed rows, and a zero feature vector is legal input that "
            "trains without complaint. Index it with source rows instead, or use `.compacted`"
        )

    def __getitem__(self, rows: object) -> np.ndarray:
        rows = np.asarray(rows)
        if rows.dtype == np.bool_:
            if rows.shape != (self._absolute_rows,):
                raise IndexError(
                    f"boolean mask has shape {rows.shape}, expected one flag per absolute v5 row "
                    f"({self._absolute_rows},)"
                )
            rows = np.flatnonzero(rows)
        rows = rows.astype(np.int64, copy=False)
        if rows.size and (rows.min() < 0 or rows.max() >= self._absolute_rows):
            raise IndexError(
                f"absolute v5 row out of range [0, {self._absolute_rows}): "
                f"[{int(rows.min())}, {int(rows.max())}]"
            )
        mapped = self._index[rows]
        if rows.size and mapped.min() < 0:
            offenders = rows[mapped < 0]
            raise IndexError(
                f"{offenders.size} row(s) are not in the v5 language substrate, first "
                f"{int(offenders[0])}: the substrate is source-only by PROTOCOL_G33 §10e, so these are "
                "sealed rows. The E23.4-class path must never reach one, and a zero-filled feature "
                "would have trained without complaint instead of raising here"
            )
        return self._block[mapped]


@dataclass
class LanguageFeaturesV5:

    state_features: SourceRowView
    frozen_extractor_features: SourceRowView
    available: np.ndarray
    sha256: str
    prompt_swap_relative_error: float
    replay_drift_vs_v5_substrate: float
    source_task_count: int

    @property
    def state_dim(self) -> int:
        return int(self.state_features.shape[1])

    def available_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        return rows[self.available[rows]]


def v5_language_substrate_available() -> bool:
    return LANGUAGE_NPZ_V5.exists() and LANGUAGE_REPORT_V5.exists()


def load_v5_language_features(data: object, verify_hash: bool = True) -> LanguageFeaturesV5:
    if not LANGUAGE_NPZ_V5.exists():
        raise FileNotFoundError(
            f"v5 language substrate missing: {LANGUAGE_NPZ_V5}; run "
            "build_v5_language_substrate.py (PROTOCOL_G33 §11j chain item 1)"
        )
    report = json.loads(LANGUAGE_REPORT_V5.read_text(encoding="ascii"))

    if report.get("status") != "pass":
        raise RuntimeError(
            f"v5 language substrate reports status={report.get('status')!r}; a failed build's arrays "
            "are not interpretable and §11j's chain item 1 is not discharged"
        )
    if int(report.get("source_task_count", -1)) != EXPECTED_SOURCE_TASKS:
        raise RuntimeError(
            f"v5 language substrate covers {report.get('source_task_count')} tasks, expected "
            f"{EXPECTED_SOURCE_TASKS}; a short pool would carry a v5 label on a partial design"
        )
    if int(report.get("sealed_task_rows", -1)) != 0:
        raise RuntimeError(
            f"v5 language substrate reports {report.get('sealed_task_rows')} sealed rows; the "
            "E23.4-class path is source-only and must never see one"
        )

    digest = sha256_file(LANGUAGE_NPZ_V5) if verify_hash else ""
    if verify_hash and digest != report["dataset_sha256"]:
        raise RuntimeError(
            f"v5 language substrate hash {digest} does not match its own gate report "
            f"{report['dataset_sha256']}"
        )

    error = report["prompt_swap_relative_error"]
    if not (error and error > 0.0):
        raise RuntimeError(
            f"v5 language substrate reports prompt_swap_relative_error={error!r}; a value of 0.0 means "
            "the feature never saw the instruction and no arm on it is interpretable"
        )

    task_split = np.asarray(getattr(data, "task_split_codes"), dtype=np.int64)
    reference = np.asarray(getattr(data, "state_features"))
    absolute_rows = int(task_split.shape[0])
    if reference.shape[0] != absolute_rows:
        raise RuntimeError(
            f"the v5 data object has {reference.shape[0]} feature rows but {absolute_rows} "
            "task_split_codes; the absolute row space is what every label column is indexed by"
        )
    source_mask = task_split == TASK_SPLIT_SOURCE
    source_rows_total = int(source_mask.sum())

    payload = np.load(LANGUAGE_NPZ_V5)
    features = payload["state_features"]
    blind = payload["frozen_extractor_features"]
    stored_available = payload["feature_available"].astype(bool)

    if int(report.get("row_count", -1)) != source_rows_total:
        raise RuntimeError(
            f"v5 language substrate reports row_count={report.get('row_count')}, the v5 substrate has "
            f"{source_rows_total} source rows; the build is source-only and the two must agree"
        )
    reference_dim = int(reference.shape[1])
    for name, block in (("state_features", features), ("frozen_extractor_features", blind)):
        if block.shape[0] != source_rows_total:
            raise RuntimeError(
                f"v5 language substrate {name} has {block.shape[0]} rows, the v5 substrate has "
                f"{source_rows_total} source rows (of {absolute_rows} total)"
            )
        if block.shape[1] != reference_dim:
            raise RuntimeError(
                f"v5 language substrate {name} is {block.shape[1]}-d, the v5 substrate is "
                f"{reference_dim}-d; the identical layout is what makes the comparison vary "
                "exactly one factor"
            )
    if stored_available.shape != (source_rows_total,):
        raise RuntimeError(
            f"feature_available has shape {stored_available.shape}, expected one flag per source row "
            f"({source_rows_total},)"
        )
    if not stored_available.any():
        raise RuntimeError("no row in the v5 language substrate was rebuilt")

    for key in ALIGNED_KEYS_V5:
        stored = getattr(data, _v5_attribute(key), None)
        if stored is None:
            raise RuntimeError(
                f"the v5 data object exposes no column for {key!r}; the alignment check cannot be "
                "skipped, because the labels are positional"
            )
        expected = np.asarray(stored, dtype=np.int64)[source_mask]
        if not np.array_equal(payload[key].astype(np.int64), expected):
            raise RuntimeError(
                f"v5 language substrate column {key!r} diverges from the v5 substrate's source half; "
                "the labels are positional and would no longer describe these rows"
            )

    index = np.full(absolute_rows, -1, dtype=np.int64)
    index[source_mask] = np.arange(source_rows_total, dtype=np.int64)

    available = np.zeros(absolute_rows, dtype=bool)
    available[source_mask] = stored_available

    return LanguageFeaturesV5(
        state_features=SourceRowView(features, index, absolute_rows),
        frozen_extractor_features=SourceRowView(blind, index, absolute_rows),
        available=available,
        sha256=digest,
        prompt_swap_relative_error=float(error),
        replay_drift_vs_v5_substrate=float(report["replay_drift_vs_v5_substrate"]),
        source_task_count=int(report["source_task_count"]),
    )


def _v5_attribute(on_disk_key: str) -> str:
    return {"direction_indices": "direction_16"}.get(on_disk_key, on_disk_key)
