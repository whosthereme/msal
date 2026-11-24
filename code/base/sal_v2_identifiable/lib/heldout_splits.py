from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SEALED_TASK_IDS = (3, 6, 9, 12, 19, 22, 25, 28)

SPLIT_KINDS = ("loto", "direction", "radius", "sealed_pattern", "radius_span", "radius_gap")

PARTITIONING_KINDS = ("loto", "direction", "radius", "sealed_pattern", "radius_span")

DIRECTION_COUNT = 8
RADIUS_COUNT = 4

RADIUS_SPAN_PAIRS = (
    ("extrapolate_outward", "interpolate_ring2"),
    ("extrapolate_inward", "interpolate_ring1"),
)

RADIUS_GAP_PAIRS = (
    ("hold_r1_fit_outer", "hold_r1_bracket_wide"),
    ("hold_r2_fit_inner", "hold_r2_bracket_wide"),
)

RADIUS_GAP_DISTANCE_CONTROLS = (
    ("hold_r1_bracket_wide", "hold_r1_bracket_near"),
    ("hold_r2_bracket_wide", "hold_r2_bracket_near"),
)


V5_FIRST_TASK_ID = 32


def direction_of(task_id: int) -> int:
    if int(task_id) >= V5_FIRST_TASK_ID:
        raise ValueError(
            f"direction_of is v4-only (task_id // {RADIUS_COUNT}); id {task_id} is a v5 pool id "
            f"(>= {V5_FIRST_TASK_ID}) and decodes wrongly. Use pool_v5.PoolTask.direction_16."
        )
    return int(task_id) // RADIUS_COUNT


def radius_of(task_id: int) -> int:
    if int(task_id) >= V5_FIRST_TASK_ID:
        raise ValueError(
            f"radius_of is v4-only (task_id % {RADIUS_COUNT}); id {task_id} is a v5 pool id "
            f"(>= {V5_FIRST_TASK_ID}) and decodes wrongly on 12 of 41 source tasks (and "
            "coincidentally correctly on the others). Use pool_v5.PoolTask.radius_index."
        )
    return int(task_id) % RADIUS_COUNT


def _reject_v5_ids_under_the_v4_sealed_set(
    label: str, held_out: tuple[int, ...], fit_on: tuple[int, ...], sealed: tuple[int, ...]
) -> None:
    if tuple(sealed) != SEALED_TASK_IDS:
        return
    v5 = sorted(t for t in {int(x) for x in (*held_out, *fit_on)} if t >= V5_FIRST_TASK_ID)
    if v5:
        raise ValueError(
            f"{label}: task ids {v5} are v5 pool ids (>= {V5_FIRST_TASK_ID}) but the sealed set is "
            f"v4's {SEALED_TASK_IDS}, which cannot see the v5 sealed ids. Pass "
            f"sealed=lib.pool_v5.sealed_task_ids() explicitly."
        )


@dataclass(frozen=True)
class HeldOutGroup:

    kind: str
    label: str
    held_out: tuple[int, ...]
    fit_on: tuple[int, ...]
    extrapolates: bool | None = None
    sealed_holdout_is_the_point: bool = False
    sealed: tuple[int, ...] = SEALED_TASK_IDS

    def __post_init__(self) -> None:
        overlap = set(self.held_out) & set(self.fit_on)
        if overlap:
            raise ValueError(f"held_out and fit_on overlap on {sorted(overlap)}")
        if not self.held_out:
            raise ValueError(f"{self.label}: nothing held out")
        if not self.fit_on:
            raise ValueError(f"{self.label}: nothing left to fit on")
        _reject_v5_ids_under_the_v4_sealed_set(self.label, self.held_out, self.fit_on, self.sealed)
        if set(self.fit_on) & set(self.sealed):
            raise ValueError(
                f"{self.label}: sealed tasks {sorted(set(self.fit_on) & set(self.sealed))} "
                f"must never be fitted on"
            )
        leaked = set(self.held_out) & set(self.sealed)
        if leaked and not self.sealed_holdout_is_the_point:
            raise ValueError(f"{self.label}: sealed tasks {sorted(leaked)} must not appear")


def build_sealed_real_group(
    source_task_ids: list[int], sealed_task_ids: list[int]
) -> HeldOutGroup:
    sealed = tuple(sorted(int(t) for t in sealed_task_ids))
    if set(sealed) != set(SEALED_TASK_IDS):
        raise ValueError(f"expected the registered sealed split {SEALED_TASK_IDS}, got {sealed}")
    return HeldOutGroup(
        "sealed_real",
        "sealed_real",
        sealed,
        tuple(sorted(int(t) for t in source_task_ids)),
        sealed_holdout_is_the_point=True,
    )


def build_groups(source_task_ids: list[int], kind: str) -> list[HeldOutGroup]:
    if kind not in SPLIT_KINDS:
        raise ValueError(f"unknown split kind {kind!r}; expected one of {SPLIT_KINDS}")
    tasks = sorted(int(t) for t in source_task_ids)
    task_set = set(tasks)
    leaked = task_set & set(SEALED_TASK_IDS)
    if leaked:
        raise ValueError(f"source task list contains sealed tasks {sorted(leaked)}")

    groups: list[HeldOutGroup] = []
    if kind == "loto":
        for task in tasks:
            groups.append(
                HeldOutGroup(
                    kind,
                    f"task_{task}",
                    (task,),
                    tuple(t for t in tasks if t != task),
                )
            )
        return groups

    if kind == "direction":
        for direction in range(DIRECTION_COUNT):
            held = tuple(t for t in tasks if direction_of(t) == direction)
            if not held:
                continue
            groups.append(
                HeldOutGroup(
                    kind,
                    f"direction_{direction}",
                    held,
                    tuple(t for t in tasks if t not in set(held)),
                )
            )
        return groups

    if kind == "radius":
        for radius in range(RADIUS_COUNT):
            held = tuple(t for t in tasks if radius_of(t) == radius)
            if not held:
                continue
            groups.append(
                HeldOutGroup(
                    kind,
                    f"radius_{radius}",
                    held,
                    tuple(t for t in tasks if t not in set(held)),
                )
            )
        return groups

    if kind == "radius_span":
        return _radius_span_groups(tasks)

    if kind == "radius_gap":
        return _radius_gap_groups(tasks)

    by_direction = {
        direction: sorted(t for t in tasks if direction_of(t) == direction)
        for direction in range(DIRECTION_COUNT)
    }
    for offset in range(RADIUS_COUNT):
        held = tuple(
            available[offset % len(available)]
            for available in by_direction.values()
            if available
        )
        held = tuple(sorted(set(held)))
        groups.append(
            HeldOutGroup(
                kind,
                f"diagonal_{offset}",
                held,
                tuple(t for t in tasks if t not in set(held)),
            )
        )
    return groups


_RADIUS_SPAN_ARMS = (
    (3, "extrapolate_outward"),
    (2, "interpolate_ring2"),
    (0, "extrapolate_inward"),
    (1, "interpolate_ring1"),
)


def _radius_span_groups(tasks: list[int]) -> list[HeldOutGroup]:
    groups: list[HeldOutGroup] = []
    for ring, label in _RADIUS_SPAN_ARMS:
        held = tuple(t for t in tasks if radius_of(t) == ring)
        if not held:
            continue
        fit = tuple(t for t in tasks if t not in set(held))
        fit_rings = {radius_of(t) for t in fit}
        if not fit_rings:
            continue
        groups.append(
            HeldOutGroup(
                "radius_span",
                label,
                held,
                fit,
                extrapolates=ring < min(fit_rings) or ring > max(fit_rings),
            )
        )
    return groups


_RADIUS_GAP_ARMS = (
    (1, (2, 3), "hold_r1_fit_outer"),
    (1, (0, 3), "hold_r1_bracket_wide"),
    (1, (0, 2), "hold_r1_bracket_near"),
    (2, (0, 1), "hold_r2_fit_inner"),
    (2, (0, 3), "hold_r2_bracket_wide"),
    (2, (1, 3), "hold_r2_bracket_near"),
)


def _radius_gap_groups(tasks: list[int]) -> list[HeldOutGroup]:
    groups: list[HeldOutGroup] = []
    for ring, fit_rings, label in _RADIUS_GAP_ARMS:
        held = tuple(t for t in tasks if radius_of(t) == ring)
        fit = tuple(t for t in tasks if radius_of(t) in fit_rings)
        if not held or not fit:
            continue
        present = {radius_of(t) for t in fit}
        groups.append(
            HeldOutGroup(
                "radius_gap",
                label,
                held,
                fit,
                extrapolates=ring < min(present) or ring > max(present),
            )
        )
    return groups


def ring_distances(group: HeldOutGroup) -> tuple[int, ...]:
    held_rings = {radius_of(t) for t in group.held_out}
    if len(held_rings) != 1:
        raise ValueError(f"{group.label}: ring distance needs exactly one held ring, got {held_rings}")
    ring = held_rings.pop()
    return tuple(sorted(abs(ring - r) for r in {radius_of(t) for t in group.fit_on}))


def group_row_mask(task_ids: np.ndarray, group: HeldOutGroup) -> np.ndarray:
    held = np.asarray(group.held_out, dtype=np.int64)
    return np.isin(np.asarray(task_ids, dtype=np.int64), held)


def describe(groups: list[HeldOutGroup]) -> dict[str, object]:
    if not groups:
        return {"groups": 0}
    sizes = [len(g.held_out) for g in groups]
    summary: dict[str, object] = {
        "kind": groups[0].kind,
        "groups": len(groups),
        "held_out_tasks_per_group": sizes,
        "mean_held_out_tasks": float(np.mean(sizes)),
        "fit_tasks_per_group": [len(g.fit_on) for g in groups],
        "labels": [g.label for g in groups],
    }
    if any(g.extrapolates is not None for g in groups):
        summary["extrapolates"] = [g.extrapolates for g in groups]
    covered = [len(g.held_out) + len(g.fit_on) for g in groups]
    total = len(set().union(*[set(g.held_out) | set(g.fit_on) for g in groups]))
    if any(c != total for c in covered):
        summary["dropped_tasks_per_group"] = [total - c for c in covered]
        summary["partitions_the_source_tasks"] = False
        summary["ring_distances_per_group"] = [list(ring_distances(g)) for g in groups]
    return summary


def build_task_count_blocks(
    source_task_ids: list[int], fit_count: int, sealed: tuple[int, ...] = SEALED_TASK_IDS
) -> list[HeldOutGroup]:
    tasks = sorted(int(t) for t in source_task_ids)
    total = len(tasks)
    if fit_count < 1:
        raise ValueError(f"fit_count must be >= 1, got {fit_count}")
    if fit_count >= total:
        raise ValueError(
            f"fit_count {fit_count} leaves nothing held out of {total} tasks"
        )

    def windows(stride: int) -> list[tuple[int, ...]]:
        seen: list[tuple[int, ...]] = []
        for start in range(0, total, stride):
            block = tuple(sorted(tasks[(start + j) % total] for j in range(fit_count)))
            if block in seen:
                continue
            seen.append(block)
        return seen

    chosen: list[tuple[int, ...]] | None = None
    for stride in range(1, total + 1):
        blocks = windows(stride)
        held = [sum(1 for b in blocks if t not in b) for t in tasks]
        fitted = [sum(1 for b in blocks if t in b) for t in tasks]
        if min(held) >= 1 and len(set(held)) == 1 and len(set(fitted)) == 1:
            if chosen is None or len(blocks) < len(chosen):
                chosen = blocks
    if chosen is None:
        raise RuntimeError(
            f"no balanced cyclic design exists for fit_count={fit_count} over {total} tasks"
        )

    groups = [
        HeldOutGroup(
            kind=f"fit{fit_count}",
            label=f"fit{fit_count}_block{index}",
            held_out=tuple(t for t in tasks if t not in block),
            fit_on=block,
            sealed=sealed,
        )
        for index, block in enumerate(chosen)
    ]
    held_counts = {t: sum(1 for g in groups if t in g.held_out) for t in tasks}
    fit_counts = {t: sum(1 for g in groups if t in g.fit_on) for t in tasks}
    if len(set(held_counts.values())) != 1 or len(set(fit_counts.values())) != 1:
        raise RuntimeError(
            f"fit_count={fit_count}: unbalanced design, held {sorted(set(held_counts.values()))} "
            f"fit {sorted(set(fit_counts.values()))}"
        )
    return groups
