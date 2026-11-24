from __future__ import annotations

from functools import lru_cache

from .heldout_splits import HeldOutGroup, describe
from .pool_v5 import (
    DIRECTION_COUNT_16,
    DIRECTION_STEP_DEG,
    PoolTask,
    build_pool,
    sealed_task_ids,
    source_task_ids,
)

RADIUS_COUNT = 4

SPLIT_KINDS_V5 = ("loto", "direction", "direction_arc", "radius", "sealed_pattern")

REFUSED_KINDS_V5 = ("radius_span", "radius_gap")

DIRECTION_ARC_WIDTH = 3

EXPECTED_SOURCE_TASKS = 41
EXPECTED_SEALED_TASKS = 14
EXPECTED_TOTAL_TASKS = 55


@lru_cache(maxsize=1)
def _geometry() -> dict[int, PoolTask]:
    pool = {task.task_id: task for task in build_pool()}
    if len(pool) != EXPECTED_TOTAL_TASKS:
        raise RuntimeError(f"pool_v5 built {len(pool)} tasks, expected {EXPECTED_TOTAL_TASKS}")
    return pool


def direction_16_of(task_id: int) -> int:
    pool = _geometry()
    key = int(task_id)
    if key not in pool:
        raise ValueError(f"task id {key} is not in the v5 pool (0..{EXPECTED_TOTAL_TASKS - 1})")
    return pool[key].direction_16


def radius_index_of(task_id: int) -> int:
    pool = _geometry()
    key = int(task_id)
    if key not in pool:
        raise ValueError(f"task id {key} is not in the v5 pool (0..{EXPECTED_TOTAL_TASKS - 1})")
    return pool[key].radius_index


def direction_step_distance(a: int, b: int) -> int:
    raw = abs(int(a) - int(b)) % DIRECTION_COUNT_16
    return min(raw, DIRECTION_COUNT_16 - raw)


def direction_arc(centre: int, width: int = DIRECTION_ARC_WIDTH) -> tuple[int, ...]:
    if width % 2 != 1:
        raise ValueError(f"arc width must be odd so the arc has a centre; got {width}")
    if not 1 <= width <= DIRECTION_COUNT_16:
        raise ValueError(f"arc width {width} outside 1..{DIRECTION_COUNT_16}")
    half = width // 2
    return tuple(
        sorted((int(centre) + offset) % DIRECTION_COUNT_16 for offset in range(-half, half + 1))
    )


def arc_centre(directions: set[int] | frozenset[int]) -> int:
    held = sorted(int(d) for d in directions)
    if not held:
        raise ValueError("no directions")
    if len(held) == DIRECTION_COUNT_16:
        raise ValueError("the whole ring is held; no arc centre is defined")
    absent = set(range(DIRECTION_COUNT_16)) - set(held)
    start = next(d for d in held if (d - 1) % DIRECTION_COUNT_16 in absent)
    ordered = [(start + offset) % DIRECTION_COUNT_16 for offset in range(len(held))]
    if sorted(ordered) != held:
        raise ValueError(f"directions {held} are not one contiguous arc")
    return ordered[len(ordered) // 2]


def _refuse_v4_only_kind(kind: str, tasks: list[int]) -> None:
    ring_sizes = {
        radius: sum(1 for t in tasks if radius_index_of(t) == radius) for radius in range(RADIUS_COUNT)
    }
    raise ValueError(
        f"{kind} is not available on the v5 pool. Both E15.6 designs require PROTOCOL_G15 §9's "
        "matched pairs -- equal fit-set size within a pair -- and the v5 rings are unbalanced: "
        f"sizes {ring_sizes}. hold_r1_fit_outer would fit "
        f"{ring_sizes[2] + ring_sizes[3]} tasks against hold_r1_bracket_wide's "
        f"{ring_sizes[0] + ring_sizes[3]}, and hold_r2_fit_inner "
        f"{ring_sizes[0] + ring_sizes[1]} against hold_r2_bracket_wide's "
        f"{ring_sizes[0] + ring_sizes[3]}. Matching them needs a subsampling rule with a free choice "
        "in it, so it belongs in a protocol amendment rather than here."
    )


def build_groups_v5(
    source_tasks: list[int] | tuple[int, ...],
    kind: str,
    arc_width: int = DIRECTION_ARC_WIDTH,
) -> list[HeldOutGroup]:
    if kind in REFUSED_KINDS_V5:
        _refuse_v4_only_kind(kind, sorted(int(t) for t in source_tasks))
    if kind not in SPLIT_KINDS_V5:
        raise ValueError(f"unknown v5 split kind {kind!r}; expected one of {SPLIT_KINDS_V5}")

    tasks = sorted(int(t) for t in source_tasks)
    if len(set(tasks)) != len(tasks):
        raise ValueError("source task list contains duplicates")
    pool = _geometry()
    unknown = [t for t in tasks if t not in pool]
    if unknown:
        raise ValueError(f"task ids {unknown} are not in the v5 pool")
    sealed = sealed_task_ids()
    leaked = sorted(set(tasks) & set(sealed))
    if leaked:
        raise ValueError(f"source task list contains v5 sealed tasks {leaked}")
    if not tasks:
        raise ValueError("no source tasks")

    if kind == "loto":
        return [
            HeldOutGroup(kind, f"task_{task}", (task,),
                         tuple(t for t in tasks if t != task), sealed=sealed)
            for task in tasks
        ]

    if kind == "direction":
        groups = []
        for direction in range(DIRECTION_COUNT_16):
            held = tuple(t for t in tasks if direction_16_of(t) == direction)
            if not held:
                continue
            fit = tuple(t for t in tasks if t not in set(held))
            if not fit:
                continue
            groups.append(
                HeldOutGroup(kind, f"direction_16_{direction:02d}", held, fit, sealed=sealed)
            )
        return groups

    if kind == "direction_arc":
        groups = []
        for centre in range(DIRECTION_COUNT_16):
            arc = set(direction_arc(centre, arc_width))
            held = tuple(t for t in tasks if direction_16_of(t) in arc)
            if not held:
                continue
            fit = tuple(t for t in tasks if t not in set(held))
            if not fit:
                continue
            groups.append(
                HeldOutGroup(
                    kind, f"arc{arc_width}_centre_{centre:02d}", held, fit, sealed=sealed
                )
            )
        return groups

    if kind == "radius":
        groups = []
        for radius in range(RADIUS_COUNT):
            held = tuple(t for t in tasks if radius_index_of(t) == radius)
            if not held:
                continue
            fit = tuple(t for t in tasks if t not in set(held))
            if not fit:
                continue
            groups.append(HeldOutGroup(kind, f"radius_{radius}", held, fit, sealed=sealed))
        return groups

    return _sealed_pattern_groups(tasks, sealed)


def _sealed_pattern_groups(tasks: list[int], sealed: tuple[int, ...]) -> list[HeldOutGroup]:
    pool = _geometry()
    sealed_directions = sorted({pool[t].direction_16 for t in sealed})
    by_direction = {
        direction: sorted(t for t in tasks if pool[t].direction_16 == direction)
        for direction in sealed_directions
    }
    by_direction = {d: available for d, available in by_direction.items() if available}
    if not by_direction:
        raise ValueError("no source tasks lie in a direction the sealed split occupies")
    groups = []
    for offset in range(RADIUS_COUNT):
        held = tuple(
            sorted({available[offset % len(available)] for available in by_direction.values()})
        )
        fit = tuple(t for t in tasks if t not in set(held))
        if not fit:
            continue
        groups.append(
            HeldOutGroup("sealed_pattern", f"diagonal_{offset}", held, fit, sealed=sealed)
        )
    return groups


def min_direction_gap_deg(group: HeldOutGroup) -> float:
    held = {direction_16_of(t) for t in group.held_out}
    fit = {direction_16_of(t) for t in group.fit_on}
    if not fit:
        raise ValueError(f"{group.label}: no fit tasks")
    return min(
        direction_step_distance(h, f) for h in held for f in fit
    ) * DIRECTION_STEP_DEG


def centre_direction_gap_deg(group: HeldOutGroup) -> float:
    held = {direction_16_of(t) for t in group.held_out}
    centre = arc_centre(held)
    fit = {direction_16_of(t) for t in group.fit_on}
    return min(direction_step_distance(centre, f) for f in fit) * DIRECTION_STEP_DEG


def describe_v5(groups: list[HeldOutGroup]) -> dict[str, object]:
    if not groups:
        return {"groups": 0}
    summary = dict(describe(groups))
    covered = [set(g.held_out) | set(g.fit_on) for g in groups]
    total = set().union(*covered)
    summary["partitions_the_source_tasks"] = all(c == total for c in covered)
    summary["source_tasks"] = len(total)
    summary["min_direction_gap_deg"] = [min_direction_gap_deg(g) for g in groups]
    summary["held_radius_histogram"] = [
        [sum(1 for t in g.held_out if radius_index_of(t) == r) for r in range(RADIUS_COUNT)]
        for g in groups
    ]
    if groups[0].kind in ("direction", "direction_arc"):
        summary["centre_direction_gap_deg"] = [centre_direction_gap_deg(g) for g in groups]
        summary["held_directions_per_group"] = [
            sorted({direction_16_of(t) for t in g.held_out}) for g in groups
        ]
    return summary


def v5_source_tasks() -> list[int]:
    tasks = sorted(source_task_ids())
    if len(tasks) != EXPECTED_SOURCE_TASKS:
        raise RuntimeError(f"pool_v5 gives {len(tasks)} source tasks, expected {EXPECTED_SOURCE_TASKS}")
    if len(sealed_task_ids()) != EXPECTED_SEALED_TASKS:
        raise RuntimeError(
            f"pool_v5 gives {len(sealed_task_ids())} sealed tasks, expected {EXPECTED_SEALED_TASKS}"
        )
    return tasks
