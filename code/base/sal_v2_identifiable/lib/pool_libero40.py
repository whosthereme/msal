from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from lib import rlds_reader

FIRST_TASK_ID = 1000

POOL_SIZE = 40

FIT_COUNTS = (8, 16, 24, 32, 36)

FIXED_DENSITY_PER_TASK = 512 // 36

EXPECTATION_PAIRED_TASKS = 27
FALSIFIER_PAIRED_TASKS = 23
EXPECTATION_GAP_POINTS = 8.0
FALSIFIER_GAP_POINTS = 3.0

LOW, HIGH = 8, 36


@dataclass(frozen=True)
class PoolTask:

    task_id: int
    suite: str
    instruction: str

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError(f"task {self.task_id}: empty instruction")


def build_pool() -> tuple[PoolTask, ...]:
    per_suite: dict[str, set[str]] = {}
    for suite in rlds_reader.SUITES:
        found: set[str] = set()
        for episode in rlds_reader.iter_episodes(suite):
            found.add(episode.instruction)
        per_suite[suite] = found

    tasks: list[PoolTask] = []
    for suite in sorted(per_suite):
        for instruction in sorted(per_suite[suite]):
            tasks.append(PoolTask(FIRST_TASK_ID + len(tasks), suite, instruction))

    if len(tasks) != POOL_SIZE:
        raise RuntimeError(
            f"expected {POOL_SIZE} distinct instructions across {len(rlds_reader.SUITES)} suites, "
            f"got {len(tasks)}: "
            + ", ".join(f"{s}={len(per_suite[s])}" for s in sorted(per_suite))
        )
    return tuple(tasks)


def task_ids() -> list[int]:
    return [t.task_id for t in build_pool()]


def assert_pool_ids(ids: Iterator[int] | list[int] | tuple[int, ...], sealed: tuple[int, ...]) -> None:
    seen = sorted({int(i) for i in ids})
    if not seen:
        raise ValueError("assert_pool_ids received no ids; an empty design cannot be validated")
    lo, hi = FIRST_TASK_ID, FIRST_TASK_ID + POOL_SIZE - 1
    stray = [i for i in seen if not lo <= i <= hi]
    if stray:
        raise ValueError(
            f"ids {stray} are outside the G39 pool range {lo}..{hi}; two pools have been mixed into "
            f"one design and no balance assertion downstream would catch it"
        )
    if sealed:
        raise ValueError(
            f"G39's pool carves no sealed split, but sealed={tuple(sealed)} is in force over ids "
            f"{seen[0]}..{seen[-1]}. Either the sealed set belongs to another pool (v4's or v5's, whose "
            f"ids this pool's range cannot contain), or a sealed split was carved without amending "
            f"PROTOCOL_G39 section 2. Both are errors."
        )


def assert_census(tasks: tuple[PoolTask, ...]) -> None:
    if len(tasks) != POOL_SIZE:
        raise RuntimeError(f"census short: {len(tasks)} of {POOL_SIZE} tasks")
    per_suite = {s: sum(1 for t in tasks if t.suite == s) for s in rlds_reader.SUITES}
    expected = POOL_SIZE // len(rlds_reader.SUITES)
    bad = {s: n for s, n in per_suite.items() if n != expected}
    if bad:
        raise RuntimeError(f"census unbalanced: expected {expected} per suite, got {bad}")
    if len({t.instruction for t in tasks}) != POOL_SIZE:
        raise RuntimeError("census contains duplicate instructions; task ids would not be distinct tasks")
