from __future__ import annotations

import math
from dataclasses import dataclass

BOARD_CENTER_XY_M = (-0.13, 0.0)

DIRECTION_COUNT_16 = 16
DIRECTION_STEP_DEG = 22.5

RADII_M = (0.07, 0.13, 0.19, 0.25)

INNER_RADIUS_INDEX = 0

UNCERTIFIED_DIRECTION_RADIUS = ((15, 3),)

DISJOINTNESS_FLOOR_M = 0.05

SEALED_TASK_IDS_V4 = (3, 6, 9, 12, 19, 22, 25, 28)

DIRECTION_PHRASES_ODD = (
    "east-northeast", "north-northeast", "north-northwest", "west-northwest",
    "west-southwest", "south-southwest", "south-southeast", "east-southeast",
)
DIRECTION_PHRASES_EVEN = (
    "east", "northeast", "north", "northwest", "west", "southwest", "south", "southeast",
)
RADIUS_PHRASES = ("inner", "middle-inner", "middle-outer", "outer")


@dataclass(frozen=True)
class PoolTask:

    task_id: int
    direction_16: int
    radius_index: int
    is_v4: bool

    @property
    def direction_angle_deg(self) -> float:
        return self.direction_16 * DIRECTION_STEP_DEG

    @property
    def radius_m(self) -> float:
        return RADII_M[self.radius_index]

    @property
    def target_xy_m(self) -> tuple[float, float]:
        angle = math.radians(self.direction_angle_deg)
        cx, cy = BOARD_CENTER_XY_M
        return (cx + self.radius_m * math.cos(angle), cy + self.radius_m * math.sin(angle))

    @property
    def language(self) -> str:
        phrases = DIRECTION_PHRASES_EVEN if self.direction_16 % 2 == 0 else DIRECTION_PHRASES_ODD
        direction_phrase = phrases[self.direction_16 // 2]
        return (
            "pick up the black bowl and place it on the "
            f"{RADIUS_PHRASES[self.radius_index]} {direction_phrase} target"
        )

    @property
    def name(self) -> str:
        return f"morse_spatial_ring_v5_d{self.direction_16:02d}_r{self.radius_index:02d}"


def _is_present(direction_16: int, radius_index: int) -> bool:
    if radius_index == INNER_RADIUS_INDEX and direction_16 % 2 == 1:
        return False
    return (direction_16, radius_index) not in UNCERTIFIED_DIRECTION_RADIUS


def build_pool() -> list[PoolTask]:
    tasks: list[PoolTask] = []
    for direction_8 in range(8):
        for radius_index in range(4):
            tasks.append(
                PoolTask(
                    task_id=direction_8 * 4 + radius_index,
                    direction_16=direction_8 * 2,
                    radius_index=radius_index,
                    is_v4=True,
                )
            )
    next_id = 32
    for direction_16 in range(1, DIRECTION_COUNT_16, 2):
        for radius_index in range(4):
            if not _is_present(direction_16, radius_index):
                continue
            tasks.append(
                PoolTask(
                    task_id=next_id,
                    direction_16=direction_16,
                    radius_index=radius_index,
                    is_v4=False,
                )
            )
            next_id += 1
    return tasks


def sealed_task_ids() -> tuple[int, ...]:
    new = tuple(
        task.task_id
        for task in build_pool()
        if not task.is_v4 and ((task.direction_16 - 1) // 2 + task.radius_index) % 4 == 3
    )
    return tuple(sorted(SEALED_TASK_IDS_V4 + new))


def source_task_ids() -> tuple[int, ...]:
    sealed = set(sealed_task_ids())
    return tuple(task.task_id for task in build_pool() if task.task_id not in sealed)
