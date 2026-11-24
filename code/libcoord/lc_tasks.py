import math
from pathlib import Path

CENTER = (-0.08, 0.0)
RADII = (0.08, 0.13, 0.18)
TARGET_HALF = 0.015
INIT_HALF = 0.01
SUCCESS_RADIUS = 0.03


def tasks():
  out = []
  for ri, r in enumerate(RADII):
    dirs = range(0, 16, 2) if ri == 0 else range(16)
    for d in dirs:
      a = math.radians(22.5 * d)
      out.append(dict(task_id=len(out), direction_16=d, radius_index=ri, radius_m=r,
        x=CENTER[0] + r * math.cos(a), y=CENTER[1] + r * math.sin(a)))
  return out


BDDL = """(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language place the black bowl at the target position)
    (:regions
      (bowl_init_region
          (:target main_table)
          (:ranges (
              ({ix0:.4f} {iy0:.4f} {ix1:.4f} {iy1:.4f})
            )
          )
      )
      (target_region
          (:target main_table)
          (:ranges (
              ({tx0:.4f} {ty0:.4f} {tx1:.4f} {ty1:.4f})
            )
          )
      )
    )

  (:fixtures
    main_table - table
  )

  (:objects
    akita_black_bowl_1 - akita_black_bowl
  )

  (:obj_of_interest
    akita_black_bowl_1
  )

  (:init
    (On akita_black_bowl_1 main_table_bowl_init_region)
  )

  (:goal
    (And (On akita_black_bowl_1 main_table_target_region))
  )

)
"""


def write_bddl(task, folder):
  folder = Path(folder)
  folder.mkdir(parents=True, exist_ok=True)
  cx, cy = CENTER
  text = BDDL.format(ix0=cx - INIT_HALF, iy0=cy - INIT_HALF, ix1=cx + INIT_HALF, iy1=cy + INIT_HALF,
    tx0=task["x"] - TARGET_HALF, ty0=task["y"] - TARGET_HALF, tx1=task["x"] + TARGET_HALF, ty1=task["y"] + TARGET_HALF)
  path = folder / ("coord_target_%02d.bddl" % task["task_id"])
  path.write_text(text)
  return path
