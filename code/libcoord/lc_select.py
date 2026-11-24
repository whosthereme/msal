import json
import re
import sys

import numpy as np

names = [x.strip() for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
vessel = re.compile(r"bowl|ramekin|mug", re.I)
exclude_place = re.compile(r"bowling|dog_bowl|plant|planter|xl_bowl|7_qt|35_ct|frother|coffeemaker|kcup|cupcake", re.I)
place = [n for n in names if vessel.search(n) and not exclude_place.search(n)]
rest = [n for n in names if n not in place and not vessel.search(n)]
rng = np.random.default_rng(20261116)
clutter = sorted(rng.choice(rest, size=120, replace=False).tolist())
json.dump(dict(place=place, clutter=clutter, generator_seed=20261116, source="kevinzakka/mujoco_scanned_objects"),
  open(sys.argv[2], "w"), indent=1)
print("place", len(place), place)
print("clutter", len(clutter), clutter[:20], "...")
