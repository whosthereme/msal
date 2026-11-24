import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import tc_lodo_adapt as A
import numpy as np
from msal_ext import run_nl
from msal_ext.nl import FiLMNonlinearReadout

EXTRA = 15
RANDOM_SEED = 20261116
RANDOM_SD = 0.7
XY6 = A.arms.DESCRIPTORS["lodo_xy6"]


def xy6r2(ids):
  extra = np.stack([np.random.default_rng([RANDOM_SEED, int(t)]).normal(0.0, RANDOM_SD, size=EXTRA) for t in ids])
  return np.concatenate([XY6(ids), extra], axis=1)


A.arms.DESCRIPTORS["lodo_xy6r2"] = xy6r2
A.arms.ARMS["lodo_xy6r2"] = dict(family="film", descriptor="lodo_xy6r2", reg=False)
run_nl.BUILDERS["lodo_xy6r2"] = lambda config, ids: FiLMNonlinearReadout(config, ids, xy6r2)
run_nl.CONDITIONED["lodo_xy6r2"] = ("film_gamma", "film_beta")
A.ARMS_RUN = ("lodo_sat", "lodo_xy6", "lodo_xy6r2")
A.PROTOCOL = "review_20261115_lodo_confirm_v1"

if __name__ == "__main__":
  A.main()
