import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import tc_lodo_adapt as A
import numpy as np
from msal_ext import run_nl
from msal_ext.nl import FiLMNonlinearReadout

EXTRA = 15
RANDOM_SEED = 20261114
RANDOM_SD = 0.7
XY6 = A.arms.DESCRIPTORS["lodo_xy6"]


def random_columns(ids):
  return np.stack([np.random.default_rng([RANDOM_SEED, int(t)]).normal(0.0, RANDOM_SD, size=EXTRA) for t in ids])


def xy6r(ids):
  return np.concatenate([XY6(ids), random_columns(ids)], axis=1)


def xy6z(ids):
  return np.concatenate([XY6(ids), np.zeros((len(ids), EXTRA))], axis=1)


def builder(fn):
  return lambda config, ids: FiLMNonlinearReadout(config, ids, fn)


for name, fn in (("lodo_xy6r", xy6r), ("lodo_xy6z", xy6z)):
  A.arms.DESCRIPTORS[name] = fn
  A.arms.ARMS[name] = dict(family="film", descriptor=name, reg=False)
  run_nl.BUILDERS[name] = builder(fn)
  run_nl.CONDITIONED[name] = ("film_gamma", "film_beta")
A.ARMS_RUN = ("lodo_xy6", "lodo_xy6r", "lodo_xy6z")
A.PROTOCOL = "review_20261115_lodo_dim_v1"

if __name__ == "__main__":
  A.main()
