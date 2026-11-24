import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RAW = Path(sys.argv[1])
STORE = dict()


def stored(seed):
  if seed not in STORE:
    f = dict((json.loads(x)["held_task"], json.loads(x)["main_selection"])
      for x in open(RAW / "n1n2" / ("blind_nl_film_nl_seed%d" % seed) / "folds.jsonl"))
    a = dict((json.loads(x)["task"], json.loads(x)["action_selection"])
      for x in open(RAW / "ideal" / ("zero_seed%d.jsonl" % seed)))
    STORE[seed] = (f, a)
  return STORE[seed]


def pct(v):
  return 100 * (math.exp(v) - 1)


for spec in sys.argv[2:]:
  label, pattern = spec.split("=", 1)
  lines = [x for p in sorted(glob.glob(pattern)) for x in open(p) if x.strip()]
  recs = dict(((r["seed"], r["held_task"]), r) for r in map(json.loads, lines))
  rows = []
  for (seed, task), r in sorted(recs.items()):
    f, a = stored(seed)
    rows.append((math.log(r["flow_selection"] / f[task]), math.log(r["action_selection"] / a[task])))
  n = len(rows)
  print("%s: %d folds %s" % (label, n, sorted(recs)))
  for name, col in (("flow MSE", 0), ("action error", 1)):
    v = np.array([r[col] for r in rows])
    if n < 2:
      continue
    half = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    inside = pct(v.mean() - half) > -2 and pct(v.mean() + half) < 2
    print("  %-12s mean %+.2f%% [%+.2f, %+.2f]  per-fold SD %.1f%%  largest |diff| %.1f%%  bit-identical %d/%d  -> %s" % (
      name, pct(v.mean()), pct(v.mean() - half), pct(v.mean() + half), 100 * v.std(ddof=1), pct(np.abs(v).max()),
      int((v == 0).sum()), n, "inside +/-2%" if inside else "NOT inside +/-2%"))
