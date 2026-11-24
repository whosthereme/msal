import json
import math
import sys
from pathlib import Path

import numpy as np

OUT, RAW = Path(sys.argv[1]), Path(sys.argv[2])
SEED = 148021
ARMS = ("tin_nl", "tfilm_nl", "tboth_nl")


def read(path):
  return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


stored = dict((int(r["held_task"]), r["main_selection"])
  for r in read(RAW / "n1n2" / ("blind_nl_film_nl_seed%d" % SEED) / "folds.jsonl"))
recs = dict((a, read(OUT / ("%s_seed%d.jsonl" % (a, SEED)))) for a in ARMS)
for a in ARMS:
  info = json.loads((OUT / ("%s_seed%d_info.json" % (a, SEED))).read_text())
  r = recs[a]
  same_ref = all(x["stored_film_nl_flow"] == stored[x["held_task"]] for x in r)
  print("%-9s folds %d, distinct tasks %d, train_steps %s, smoke %s, data hash verified %s, selfcheck gap %s,"
    " reference scores identical to the handout copy %s" % (a, len(r), len(set(x["held_task"] for x in r)),
    sorted(set(x["train_steps"] for x in r)), info["smoke"], info["data_sha256"] != "not_verified",
    info["selfcheck"]["parent_gap"], same_ref))


def compare(a, b, key):
  x = dict((r["held_task"], r[key]) for r in recs[a])
  y = dict((r["held_task"], r[key]) for r in recs[b])
  lr = np.array([math.log(x[t] / y[t]) for t in sorted(x)])
  rng = np.random.default_rng(0)
  boot = np.exp(rng.choice(lr, size=(20000, len(lr)), replace=True).mean(axis=1)) - 1
  return "%+5.1f%% [%+5.1f, %+5.1f] won %d/41" % (100 * (math.exp(lr.mean()) - 1), 100 * np.percentile(boot, 2.5),
    100 * np.percentile(boot, 97.5), int((lr < 0).sum()))


print()
print("post hoc (not in the plan):")
for a, b in (("tboth_nl", "tfilm_nl"), ("tfilm_nl", "tin_nl")):
  print("  %s vs %s: flow %s | action %s" % (a, b, compare(a, b, "flow_selection"), compare(a, b, "action_selection")))
