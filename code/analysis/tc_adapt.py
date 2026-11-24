import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
REC = dict()
for p in sys.argv[2:]:
  for line in Path(p).read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      REC[(r["arm"], int(r["held_task"]))] = r
TASKS = sorted(t for (a, t) in REC if a == "tfilm_nl" and ("film_nl", t) in REC)
MACHINE = lambda r: "local" if r.get("lane", "") in ("local", "") else "nodes"
for t in TASKS:
  assert MACHINE(REC[("film_nl", t)]) == MACHINE(REC[("tfilm_nl", t)]), t
REPORT = dict(tasks=len(TASKS))


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def boot(lr):
  v = np.asarray(lr, dtype=float)
  idx = np.random.default_rng(0).integers(0, len(v), size=(20000, len(v)))
  m = v[idx].mean(axis=1)
  return dict(n=len(v), pct=pct(v.mean()), lo=pct(np.percentile(m, 2.5)), hi=pct(np.percentile(m, 97.5)),
    won=int((v < 0).sum()))


def show(name, get_a, get_b):
  lr = [math.log(get_a(t) / get_b(t)) for t in TASKS]
  r = boot(lr)
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  print("  %-52s %+6.1f%% [%+6.1f, %+6.1f] %-6s won %d/%d" % (name, r["pct"], r["lo"], r["hi"], word, r["won"], r["n"]))
  REPORT[name] = r


def flow(arm, stage, k=None, split="selection"):
  if stage == "zero_shot":
    return lambda t: REC[(arm, t)]["zero_shot"][split]
  return lambda t: REC[(arm, t)][stage][str(k)][split]


def act(arm, key):
  return lambda t: REC[(arm, t)]["action"][key]


print("tasks with both arms: %d (local %d, nodes %d)" % (len(TASKS),
  sum(MACHINE(REC[("film_nl", t)]) == "local" for t in TASKS), sum(MACHINE(REC[("film_nl", t)]) == "nodes" for t in TASKS)))
print("== primary")
for k in (1, 16):
  show("A1 MSAL trunk vs MSAL head, flow, k=%d" % k, flow("tfilm_nl", "combination", k), flow("film_nl", "combination", k))
  show("A2 MSAL trunk vs MSAL head, sampled chunk, k=%d" % k, act("tfilm_nl", "combination_%d" % k),
    act("film_nl", "combination_%d" % k))
print("== secondary")
for k in (1, 16):
  show("code beyond inversion, trunk, flow, k=%d" % k, flow("tfilm_nl", "combination", k), flow("tfilm_nl", "inversion", k))
  show("code beyond inversion, head (replication), flow, k=%d" % k, flow("film_nl", "combination", k),
    flow("film_nl", "inversion", k))
  show("inversion trunk vs head, flow, k=%d" % k, flow("tfilm_nl", "inversion", k), flow("film_nl", "inversion", k))
  show("code beyond inversion, trunk, sampled chunk, k=%d" % k, act("tfilm_nl", "combination_%d" % k),
    act("tfilm_nl", "inversion_%d" % k))
  show("code beyond inversion, head, sampled chunk, k=%d" % k, act("film_nl", "combination_%d" % k),
    act("film_nl", "inversion_%d" % k))
show("zero-shot trunk vs head, flow", flow("tfilm_nl", "zero_shot"), flow("film_nl", "zero_shot"))
show("zero-shot trunk vs head, sampled chunk", act("tfilm_nl", "zero_shot"), act("film_nl", "zero_shot"))
print("== descriptive")
for k in (1, 2, 4, 8, 16):
  show("MSAL trunk vs head, flow, k=%d" % k, flow("tfilm_nl", "combination", k), flow("film_nl", "combination", k))
  show("MSAL trunk vs head, ROBUSTNESS flow, k=%d" % k, flow("tfilm_nl", "combination", k, "robustness"),
    flow("film_nl", "combination", k, "robustness"))
for arm in ("film_nl", "tfilm_nl"):
  for k in (1, 16):
    show("%s: MSAL vs its zero-shot head, flow, k=%d" % (arm, k), flow(arm, "combination", k), flow(arm, "zero_shot"))
    show("%s: MSAL vs its zero-shot head, sampled chunk, k=%d" % (arm, k), act(arm, "combination_%d" % k),
      act(arm, "zero_shot"))
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
