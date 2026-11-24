import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(sys.argv[1])
SEEDS = (148021, 148022, 148023, 148024, 148025)
X, S = "lodo_xy6", "lodo_sat"
REC = dict()
for p in sys.argv[2:]:
  for line in Path(p).read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      key = (int(r["seed"]), r["arm"], int(r["held_task"]))
      assert key not in REC, key
      REC[key] = r
TASKS = sorted(set(t for (_, _, t) in REC))
DIRECTION = dict((t, REC[k]["held_direction"]) for k in REC for t in [k[2]])
missing = [(s, a, t) for s in SEEDS for a in (X, S) for t in TASKS if (s, a, t) not in REC]
print("records:", len(REC), " tasks:", len(TASKS), " directions:", len(set(DIRECTION.values())), " missing:", len(missing))
assert not missing and len(TASKS) == 41, missing[:5]
for s in SEEDS:
  for t in TASKS:
    assert REC[(s, X, t)]["lane"] == REC[(s, S, t)]["lane"], (s, t)
machines = dict()
for s in SEEDS:
  for t in TASKS:
    m = REC[(s, X, t)]["lane"][:3]
    machines[m] = machines.get(m, 0) + 1
print("pairs per machine:", machines)
REPORT = dict(machines=machines)


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(v):
  v = np.asarray(v, dtype=float)
  m = float(v.mean())
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(n=len(v), pct=pct(m), lo=pct(m - h), hi=pct(m + h))


def boot_directions(per_task):
  dirs = sorted(set(DIRECTION.values()))
  by = dict((d, [per_task[i] for i, t in enumerate(TASKS) if DIRECTION[t] == d]) for d in dirs)
  rng = np.random.default_rng(0)
  means = np.empty(20000)
  for i in range(20000):
    pick = rng.integers(0, len(dirs), size=len(dirs))
    vals = np.concatenate([by[dirs[j]] for j in pick])
    means[i] = vals.mean()
  return dict(pct=pct(float(np.mean(per_task))), lo=pct(np.percentile(means, 2.5)), hi=pct(np.percentile(means, 97.5)),
    tasks_won=int((np.asarray(per_task) < 0).sum()),
    directions_won=int(sum(np.mean(by[d]) < 0 for d in dirs)))


def fmt(r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


def show(name, a, b):
  lr = dict((s, np.array([math.log(a(s, t) / b(s, t)) for t in TASKS])) for s in SEEDS)
  means = [float(lr[s].mean()) for s in SEEDS]
  entry = dict(all=tci(means), four=tci(means[1:]), per_seed_pct=[pct(m) for m in means],
    won=int(sum((lr[s] < 0).sum() for s in SEEDS)), folds=5 * len(TASKS))
  entry["view"] = boot_directions(np.mean([lr[s] for s in SEEDS], axis=0))
  REPORT[name] = entry
  v = entry["view"]
  print("  %-50s %s | dirs: %s, tasks %d/41, dirs %d/16 | folds %d/%d" % (name, fmt(entry["all"]), fmt(v),
    v["tasks_won"], v["directions_won"], entry["won"], entry["folds"]))
  print("  %50s per seed: %s" % ("", " ".join("%+.1f" % x for x in entry["per_seed_pct"])))
  return entry


def flow(arm, stage, k=None, split="selection"):
  if stage == "zero_shot":
    return lambda s, t: REC[(s, arm, t)]["zero_shot"][split]
  return lambda s, t: REC[(s, arm, t)][stage][str(k)][split]


def act(arm, key):
  return lambda s, t: REC[(s, arm, t)]["action"][key]


print("== primary: complete method, coordinate (xy6) vs harmonic (sat)")
c1 = dict()
for k in (1, 16):
  c1[k] = show("C1 sampled chunk, k=%d" % k, act(X, "combination_%d" % k), act(S, "combination_%d" % k))
  show("C2 flow, k=%d" % k, flow(X, "combination", k), flow(S, "combination", k))
lower = all(c1[k]["all"]["hi"] < 0 for k in (1, 16))
higher = any(c1[k]["all"]["lo"] > 0 for k in (1, 16))
REPORT["c1_reading"] = "advantage holds" if lower else ("reversed" if higher else "gap closed or mixed")
print("  C1 reading:", REPORT["c1_reading"])
print("== secondary")
for arm, label in ((S, "harmonic"), (X, "coordinate")):
  for k in (1, 16):
    show("C3 %s: method vs own zero-shot, chunk, k=%d" % (label, k), act(arm, "combination_%d" % k), act(arm, "zero_shot"))
    show("C3 %s: method vs own zero-shot, flow, k=%d" % (label, k), flow(arm, "combination", k), flow(arm, "zero_shot"))
for arm, label in ((S, "harmonic"), (X, "coordinate")):
  for k in (1, 4, 16):
    show("C4 %s: code beyond inversion, flow, k=%d" % (label, k), flow(arm, "combination", k), flow(arm, "inversion", k))
  for k in (1, 16):
    show("C4 %s: code beyond inversion, chunk, k=%d" % (label, k), act(arm, "combination_%d" % k), act(arm, "inversion_%d" % k))
for k in (1, 16):
  show("C5 inversion, xy6 vs sat, chunk, k=%d" % k, act(X, "inversion_%d" % k), act(S, "inversion_%d" % k))
  show("C5 inversion, xy6 vs sat, flow, k=%d" % k, flow(X, "inversion", k), flow(S, "inversion", k))
show("C6 zero-shot, xy6 vs sat, chunk", act(X, "zero_shot"), act(S, "zero_shot"))
show("C6 zero-shot, xy6 vs sat, flow", flow(X, "zero_shot"), flow(S, "zero_shot"))
print("== descriptive")
for k in (2, 4, 8):
  show("C2 flow, k=%d" % k, flow(X, "combination", k), flow(S, "combination", k))
for k in (1, 16):
  show("C2 test rows, flow, k=%d" % k, flow(X, "combination", k, "robustness"), flow(S, "combination", k, "robustness"))
for arm in (S, X):
  for k in ("1", "16"):
    sh = [REC[(s, arm, t)]["combination"][k]["descriptor_shift"] for s in SEEDS for t in TASKS]
    print("  descriptor shift k=%s, %s: mean %.3f" % (k, arm, float(np.mean(sh))))
for k in (1, 16):
  print("  four added seeds, C1 k=%d: %s" % (k, fmt(REPORT["C1 sampled chunk, k=%d" % k]["four"])))
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
