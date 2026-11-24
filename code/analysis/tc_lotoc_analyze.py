import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(sys.argv[1])
SEEDS = (148021, 148022, 148023, 148024, 148025)
H, X = "film_nl", "lodo_xy6r2"
REC = dict()
for p in sys.argv[2:]:
  for line in Path(p).read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      key = (int(r["seed"]), r["arm"], int(r["held_task"]))
      assert key not in REC, key
      REC[key] = r
TASKS = sorted(set(k[2] for k in REC))
missing = [(s, a, t) for s in SEEDS for a in (H, X) for t in TASKS if (s, a, t) not in REC]
print("records:", len(REC), " tasks:", len(TASKS), " missing:", len(missing))
assert not missing and len(TASKS) == 41
machines = dict()
for s in SEEDS:
  for t in TASKS:
    assert REC[(s, H, t)]["lane"] == REC[(s, X, t)]["lane"]
    m = REC[(s, H, t)]["lane"][:3]
    machines[m] = machines.get(m, 0) + 1
print("pairs per machine:", machines)
REPORT = dict(machines=machines)


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(v):
  v = np.asarray(v, dtype=float)
  m = float(v.mean())
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(pct=pct(m), lo=pct(m - h), hi=pct(m + h))


def boot(v):
  v = np.asarray(v, dtype=float)
  m = v[np.random.default_rng(0).integers(0, len(v), size=(20000, len(v)))].mean(axis=1)
  return dict(pct=pct(v.mean()), lo=pct(np.percentile(m, 2.5)), hi=pct(np.percentile(m, 97.5)), tasks_won=int((v < 0).sum()))


def fmt(r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


def show(name, a, b, key, key_b=None):
  key_b = key_b or key
  lr = dict((s, np.array([math.log(key(REC[(s, a, t)]) / key_b(REC[(s, b, t)])) for t in TASKS])) for s in SEEDS)
  means = [float(lr[s].mean()) for s in SEEDS]
  entry = dict(all=tci(means), per_seed_pct=[pct(m) for m in means], won=int(sum((v < 0).sum() for v in lr.values())),
    view=boot(np.mean([lr[s] for s in SEEDS], axis=0)))
  REPORT[name] = entry
  print("  %-46s %s | tasks: %s, %d/41 | folds %d/205" % (name, fmt(entry["all"]), fmt(entry["view"]),
    entry["view"]["tasks_won"], entry["won"]))
  print("  %46s per seed: %s" % ("", " ".join("%+.1f" % x for x in entry["per_seed_pct"])))
  return entry


chunk = lambda k: (lambda r: r["action"]["combination_%d" % k])
flow = lambda k: (lambda r: r["combination"][str(k)]["selection"])
ZC = lambda r: r["action"]["zero_shot"]
ZF = lambda r: r["zero_shot"]["selection"]
print("== primary")
p1 = dict((k, show("P1 xy6r2 vs harmonic, chunk, k=%d" % k, X, H, chunk(k))) for k in (1, 16))
p2 = show("P2 zero-shot xy6r2 vs harmonic, chunk", X, H, ZC)
print("== secondary")
for k in (1, 16):
  show("xy6r2 vs harmonic, flow, k=%d" % k, X, H, flow(k))
show("zero-shot xy6r2 vs harmonic, flow", X, H, ZF)
for k in (1, 16):
  show("inversion xy6r2 vs harmonic, chunk, k=%d" % k, X, H, lambda r, k=k: r["action"]["inversion_%d" % k])
for arm, label in ((H, "harmonic"), (X, "coord + random")):
  for k in (1, 16):
    show("%s: method vs own zero-shot, chunk, k=%d" % (label, k), arm, arm, chunk(k), ZC)
for arm in (H, X):
  sh = [REC[(s, arm, t)]["combination"]["1"]["descriptor_shift"] for s in SEEDS for t in TASKS]
  print("  descriptor shift k=1, %s: mean %.3f" % (arm, float(np.mean(sh))))
ok = all(p1[k]["all"]["lo"] <= 0 for k in (1, 16)) and p2["all"]["lo"] <= 0
REPORT["general_default_proposed"] = bool(ok)
print("  RULE: xy6r2 proposed as general default descriptor:", "YES" if ok else "NO")
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
