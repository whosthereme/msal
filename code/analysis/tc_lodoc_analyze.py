import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path(sys.argv[1])
SEEDS = (148026, 148027, 148028, 148029, 148030)
S, X, R = "lodo_sat", "lodo_xy6", "lodo_xy6r2"
REC = dict()
for p in sys.argv[2:]:
  for line in Path(p).read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      key = (int(r["seed"]), r["arm"], int(r["held_task"]))
      assert key not in REC, key
      REC[key] = r
TASKS = sorted(set(k[2] for k in REC))
DIRECTION = dict((k[2], REC[k]["held_direction"]) for k in REC)
missing = [(s, a, t) for s in SEEDS for a in (S, X, R) for t in TASKS if (s, a, t) not in REC]
print("records:", len(REC), " tasks:", len(TASKS), " missing:", len(missing))
assert not missing and len(TASKS) == 41
machines = dict()
for s in SEEDS:
  for t in TASKS:
    assert REC[(s, S, t)]["lane"] == REC[(s, X, t)]["lane"] == REC[(s, R, t)]["lane"]
    m = REC[(s, S, t)]["lane"][:3]
    machines[m] = machines.get(m, 0) + 1
print("task triples per machine:", machines)
REPORT = dict(machines=machines)


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(v):
  v = np.asarray(v, dtype=float)
  m = float(v.mean())
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(pct=pct(m), lo=pct(m - h), hi=pct(m + h))


def boot_dirs(per_task):
  dirs = sorted(set(DIRECTION.values()))
  by = dict((d, [v for t, v in zip(TASKS, per_task) if DIRECTION[t] == d]) for d in dirs)
  rng = np.random.default_rng(0)
  m = [np.concatenate([by[dirs[j]] for j in rng.integers(0, len(dirs), len(dirs))]).mean() for _ in range(20000)]
  return dict(pct=pct(float(np.mean(per_task))), lo=pct(np.percentile(m, 2.5)), hi=pct(np.percentile(m, 97.5)),
    tasks_won=int((np.asarray(per_task) < 0).sum()), directions_won=int(sum(np.mean(by[d]) < 0 for d in dirs)))


def fmt(r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


def show(name, a, b, key, key_b=None):
  key_b = key_b or key
  lr = dict((s, np.array([math.log(key(REC[(s, a, t)]) / key_b(REC[(s, b, t)])) for t in TASKS])) for s in SEEDS)
  means = [float(lr[s].mean()) for s in SEEDS]
  entry = dict(all=tci(means), per_seed_pct=[pct(m) for m in means], won=int(sum((v < 0).sum() for v in lr.values())),
    folds=5 * len(TASKS), view=boot_dirs(np.mean([lr[s] for s in SEEDS], axis=0)))
  REPORT[name] = entry
  v = entry["view"]
  print("  %-48s %s | dirs: %s, tasks %d/41, dirs %d/16 | folds %d/205" % (name, fmt(entry["all"]), fmt(v),
    v["tasks_won"], v["directions_won"], entry["won"]))
  print("  %48s per seed: %s" % ("", " ".join("%+.1f" % x for x in entry["per_seed_pct"])))
  return entry


def chunk(k):
  return lambda r: r["action"]["combination_%d" % k]


def flow(k):
  return lambda r: r["combination"][str(k)]["selection"]


ZC = lambda r: r["action"]["zero_shot"]
ZF = lambda r: r["zero_shot"]["selection"]
print("== primary: K1 complete method, coordinates + new random columns vs harmonic, sampled chunk")
k1 = dict((k, show("K1 xy6r2 vs sat, chunk, k=%d" % k, R, S, chunk(k))) for k in (1, 16))
print("== secondary")
k3 = show("K3 zero-shot xy6r2 vs sat, chunk", R, S, ZC)
show("K3 zero-shot xy6r2 vs xy6, chunk", R, X, ZC)
for k in (1, 16):
  show("K2 xy6r2 vs xy6, chunk, k=%d" % k, R, X, chunk(k))
for k in (1, 16):
  show("K4 xy6r2 vs sat, flow, k=%d" % k, R, S, flow(k))
  show("K4 xy6r2 vs xy6, flow, k=%d" % k, R, X, flow(k))
for k in (1, 16):
  show("K5 xy6 vs sat, chunk, k=%d (reversal)" % k, X, S, chunk(k))
show("zero-shot xy6 vs sat, chunk", X, S, ZC)
show("zero-shot xy6r2 vs sat, flow", R, S, ZF)
for arm, label in ((S, "harmonic"), (X, "coordinates"), (R, "coord + random")):
  for k in (1, 16):
    show("%s: method vs own zero-shot, chunk, k=%d" % (label, k), arm, arm, chunk(k), ZC)
for arm in (S, X, R):
  sh = [REC[(s, arm, t)]["combination"]["1"]["descriptor_shift"] for s in SEEDS for t in TASKS]
  print("  descriptor shift k=1, %s: mean %.3f" % (arm, float(np.mean(sh))))
lower = [k1[k]["all"]["hi"] < 0 for k in (1, 16)]
not_higher = [k1[k]["all"]["lo"] <= 0 for k in (1, 16)]
zs = k3["all"]["hi"] < 0
verdict = ("confirmed" if all(lower) and zs else
  "partly confirmed" if any(lower) and all(not_higher) and zs else "not confirmed")
REPORT["verdict"] = verdict
print("  VERDICT (registered rule):", verdict)
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
