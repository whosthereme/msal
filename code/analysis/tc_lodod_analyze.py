import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT, STORED = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (148021, 148022, 148023, 148024, 148025)
X, R, Z, S = "lodo_xy6", "lodo_xy6r", "lodo_xy6z", "lodo_sat"


def load(paths):
  rec = dict()
  for p in paths:
    for line in Path(p).read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        key = (int(r["seed"]), r["arm"], int(r["held_task"]))
        assert key not in rec, key
        rec[key] = r
  return rec


NEW = load(sys.argv[3:])
OLD = load(sorted(glob.glob(str(STORED / "*.jsonl"))))
TASKS = sorted(set(k[2] for k in NEW))
DIRECTION = dict((k[2], NEW[k]["held_direction"]) for k in NEW)
missing = [(s, a, t) for s in SEEDS for a in (X, R, Z) for t in TASKS if (s, a, t) not in NEW]
print("new records:", len(NEW), " tasks:", len(TASKS), " missing:", len(missing))
assert not missing and len(TASKS) == 41
FIELDS = ("zero_shot", "inversion", "combination", "action", "score_forward")
IDENT = dict(((s, t), all(NEW[(s, X, t)][f] == OLD[(s, X, t)][f] for f in FIELDS)) for s in SEEDS for t in TASKS)
print("xy6 rerun bit-identical to stored: %d of %d" % (sum(IDENT.values()), len(IDENT)))
for s in SEEDS:
  for t in TASKS:
    assert NEW[(s, X, t)]["lane"] == NEW[(s, R, t)]["lane"] == NEW[(s, Z, t)]["lane"]


def get(s, arm, t):
  return OLD[(s, arm, t)] if arm == S else NEW[(s, arm, t)]


REPORT = dict(bit_identical=sum(IDENT.values()))


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(v):
  v = np.asarray(v, dtype=float)
  m = float(v.mean())
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(pct=pct(m), lo=pct(m - h), hi=pct(m + h))


def fmt(r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


def boot_dirs(task_list, per_task):
  dirs = sorted(set(DIRECTION[t] for t in task_list))
  by = dict((d, [v for t, v in zip(task_list, per_task) if DIRECTION[t] == d]) for d in dirs)
  rng = np.random.default_rng(0)
  m = [np.concatenate([by[dirs[j]] for j in rng.integers(0, len(dirs), len(dirs))]).mean() for _ in range(20000)]
  return dict(pct=pct(float(np.mean(per_task))), lo=pct(np.percentile(m, 2.5)), hi=pct(np.percentile(m, 97.5)))


def show(name, a, b, key, key_b=None):
  key_b = key_b or key
  use = dict((s, [t for t in TASKS if (S not in (a, b)) or IDENT[(s, t)]]) for s in SEEDS)
  lr = dict((s, np.array([math.log(key(get(s, a, t)) / key_b(get(s, b, t))) for t in use[s]])) for s in SEEDS)
  means = [float(lr[s].mean()) for s in SEEDS]
  common = [t for t in TASKS if all(t in use[s] for s in SEEDS)]
  per_task = [np.mean([lr[s][use[s].index(t)] for s in SEEDS]) for t in common]
  entry = dict(all=tci(means), per_seed_pct=[pct(m) for m in means], folds=int(sum(len(v) for v in lr.values())),
    won=int(sum((v < 0).sum() for v in lr.values())), view=boot_dirs(common, per_task))
  REPORT[name] = entry
  print("  %-46s %s | dirs: %s | folds %d/%d" % (name, fmt(entry["all"]), fmt(entry["view"]), entry["won"], entry["folds"]))
  print("  %46s per seed: %s" % ("", " ".join("%+.1f" % x for x in entry["per_seed_pct"])))
  return entry


def chunk(k):
  return lambda r: r["action"]["combination_%d" % k]


def flow(k):
  return lambda r: r["combination"][str(k)]["selection"]


print("== primary: complete method, sampled-chunk error")
res = dict()
for label, arm in (("random", R), ("zero", Z)):
  for k in (1, 16):
    res[(arm, "x", k)] = show("W %s cols vs xy6, k=%d" % (label, k), arm, X, chunk(k))
    res[(arm, "s", k)] = show("W %s cols vs sat, k=%d" % (label, k), arm, S, chunk(k))
for label, arm in (("random", R), ("zero", Z)):
  lower = all(res[(arm, "x", k)]["all"]["hi"] < 0 for k in (1, 16))
  not_higher = all(res[(arm, "s", k)]["all"]["lo"] <= 0 for k in (1, 16))
  verdict = ("width explains the reversal" if lower and not_higher else
    "width explains part of it" if lower else "not supported")
  REPORT["verdict_" + label] = verdict
  print("  verdict, %s columns: %s" % (label, verdict))
print("== secondary")
for label, arm in (("random", R), ("zero", Z)):
  for k in (1, 16):
    show("%s cols vs xy6, flow, k=%d" % (label, k), arm, X, flow(k))
    show("%s cols vs sat, flow, k=%d" % (label, k), arm, S, flow(k))
  show("%s cols vs xy6, zero-shot chunk" % label, arm, X, lambda r: r["action"]["zero_shot"])
  show("%s cols vs xy6, zero-shot flow" % label, arm, X, lambda r: r["zero_shot"]["selection"])
  for k in (1, 16):
    show("%s cols: method vs own zero-shot, chunk, k=%d" % (label, k), arm, arm, chunk(k), lambda r: r["action"]["zero_shot"])
  show("%s cols vs xy6, inversion chunk, k=1" % label, arm, X, lambda r: r["action"]["inversion_1"])
show("xy6 vs sat, chunk, k=1 (replication)", X, S, chunk(1))
for arm in (X, R, Z):
  sh = [NEW[(s, arm, t)]["combination"]["1"]["descriptor_shift"] for s in SEEDS for t in TASKS]
  print("  descriptor shift k=1, %s: mean %.3f" % (arm, float(np.mean(sh))))
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
