import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT, STORED_DIR = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (148021, 148022, 148023, 148024, 148025)
UNSEEN = (1000, 1001, 1005, 1006, 1008, 1009, 1010, 1011, 1012, 1017, 1021, 1025, 1026, 1027, 1028, 1030, 1031)
ALL = tuple(range(1000, 1040))
T, H = "tfilm_nl", "film_nl"


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
OLD = load(sorted(glob.glob(str(STORED_DIR / "out_*.jsonl"))))
missing = [(s, a, t) for s in SEEDS for a in (T, H) for t in UNSEEN if (s, a, t) not in NEW]
print("new records:", len(NEW), " missing:", len(missing))
assert not missing, missing[:5]
for s in SEEDS:
  for t in UNSEEN:
    assert NEW[(s, T, t)]["lane"] == NEW[(s, H, t)]["lane"], (s, t)
    assert NEW[(s, T, t)]["zeroed_entries"] > 0
REPORT = dict()


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(v):
  v = np.asarray(v, dtype=float)
  m = float(v.mean())
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(n=len(v), pct=pct(m), lo=pct(m - h), hi=pct(m + h), mean_log=m)


def boot(v):
  v = np.asarray(v, dtype=float)
  idx = np.random.default_rng(0).integers(0, len(v), size=(20000, len(v)))
  m = v[idx].mean(axis=1)
  return dict(n=len(v), pct=pct(v.mean()), lo=pct(np.percentile(m, 2.5)), hi=pct(np.percentile(m, 97.5)),
    tasks_won=int((v < 0).sum()))


def fmt(r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


def get(s, arm, t, which, path):
  if t in UNSEEN:
    node = NEW[(s, arm, t)][which]
  else:
    node = OLD[(s, arm, t)]
  for p in path:
    node = node[p]
  return node


def show(name, a, b, tasks):
  lr = dict((s, np.array([math.log(a(s, t) / b(s, t)) for t in tasks])) for s in SEEDS)
  means = [float(lr[s].mean()) for s in SEEDS]
  entry = dict(all=tci(means), per_seed_pct=[pct(m) for m in means], won=int(sum((lr[s] < 0).sum() for s in SEEDS)),
    folds=int(sum(len(lr[s]) for s in SEEDS)))
  entry["view"] = boot(np.mean([lr[s] for s in SEEDS], axis=0))
  REPORT[name] = entry
  print("  %-52s %s | tasks: %s, %d/%d | folds %d/%d" % (name, fmt(entry["all"]), fmt(entry["view"]),
    entry["view"]["tasks_won"], len(tasks), entry["won"], entry["folds"]))
  print("  %52s per seed: %s" % ("", " ".join("%+.1f" % x for x in entry["per_seed_pct"])))
  return entry


def f(arm, which, *path):
  return lambda s, t: get(s, arm, t, which, path)


ZF, ZC = ("zero_shot", "selection"), ("action", "zero_shot")
print("== E (explanation), 85 folds with an unseen word")
e1f = show("E1 zero-shot trunk, projected vs raw, flow", f(T, "projected", *ZF), f(T, "raw", *ZF), UNSEEN)
show("E1 zero-shot trunk, projected vs raw, chunk", f(T, "projected", *ZC), f(T, "raw", *ZC), UNSEEN)
e2f = show("E2 zero-shot trunk vs head, projected, flow", f(T, "projected", *ZF), f(H, "projected", *ZF), UNSEEN)
show("E2 zero-shot trunk vs head, projected, chunk", f(T, "projected", *ZC), f(H, "projected", *ZC), UNSEEN)
e2r = show("   (raw gap) zero-shot trunk vs head, raw, flow", f(T, "raw", *ZF), f(H, "raw", *ZF), UNSEEN)
show("   (raw gap) zero-shot trunk vs head, raw, chunk", f(T, "raw", *ZC), f(H, "raw", *ZC), UNSEEN)
show("E3 zero-shot head, projected vs raw, flow", f(H, "projected", *ZF), f(H, "raw", *ZF), UNSEEN)
show("E3 zero-shot head, projected vs raw, chunk", f(H, "projected", *ZC), f(H, "raw", *ZC), UNSEEN)
supported = e1f["all"]["hi"] < 0 and abs(e2f["all"]["mean_log"]) < 0.5 * e2r["all"]["mean_log"]
REPORT["explanation_supported"] = bool(supported)
print("  explanation (E1 flow lower and projected gap < half the raw gap):", "SUPPORTED" if supported else "NOT SUPPORTED",
  " gap log raw %.3f projected %.3f" % (e2r["all"]["mean_log"], e2f["all"]["mean_log"]))

print("== D (decision), 200 folds, complete method, trunk vs head, both projected")
worse = False
for k in (1, 16):
  d1 = show("D1 flow, k=%d" % k, f(T, "projected", "combination", str(k), "selection"),
    f(H, "projected", "combination", str(k), "selection"), ALL)
  d2 = show("D2 sampled chunk, k=%d" % k, f(T, "projected", "action", "combination_%d" % k),
    f(H, "projected", "action", "combination_%d" % k), ALL)
  worse = worse or d1["all"]["lo"] > 0 or d2["all"]["lo"] > 0
REPORT["trunk_proposed"] = not worse
print("  rule: trunk head with projection proposed as main method:", "YES" if not worse else "NO")
print("  same on the 85 folds only:")
for k in (1, 16):
  show("D1 flow, k=%d, 85 folds" % k, f(T, "projected", "combination", str(k), "selection"),
    f(H, "projected", "combination", str(k), "selection"), UNSEEN)
  show("D2 sampled chunk, k=%d, 85 folds" % k, f(T, "projected", "action", "combination_%d" % k),
    f(H, "projected", "action", "combination_%d" % k), UNSEEN)

print("== P (projection for the draft's head), 85 folds, complete method, projected vs raw")
adopt = True
for arm, label in ((H, "head"), (T, "trunk")):
  for k in (1, 16):
    pf = show("P %s flow, k=%d" % (label, k), f(arm, "projected", "combination", str(k), "selection"),
      f(arm, "raw", "combination", str(k), "selection"), UNSEEN)
    pc = show("P %s sampled chunk, k=%d" % (label, k), f(arm, "projected", "action", "combination_%d" % k),
      f(arm, "raw", "action", "combination_%d" % k), UNSEEN)
    if arm == H:
      adopt = adopt and pc["all"]["hi"] < 0 and pf["all"]["lo"] <= 0
REPORT["projection_proposed"] = bool(adopt)
print("  rule: projection proposed for MSAL's head:", "YES" if adopt else "NO")

print("== secondary, descriptive (85 folds)")
for arm, label in ((T, "trunk"), (H, "head")):
  for k in (1, 16):
    show("inversion %s, projected vs raw, flow, k=%d" % (label, k), f(arm, "projected", "inversion", str(k), "selection"),
      f(arm, "raw", "inversion", str(k), "selection"), UNSEEN)
  for k in (2, 4, 8):
    show("complete %s, projected vs raw, flow, k=%d" % (label, k), f(arm, "projected", "combination", str(k), "selection"),
      f(arm, "raw", "combination", str(k), "selection"), UNSEEN)
  show("zero-shot %s, projected vs raw, test rows" % label, f(arm, "projected", "zero_shot", "robustness"),
    f(arm, "raw", "zero_shot", "robustness"), UNSEEN)
for k in (2, 4, 8):
  show("D1 flow, k=%d (200 folds)" % k, f(T, "projected", "combination", str(k), "selection"),
    f(H, "projected", "combination", str(k), "selection"), ALL)
for arm in (T, H):
  for which in ("raw", "projected"):
    sh = [NEW[(s, arm, t)][which]["combination"]["16"]["descriptor_shift"] for s in SEEDS for t in UNSEEN]
    print("  descriptor shift k=16, %s, %s: mean %.3f" % (arm, which, float(np.mean(sh))))

print("== check: raw results vs the stored PLAN_L40_TRUNK.md records")
CHECK = dict()
for s in SEEDS:
  for t in UNSEEN:
    for arm in (T, H):
      n, o = NEW[(s, arm, t)], OLD[(s, arm, t)]
      same = all(n["raw"][x] == o[x] for x in ("zero_shot", "action", "inversion", "combination"))
      key = "%s->%s" % (o["lane"][:3], n["lane"][:3])
      c = CHECK.setdefault(key, [0, 0])
      c[0] += 1
      c[1] += bool(same)
for key, (n, same) in sorted(CHECK.items()):
  print("  stored lane %s new lane: %d records, %d bit-identical" % (key, n, same))
REPORT["check"] = CHECK
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
