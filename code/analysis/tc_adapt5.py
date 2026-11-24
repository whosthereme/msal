import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (148021, 148022, 148023, 148024, 148025)
REC = dict()
for p in sys.argv[3:]:
  for line in Path(p).read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      key = (int(r["seed"]), r["arm"], int(r["held_task"]))
      assert key not in REC, key
      REC[key] = r
TASKS = sorted(set(t for (s, a, t) in REC if s == 148021))
assert len(TASKS) == 41
DONE = [s for s in SEEDS if all((s, a, t) in REC for a in ("film_nl", "tfilm_nl") for t in TASKS)]
NEW = [s for s in DONE if s != 148021]
family = lambda r: "local" if r.get("lane", "") in ("local", "") else "nodes"
for s in DONE:
  for t in TASKS:
    assert family(REC[(s, "film_nl", t)]) == family(REC[(s, "tfilm_nl", t)]), (s, t)
    assert REC[(s, "film_nl", t)]["lane"] == REC[(s, "tfilm_nl", t)]["lane"], (s, t)
REPORT = dict(seeds=DONE)
print("complete seeds:", DONE, " partial:", dict((s, sum((s, "tfilm_nl", t) in REC for t in TASKS)) for s in SEEDS if s not in DONE))
print("pairs per machine:", dict((f, sum(family(REC[(s, "film_nl", t)]) == f for s in DONE for t in TASKS)) for f in ("local", "nodes")))


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(v):
  v = np.asarray(v, dtype=float)
  m = float(v.mean())
  if len(v) < 2:
    return dict(n=len(v), pct=pct(m), lo=float("nan"), hi=float("nan"))
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(n=len(v), pct=pct(m), lo=pct(m - h), hi=pct(m + h))


def boot(v):
  v = np.asarray(v, dtype=float)
  idx = np.random.default_rng(0).integers(0, len(v), size=(20000, len(v)))
  m = v[idx].mean(axis=1)
  return dict(n=len(v), pct=pct(v.mean()), lo=pct(np.percentile(m, 2.5)), hi=pct(np.percentile(m, 97.5)),
    tasks_won=int((v < 0).sum()))


def fmt(r):
  if r["lo"] != r["lo"]:
    return "%+6.1f%% [one seed]" % r["pct"]
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


def flow(arm, stage, k=None, split="selection"):
  if stage == "zero_shot":
    return lambda s, t: REC[(s, arm, t)]["zero_shot"][split]
  return lambda s, t: REC[(s, arm, t)][stage][str(k)][split]


def act(arm, key):
  return lambda s, t: REC[(s, arm, t)]["action"][key]


def show(name, a, b):
  lr = dict((s, np.array([math.log(a(s, t) / b(s, t)) for t in TASKS])) for s in DONE)
  means = [float(lr[s].mean()) for s in DONE]
  entry = dict(all=tci(means), new=tci([float(lr[s].mean()) for s in NEW]) if NEW else None,
    view=boot(np.mean([lr[s] for s in DONE], axis=0)), per_seed_pct=[pct(m) for m in means],
    won=int(sum((lr[s] < 0).sum() for s in DONE)), folds=41 * len(DONE))
  REPORT[name] = entry
  line = "  %-46s %s" % (name, fmt(entry["all"]))
  if NEW and len(NEW) > 1:
    line += " | new seeds: %s" % fmt(entry["new"])
  line += " | tasks: %s, %d/41 | folds %d/%d" % (fmt(entry["view"]), entry["view"]["tasks_won"], entry["won"], entry["folds"])
  print(line)
  print("  %46s per seed: %s" % ("", " ".join("%+.1f" % x for x in entry["per_seed_pct"])))


T, H = "tfilm_nl", "film_nl"
print("== primary (trunk vs head, complete method)")
for k in (1, 16):
  show("A1 flow, k=%d" % k, flow(T, "combination", k), flow(H, "combination", k))
  show("A2 sampled chunk, k=%d" % k, act(T, "combination_%d" % k), act(H, "combination_%d" % k))
print("== secondary")
for k in (1, 16):
  for arm, label in ((T, "trunk"), (H, "head")):
    show("code beyond inversion, %s, flow, k=%d" % (label, k), flow(arm, "combination", k), flow(arm, "inversion", k))
    show("code beyond inversion, %s, chunk, k=%d" % (label, k), act(arm, "combination_%d" % k), act(arm, "inversion_%d" % k))
  show("inversion trunk vs head, flow, k=%d" % k, flow(T, "inversion", k), flow(H, "inversion", k))
show("zero-shot trunk vs head, flow", flow(T, "zero_shot"), flow(H, "zero_shot"))
show("zero-shot trunk vs head, chunk", act(T, "zero_shot"), act(H, "zero_shot"))
for arm, label in ((H, "head"), (T, "trunk")):
  for k in (1, 16):
    show("%s: method vs own zero-shot, flow, k=%d" % (label, k), flow(arm, "combination", k), flow(arm, "zero_shot"))
    show("%s: method vs own zero-shot, chunk, k=%d" % (label, k), act(arm, "combination_%d" % k), act(arm, "zero_shot"))
print("== descriptive")
for k in (2, 4, 8):
  show("A1 flow, k=%d" % k, flow(T, "combination", k), flow(H, "combination", k))
for k in (1, 16):
  show("A1 ROBUSTNESS flow, k=%d" % k, flow(T, "combination", k, "robustness"), flow(H, "combination", k, "robustness"))

print("== check: retrained film_nl vs stored ws2 scores (flow MSE; log ratio over folds)")
CHECK = dict()
for s in DONE:
  path = RAW / "combo" / ("seed%d" % s) / "folds.jsonl"
  stored = dict((int(r["held_task"]), r) for r in map(json.loads, path.read_text().splitlines()))
  for name, get in (("zero-shot", lambda r: r["zero_shot"]["selection"]), ("method k=16", lambda r: r["combination"]["16"]["selection"])):
    v = np.array([math.log(get(REC[(s, H, t)]) / get(stored[t])) for t in TASKS])
    CHECK["%d %s" % (s, name)] = dict(mean_pct=pct(v.mean()), sd_pct=100 * float(v.std(ddof=1)), identical=int((v == 0).sum()))
    print("  seed %d %-12s mean %+5.1f%% sd %4.1f%% identical %d" % (s, name, pct(v.mean()), 100 * v.std(ddof=1), int((v == 0).sum())))
REPORT["check"] = CHECK
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
