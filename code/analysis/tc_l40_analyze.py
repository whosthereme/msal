import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (148021, 148022, 148023, 148024, 148025)
TASKS = list(range(1000, 1040))
T, H = "tfilm_nl", "film_nl"
REC = dict()
for p in sys.argv[3:]:
  for line in Path(p).read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      key = (int(r["seed"]), r["arm"], int(r["held_task"]))
      assert key not in REC, key
      REC[key] = r
STORED = dict()
for s in SEEDS:
  for part in ("a", "b"):
    for line in (RAW / "l40" / ("seed%d_%s" % (s, part)) / "folds.jsonl").read_text().splitlines():
      r = json.loads(line)
      STORED[(s, int(r["held_task"]))] = r
assert len(STORED) == 200
DONE = [s for s in SEEDS if all((s, a, t) in REC for a in (T, H) for t in TASKS)]
print("complete seeds:", DONE, " partial:", dict((s, sum((s, T, t) in REC for t in TASKS)) for s in SEEDS if s not in DONE))
if not DONE:
  sys.exit(0)


def machine(r):
  lane = r.get("lane", "")
  return "ws2" if lane.startswith("ws2") else ("local" if lane == "local" else "nodes")


for s in DONE:
  for t in TASKS:
    assert REC[(s, H, t)]["lane"] == REC[(s, T, t)]["lane"], (s, t)
PAIRS = dict((m, sum(machine(REC[(s, H, t)]) == m for s in DONE for t in TASKS)) for m in ("ws2", "nodes", "local"))
print("pairs per machine:", PAIRS)
REPORT = dict(seeds=DONE, pairs=PAIRS)


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


def show(name, a, b, tasks=None, extra=None):
  def ratios(s):
    use = tasks(s) if tasks else TASKS
    return np.array([math.log(a(s, t) / b(s, t)) + (extra(s, t) if extra else 0.0) for t in use])
  lr = dict((s, ratios(s)) for s in DONE)
  means = [float(lr[s].mean()) for s in DONE]
  entry = dict(all=tci(means), per_seed_pct=[pct(m) for m in means], won=int(sum((lr[s] < 0).sum() for s in DONE)),
    folds=int(sum(len(lr[s]) for s in DONE)))
  line = "  %-48s %s" % (name, fmt(entry["all"]))
  if not tasks:
    entry["view"] = boot(np.mean([lr[s] for s in DONE], axis=0))
    line += " | tasks: %s, %d/40" % (fmt(entry["view"]), entry["view"]["tasks_won"])
  line += " | folds %d/%d" % (entry["won"], entry["folds"])
  REPORT[name] = entry
  print(line)
  print("  %48s per seed: %s" % ("", " ".join("%+.1f" % x for x in entry["per_seed_pct"])))


print("== primary (trunk vs head, complete method)")
for k in (1, 16):
  show("B1 flow, k=%d" % k, flow(T, "combination", k), flow(H, "combination", k))
  show("B2 sampled chunk, k=%d" % k, act(T, "combination_%d" % k), act(H, "combination_%d" % k))
print("== secondary")
show("zero-shot trunk vs head, flow", flow(T, "zero_shot"), flow(H, "zero_shot"))
show("zero-shot trunk vs head, chunk", act(T, "zero_shot"), act(H, "zero_shot"))
for k in (1, 16):
  show("inversion trunk vs head, flow, k=%d" % k, flow(T, "inversion", k), flow(H, "inversion", k))
for arm, label in ((T, "trunk"), (H, "head")):
  for k in (1, 4, 16):
    show("code beyond inversion, %s, flow, k=%d" % (label, k), flow(arm, "combination", k), flow(arm, "inversion", k))
  for k in (1, 16):
    show("code beyond inversion, %s, chunk, k=%d" % (label, k), act(arm, "combination_%d" % k), act(arm, "inversion_%d" % k))
for arm, label in ((H, "head"), (T, "trunk")):
  for k in (1, 16):
    show("%s: method vs own zero-shot, flow, k=%d" % (label, k), flow(arm, "combination", k), flow(arm, "zero_shot"))
    show("%s: method vs own zero-shot, chunk, k=%d" % (label, k), act(arm, "combination_%d" % k), act(arm, "zero_shot"))
for ref in ("plain_nl", "film"):
  chain = lambda s, t, ref=ref: math.log(STORED[(s, t)]["zero_shot_float32"]["film_nl"]["selection"]
    / STORED[(s, t)]["zero_shot_float32"][ref]["selection"])
  show("zero-shot trunk vs stored %s, flow (chained)" % ref, flow(T, "zero_shot"), flow(H, "zero_shot"), extra=chain)
  one = lambda s, t: 1.0
  show("zero-shot head vs stored %s, flow (stored)" % ref, one, one, extra=chain)
print("== descriptive")
for k in (2, 4, 8):
  show("B1 flow, k=%d" % k, flow(T, "combination", k), flow(H, "combination", k))
for k in (1, 16):
  show("B1 test rows, flow, k=%d" % k, flow(T, "combination", k, "robustness"), flow(H, "combination", k, "robustness"))
show("zero-shot trunk vs head, test rows, flow", flow(T, "zero_shot", split="robustness"), flow(H, "zero_shot", split="robustness"))
seen = lambda s: [t for t in TASKS if not REC[(s, H, t)]["unseen_words"]]
unseen = lambda s: [t for t in TASKS if REC[(s, H, t)]["unseen_words"]]
print("  folds with all words seen: %d, with an unseen word: %d" % (sum(len(seen(s)) for s in DONE), sum(len(unseen(s)) for s in DONE)))
for label, pick in (("words seen", seen), ("unseen word", unseen)):
  show("zero-shot trunk vs head, flow, %s" % label, flow(T, "zero_shot"), flow(H, "zero_shot"), tasks=pick)
  show("zero-shot trunk vs head, chunk, %s" % label, act(T, "zero_shot"), act(H, "zero_shot"), tasks=pick)
  show("B1 flow, k=16, %s" % label, flow(T, "combination", 16), flow(H, "combination", 16), tasks=pick)
  show("B2 chunk, k=16, %s" % label, act(T, "combination_16"), act(H, "combination_16"), tasks=pick)

print("== check: retrained film_nl vs the stored study")
CHECK = dict()
for m in ("ws2", "nodes", "local"):
  keys = [(s, t) for s in DONE for t in TASKS if machine(REC[(s, H, t)]) == m]
  if not keys:
    continue
  identical = 0
  for s, t in keys:
    mine, st = REC[(s, H, t)], STORED[(s, t)]
    same = mine["zero_shot"] == st["zero_shot"]
    for k in ("1", "2", "4", "8", "16"):
      same = same and mine["inversion"][k] == st["inversion"][k]
      same = same and all(mine["combination"][k][x] == st["combination"][k][x] for x in ("selection", "robustness"))
    identical += bool(same)
  z = np.array([math.log(REC[(s, H, t)]["zero_shot"]["selection"] / STORED[(s, t)]["zero_shot"]["selection"]) for s, t in keys])
  c = np.array([math.log(REC[(s, H, t)]["combination"]["16"]["selection"] / STORED[(s, t)]["combination"]["16"]["selection"]) for s, t in keys])
  CHECK[m] = dict(folds=len(keys), bit_identical=identical, zero_mean_pct=pct(z.mean()), zero_sd_pct=100 * float(z.std(ddof=1)),
    method16_mean_pct=pct(c.mean()), method16_sd_pct=100 * float(c.std(ddof=1)))
  print("  %-5s folds %3d  bit-identical %3d  zero-shot mean %+5.2f%% sd %4.1f%%  method k=16 mean %+5.2f%% sd %4.1f%%" % (
    m, len(keys), identical, pct(z.mean()), 100 * z.std(ddof=1), pct(c.mean()), 100 * c.std(ddof=1)))
REPORT["check"] = CHECK
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
