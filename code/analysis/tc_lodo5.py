import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RES, RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEEDS = (148021, 148022, 148023, 148024, 148025)
ARMS = ("lodo_plain", "lodo_sat", "lodo_xy6", "lodo_txy6", "lodo_h9")
METRIC = dict(act="action_selection", flow="flow_selection", rob="flow_robustness")
PAIRS = (("L1", "lodo_xy6", "lodo_sat"), ("L2", "lodo_xy6", "lodo_plain"), ("L3", "lodo_sat", "lodo_plain"),
  ("L4", "lodo_txy6", "lodo_xy6"), ("txy6_plain", "lodo_txy6", "lodo_plain"), ("h9_xy6", "lodo_h9", "lodo_xy6"))
DIRS_OUT = [RES / "local" / "out_lodo", RES / "final_nodes" / "out_lodo"]
REPORT = dict()


def read(path):
  if not path.exists():
    return []
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def load(arm, seed):
  out = dict()
  for d in DIRS_OUT:
    for r in read(d / ("%s_seed%d.jsonl" % (arm, seed))):
      out.setdefault(int(r["held_task"]), r)
  return out


DATA = dict(((a, s), load(a, s)) for a in ARMS for s in SEEDS)
TASKS = sorted(DATA[("lodo_plain", 148021)])
assert len(TASKS) == 41
DIR = dict((t, int(DATA[("lodo_plain", 148021)][t]["held_direction"])) for t in TASKS)
DIRS = sorted(set(DIR.values()))


def complete(arm, seed):
  return sorted(DATA[(arm, seed)]) == TASKS


def ratios(a, b, seed, metric):
  return np.array([math.log(float(DATA[(a, seed)][t][METRIC[metric]]) / float(DATA[(b, seed)][t][METRIC[metric]]))
    for t in TASKS])


def tci(v):
  v = np.asarray(v, dtype=float)
  m = float(v.mean())
  if len(v) < 2:
    return dict(n=len(v), pct=pct(m), lo=float("nan"), hi=float("nan"))
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(n=len(v), mean_log=m, pct=pct(m), lo=pct(m - h), hi=pct(m + h))


def cluster(lr):
  lr = np.asarray(lr, dtype=float)
  sums = np.array([lr[[i for i, t in enumerate(TASKS) if DIR[t] == d]].sum() for d in DIRS])
  counts = np.array([sum(DIR[t] == d for t in TASKS) for d in DIRS], dtype=float)
  idx = np.random.default_rng(0).integers(0, len(DIRS), size=(20000, len(DIRS)))
  means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
  return dict(n=len(lr), pct=pct(lr.mean()), lo=pct(np.percentile(means, 2.5)), hi=pct(np.percentile(means, 97.5)),
    tasks_won=int((lr < 0).sum()), dirs_won=int(sum(x < 0 for x in sums / counts)))


def fmt(r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


print("complete (arm, seed):", [(a, s) for a in ARMS for s in SEEDS if complete(a, s)])
for name, a, b in PAIRS:
  seeds = [s for s in SEEDS if complete(a, s) and complete(b, s)]
  new = [s for s in seeds if s != 148021]
  REPORT[name] = dict(a=a, b=b, seeds=seeds)
  for metric in ("act", "flow", "rob"):
    per_seed = [ratios(a, b, s, metric) for s in seeds]
    means = [float(v.mean()) for v in per_seed]
    entry = dict(per_seed_pct=[pct(m) for m in means], all=tci(means), new=tci([float(ratios(a, b, s, metric).mean())
      for s in new]) if new else None, view=cluster(np.mean(per_seed, axis=0)),
      won=int(sum((v < 0).sum() for v in per_seed)), folds=41 * len(seeds),
      per_task=dict((str(t), float(np.mean([v[i] for v in per_seed]))) for i, t in enumerate(TASKS)))
    REPORT[name][metric] = entry
    line = "%-11s %-4s seeds %d  t: %s" % (name, metric, len(seeds), fmt(entry["all"]) if len(seeds) > 1 else "-")
    if new and len(new) > 1:
      line += "  | 4 new: %s" % fmt(entry["new"])
    line += "  | dirs: %s won %d/41 tasks, %d/16 dirs" % (fmt(entry["view"]), entry["view"]["tasks_won"],
      entry["view"]["dirs_won"])
    print(line)
    print("            per seed:", " ".join("%+.1f" % x for x in entry["per_seed_pct"]))

print("\ncontext (seed-mean over complete seeds; bootstrap over directions)")
CTX = dict()
for arm in ("lodo_plain", "lodo_sat", "lodo_xy6", "lodo_txy6"):
  seeds = [s for s in SEEDS if complete(arm, s)]
  for key, metric, src in (("vs_loto_film_nl_act", "act", "zero"), ("vs_seen_act", "act", "seen"),
      ("vs_seen_flow", "flow", "seen"), ("vs_loto_film_nl_flow", "flow", "film_nl"),
      ("vs_loto_plain_nl_flow", "flow", "plain_nl")):
    per = []
    for s in seeds:
      if src in ("film_nl", "plain_nl"):
        rows = read(RAW / "n1n2" / ("blind_nl_%s_seed%d" % (src, s)) / "folds.jsonl")
        ref = dict((int(r["held_task"]), float(r["main_selection"])) for r in rows)
      elif src == "zero":
        ref = dict((int(r["task"]), float(r["action_selection"])) for r in read(RAW / "ideal" / ("zero_seed%d.jsonl" % s)))
      else:
        k = "action_selection" if metric == "act" else "flow_selection"
        ref = dict((int(r["task"]), float(r[k])) for r in read(RAW / "ideal" / ("seen_seed%d.jsonl" % s)))
      per.append([math.log(float(DATA[(arm, s)][t][METRIC[metric]]) / ref[t]) for t in TASKS])
    r = cluster(np.mean(per, axis=0))
    r["seeds"] = seeds
    CTX["%s_%s" % (arm, key)] = r
    print("  %-10s %-20s %s (seeds %d)" % (arm, key, fmt(r), len(seeds)))
REPORT["context"] = CTX
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
