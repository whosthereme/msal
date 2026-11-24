import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = (148021, 148022, 148023, 148024, 148025)
BASE_ARM = "lodo_xy6"
ID_ARMS = ("lodo_xy6id05", "lodo_xy6id1", "lodo_xy6id2")
REC = {}
for p in sys.argv[2:]:
  for line in Path(p).read_text().splitlines():
    if line.strip():
      r = json.loads(line)
      if r.get("train_steps") != 12800:
        continue
      k = (int(r["seed"]), r["arm"], int(r["held_task"]))
      assert k not in REC, k
      REC[k] = r
TASKS = sorted(set(k[2] for k in REC))
missing = [(s, a, t) for s in SEEDS for a in (BASE_ARM,) + ID_ARMS for t in TASKS if (s, a, t) not in REC]
print("records", len(REC), "tasks", len(TASKS), "missing", len(missing))
assert not missing and len(TASKS) == 41
for s in SEEDS:
  for t in TASKS:
    assert len(set(REC[(s, a, t)]["lane"] for a in (BASE_ARM,) + ID_ARMS)) == 1, (s, t)
DIR = dict((t, int(REC[(SEEDS[0], BASE_ARM, t)]["held_direction"])) for t in TASKS)
pct = lambda v: 100.0 * (math.exp(v) - 1.0)
REPORT = dict(machines={})
for s in SEEDS:
  for t in TASKS:
    m = REC[(s, BASE_ARM, t)]["lane"][:3]
    REPORT["machines"][m] = REPORT["machines"].get(m, 0) + 1


def reading(lo, hi):
  return "confirmed (higher)" if lo > 0 else ("contradicted (lower)" if hi < 0 else "not distinguishable")


def compare(arm, field):
  lr = dict((s, np.array([math.log(REC[(s, arm, t)][field] / REC[(s, BASE_ARM, t)][field]) for t in TASKS])) for s in SEEDS)
  means = np.array([lr[s].mean() for s in SEEDS])
  h = stats.t.ppf(0.975, len(SEEDS) - 1) * means.std(ddof=1) / math.sqrt(len(SEEDS))
  per_task = np.mean([lr[s] for s in SEEDS], axis=0)
  dirs = sorted(set(DIR.values()))
  per_dir = np.array([per_task[[i for i, t in enumerate(TASKS) if DIR[t] == d]].mean() for d in dirs])
  b = per_dir[np.random.default_rng(0).integers(0, len(dirs), size=(20000, len(dirs)))].mean(axis=1)
  out = dict(pct=pct(means.mean()), lo=pct(means.mean() - h), hi=pct(means.mean() + h), per_seed=[pct(x) for x in means],
    mean_log=float(means.mean()), dir_view=dict(lo=pct(np.percentile(b, 2.5)), hi=pct(np.percentile(b, 97.5)),
    higher_dirs=int((per_dir > 0).sum())), tasks_higher=int((per_task > 0).sum()), per_task=per_task.tolist())
  out["reading"] = reading(out["lo"], out["hi"])
  return out


for field in ("action_selection", "flow_selection", "flow_robustness"):
  REPORT[field] = {}
  print("==", field)
  for arm in ID_ARMS:
    r = compare(arm, field)
    REPORT[field][arm] = r
    print("  %-13s vs xy6  %+6.1f%% [%+6.1f, %+6.1f]  %-22s per seed %s | directions %+.1f..%+.1f, higher %d/16 | tasks higher %d/41"
      % (arm, r["pct"], r["lo"], r["hi"], r["reading"], " ".join("%+.1f" % x for x in r["per_seed"]),
         r["dir_view"]["lo"], r["dir_view"]["hi"], r["dir_view"]["higher_dirs"], r["tasks_higher"]))
  order = [0.0] + [REPORT[field][a]["mean_log"] for a in ID_ARMS]
  REPORT[field]["monotone"] = bool(all(order[i] <= order[i + 1] for i in range(3)))
  print("  monotone in alpha (0 <= 0.5 <= 1 <= 2):", REPORT[field]["monotone"])
iota = np.array([REC[(SEEDS[0], a, t)]["iota"] for a in ID_ARMS for t in TASKS])
y = np.array([v for a in ID_ARMS for v in REPORT["action_selection"][a]["per_task"]])
rho = stats.spearmanr(iota, y)[0]
REPORT["pooled_spearman_iota_action"] = float(rho)
print("  pooled per-task Spearman(iota, log ratio), three identifier arms: %+.3f" % rho)
REPORT["B1"] = REPORT["action_selection"]["lodo_xy6id2"]["reading"]
print("  B1 (primary):", REPORT["B1"])
print("  pairs per machine:", REPORT["machines"])
Path(sys.argv[1]).write_text(json.dumps(REPORT, indent=1))
print("written", sys.argv[1])
