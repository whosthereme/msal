import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RES, RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEEDS = (148021, 148022, 148023, 148024, 148025)
WS2 = (0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 27)
FAMILY = {148022: "ws34", 148025: "ws34", 148023: "local", 148024: "local"}
METRIC = dict(act="action_selection", flow="flow_selection")
LANES = ("out_q_ws3a", "out_q_ws3b", "out_q_ws4a", "out_q_ws4b", "out_ws3", "out_ws4", "out_ws4adup")


def read(path):
  path = Path(path)
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def table(path, key, task_key="held_task"):
  return dict((int(r[task_key]), float(r[key])) for r in read(path))


STORED = {}
for s in SEEDS:
  STORED[s] = dict(flow=table(RAW / "n1n2" / ("blind_nl_film_nl_seed%d" % s) / "folds.jsonl", "main_selection"),
    act=table(RAW / "ideal" / ("zero_seed%d.jsonl" % s), "action_selection", "task"))
TASKS = sorted(STORED[148021]["flow"])
REST = [t for t in TASKS if t not in WS2]
assert len(TASKS) == 41

TF, TF_RAW, TF_SRC = {}, {}, {}
for s in SEEDS:
  for r in read(RES / ("tfilm_nl_seed%d.jsonl" % s)):
    t = int(r["held_task"])
    for m, k in METRIC.items():
      TF[(s, t, m)] = math.log(float(r[k]) / STORED[s][m][t])
      TF_RAW[(s, t, m)] = float(r[k])
    TF_SRC[(s, t)] = "ws2"
for s in SEEDS[1:]:
  def paths(arm):
    if FAMILY[s] == "ws34":
      return [RES / "final_nodes" / lane / ("%s_seed%d.jsonl" % (arm, s)) for lane in LANES]
    return [RES / "local" / "out" / ("%s_seed%d.jsonl" % (arm, s))]
  got = {}
  for arm in ("tfilm_nl", "film_nl"):
    got[arm] = {}
    for p in paths(arm):
      for r in read(p):
        got[arm].setdefault(int(r["held_task"]), r)
  for t in REST:
    for m, k in METRIC.items():
      TF[(s, t, m)] = math.log(float(got["tfilm_nl"][t][k]) / float(got["film_nl"][t][k]))
    TF_SRC[(s, t)] = FAMILY[s]
assert all((s, t, "act") in TF for s in SEEDS for t in TASKS)

REC = {}
for p in sorted((RES / "heads5" / "full").glob("folds_*.jsonl")):
  for r in read(p):
    if r.get("train_steps") == 12800:
      REC.setdefault((int(r["seed"]), r["arm"], int(r["held_task"])), r)


def machine(s, t):
  return REC[(s, "plain_nl", t)]["lane"][:3]


TI, TI_RAW = {}, {}
for s in SEEDS:
  for t in TASKS:
    for m, k in METRIC.items():
      film = STORED[s][m][t] if machine(s, t) == "ws2" else float(REC[(s, "film_nl", t)][k])
      TI[(s, t, m)] = math.log(float(REC[(s, "tin_nl", t)][k]) / film)
      TI_RAW[(s, t, m)] = float(REC[(s, "tin_nl", t)][k])


def summarize(lr):
  means = np.array([lr[s].mean() for s in SEEDS])
  h = stats.t.ppf(0.975, len(means) - 1) * means.std(ddof=1) / math.sqrt(len(means))
  per_task = np.mean([lr[s] for s in SEEDS], axis=0)
  b = per_task[np.random.default_rng(0).integers(0, len(per_task), size=(20000, len(per_task)))].mean(axis=1)
  return dict(pct=pct(means.mean()), lo=pct(means.mean() - h), hi=pct(means.mean() + h),
    per_seed=[round(pct(x), 1) for x in means], folds_lower=int(sum((lr[s] < 0).sum() for s in SEEDS)),
    folds=int(sum(len(lr[s]) for s in SEEDS)), tasks_lower=int((per_task < 0).sum()),
    task_boot=[pct(np.percentile(b, 2.5)), pct(np.percentile(b, 97.5))])


REPORT = dict(note="post hoc, stored records only", indirect={}, direct_ws2={}, one_process_seed148021={})
print("trunk modulation (tfilm_nl) against input concatenation (tin_nl), zero-shot, development family; post hoc")
for m in METRIC:
  for name, d in (("tfilm vs film_nl", TF), ("tin vs film_nl", TI)):
    r = summarize(dict((s, np.array([d[(s, t, m)] for t in TASKS])) for s in SEEDS))
    print("  check  %-18s %-5s %+6.1f%% [%+6.1f, %+6.1f] per seed %s" % (name, m, r["pct"], r["lo"], r["hi"], r["per_seed"]))
  r = summarize(dict((s, np.array([TF[(s, t, m)] - TI[(s, t, m)] for t in TASKS])) for s in SEEDS))
  REPORT["indirect"][m] = r
  print("  INDIRECT tfilm vs tin %-5s %+6.1f%% [%+6.1f, %+6.1f]  folds lower %d/%d, tasks lower %d/41, task bootstrap "
        "%+.1f..%+.1f, per seed %s" % (m, r["pct"], r["lo"], r["hi"], r["folds_lower"], r["folds"], r["tasks_lower"],
        r["task_boot"][0], r["task_boot"][1], r["per_seed"]))
  both = [(s, t) for s in SEEDS for t in TASKS if TF_SRC[(s, t)] == "ws2" and machine(s, t) == "ws2"]
  v = np.array([math.log(TF_RAW[(s, t, m)] / TI_RAW[(s, t, m)]) for s, t in both])
  idx = np.random.default_rng(0).integers(0, len(v), size=(20000, len(v)))
  bm = v[idx].mean(axis=1)
  REPORT["direct_ws2"][m] = dict(n=len(v), pct=pct(v.mean()), lo=pct(np.percentile(bm, 2.5)), hi=pct(np.percentile(bm, 97.5)),
    lower=int((v < 0).sum()), seeds=sorted(set(s for s, _ in both)))
  print("  DIRECT, both on ws2   %-5s %+6.1f%% [%+6.1f, %+6.1f] (bootstrap over %d folds; lower on %d)" % (m, pct(v.mean()),
        pct(np.percentile(bm, 2.5)), pct(np.percentile(bm, 97.5)), len(v), int((v < 0).sum())))
  a = table(RES / "tfilm_nl_seed148021.jsonl", METRIC[m])
  b = table(RES / "tin_nl_seed148021.jsonl", METRIC[m])
  v = np.array([math.log(a[t] / b[t]) for t in TASKS])
  REPORT["one_process_seed148021"][m] = dict(pct=pct(v.mean()), lower=int((v < 0).sum()))
  print("  seed 148021, one study %-5s %+6.1f%% (lower on %d/41)" % (m, pct(v.mean()), int((v < 0).sum())))
OUT.write_text(json.dumps(REPORT, indent=1))
