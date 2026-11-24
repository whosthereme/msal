import json
import math
import sys
from pathlib import Path

import numpy as np

RES, RAW = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (148021, 148022, 148023, 148024, 148025)
NEW = SEEDS[1:]
NODE = {148022: "ws3", 148023: "ws3", 148024: "ws4", 148025: "ws4"}
OUTLIERS = (1, 2, 31, 43)
T975 = {3: 3.182446305284263, 4: 2.7764451051977987, 11: 2.200985160082949, 23: 2.0686576104190406}
METRICS = (("flow", "flow_selection"), ("act", "action_selection"), ("rob", "flow_robustness"))


def read(path):
  if not Path(path).exists():
    return []
  return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def table(path, key, task_key="held_task"):
  return dict((int(r[task_key]), float(r[key])) for r in read(path))


def pct(v):
  return 100 * (math.exp(v) - 1)


STORED = dict()
for s in SEEDS:
  n1 = RAW / "n1n2"
  d = dict()
  d["flow"] = table(n1 / ("blind_nl_film_nl_seed%d" % s) / "folds.jsonl", "main_selection")
  d["rob"] = table(n1 / ("blind_nl_film_nl_seed%d" % s) / "folds.jsonl", "main_robustness")
  d["act"] = table(RAW / "ideal" / ("zero_seed%d.jsonl" % s), "action_selection", "task")
  d["plain_flow"] = table(n1 / ("blind_nl_plain_nl_seed%d" % s) / "folds.jsonl", "main_selection")
  d["seen_flow"] = table(RAW / "ideal" / ("seen_seed%d.jsonl" % s), "flow_selection", "task")
  d["seen_act"] = table(RAW / "ideal" / ("seen_seed%d.jsonl" % s), "action_selection", "task")
  STORED[s] = d
TASKS = sorted(STORED[SEEDS[0]]["flow"])
assert len(TASKS) == 41

print("== calibration: film_nl retrained on ws3 / ws4 vs stored ws2 score (log ratio; t interval over folds)")
CAL = dict(ws3=dict(flow=[], act=[]), ws4=dict(flow=[], act=[]))
NODE_FILM = dict()
for s in NEW:
  node = NODE[s]
  recs = read(RES / node / ("film_nl_seed%d.jsonl" % s))
  NODE_FILM[s] = dict((m, dict((int(r["held_task"]), float(r[k])) for r in recs)) for m, k in METRICS)
  for r in recs:
    t = int(r["held_task"])
    CAL[node]["flow"].append(math.log(r["flow_selection"] / STORED[s]["flow"][t]))
    CAL[node]["act"].append(math.log(r["action_selection"] / STORED[s]["act"][t]))
USE_STORED = dict()
calibration = dict()
for node in ("ws3", "ws4"):
  ok = True
  calibration[node] = dict()
  for m in ("flow", "act"):
    v = np.array(CAL[node][m])
    n = len(v)
    if n < 2:
      print("  %s %-4s: only %d calibration folds" % (node, m, n))
      ok = False
      continue
    half = T975.get(n - 1, 2.2) * v.std(ddof=1) / math.sqrt(n)
    lo, hi = pct(v.mean() - half), pct(v.mean() + half)
    inside = lo > -2.0 and hi < 2.0
    ok = ok and inside
    calibration[node][m] = dict(folds=n, mean_pct=pct(v.mean()), lo=lo, hi=hi, max_abs_pct=pct(np.abs(v).max()),
      inside=inside)
    print("  %s %-4s: %d folds, mean %+.2f%% [%+.2f, %+.2f], largest single fold %.1f%%, bit-identical %d -> %s" % (
      node, m, n, pct(v.mean()), lo, hi, pct(np.abs(v).max()), int((v == 0).sum()),
      "inside +/-2%" if inside else "OUTSIDE +/-2%"))
  USE_STORED[node] = ok
  print("  %s: ws3/ws4 folds of this machine are compared with %s" % (node,
    "the stored film_nl scores" if ok else "film_nl retrained on this machine (must exist for every fold)"))

TF = dict((s, dict(flow=dict(), act=dict(), rob=dict())) for s in SEEDS)
REF = dict((s, dict(flow=dict(), act=dict(), rob=dict())) for s in SEEDS)
SRC = dict((s, dict()) for s in SEEDS)
missing = []


def reference(s, m, t, where):
  if where == "ws2" or USE_STORED[where]:
    return STORED[s][m][t]
  return NODE_FILM[s][m].get(t)


def add_fold(s, r, where):
  t = int(r["held_task"])
  assert t not in SRC[s], "fold %d of seed %d exists on two machines" % (t, s)
  SRC[s][t] = where
  for m, k in METRICS:
    TF[s][m][t] = float(r[k])
    ref = reference(s, m, t, where)
    if ref is None:
      missing.append((s, t))
    else:
      REF[s][m][t] = ref


for s in SEEDS:
  for r in read(RES / ("tfilm_nl_seed%d.jsonl" % s)):
    add_fold(s, r, "ws2")
  node = NODE.get(s)
  for r in (read(RES / node / ("tfilm_nl_seed%d.jsonl" % s)) if node else []):
    add_fold(s, r, node)
have = dict((s, sorted(SRC[s])) for s in SEEDS)
print("\nfolds per seed: " + ", ".join("%d: %d (ws2 %d)" % (s, len(have[s]), sum(1 for t in have[s] if SRC[s][t] == "ws2"))
  for s in SEEDS))
if missing:
  print("MISSING film_nl reference on the node for %d folds: analysis of those folds is not possible yet" % len(set(missing)))
complete = all(len(have[s]) == 41 for s in SEEDS) and not missing


def ratios(metric, seeds, where=None, tasks=None, ref=None):
  out = []
  for s in seeds:
    row = []
    for t in have[s]:
      if tasks is not None and t not in tasks:
        continue
      if where is not None and (SRC[s][t] == "ws2") != (where == "ws2"):
        continue
      base = REF[s][metric][t] if ref is None else STORED[s][ref][t]
      if metric in TF[s] and t in TF[s][metric] and (ref is not None or t in REF[s][metric]):
        row.append(math.log(TF[s][metric][t] / base))
    out.append(np.array(row))
  return out


def seed_reading(rows):
  per_seed = np.array([r.mean() for r in rows])
  n = len(per_seed)
  half = T975[n - 1] * per_seed.std(ddof=1) / math.sqrt(n)
  lo, hi = per_seed.mean() - half, per_seed.mean() + half
  reading = "better" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
  won = int(sum((r < 0).sum() for r in rows))
  return dict(pct=pct(per_seed.mean()), lo=pct(lo), hi=pct(hi), reading=reading, df=n - 1,
    per_seed_pct=[pct(v) for v in per_seed], folds_won=won, folds=int(sum(len(r) for r in rows)))


def line(title, c):
  print("  %-42s %+7.1f%% [%+7.1f, %+7.1f]  %-19s folds won %d/%d  per seed: %s" % (title, c["pct"], c["lo"], c["hi"],
    c["reading"], c["folds_won"], c["folds"], " ".join("%+.1f" % v for v in c["per_seed_pct"])))


result = dict(complete=complete, calibration=calibration, use_stored_reference=USE_STORED)
tag = "" if complete else "  [INCOMPLETE: not all 205 folds present]"
print("\n== PRIMARY, five seeds (t interval over seeds, df 4): tfilm_nl vs film_nl" + tag)
result["P1_action_5"] = seed_reading(ratios("act", SEEDS))
result["P2_flow_5"] = seed_reading(ratios("flow", SEEDS))
line("P1 sampled-action error", result["P1_action_5"])
line("P2 flow MSE", result["P2_flow_5"])
print("\n== CONFIRMATION, seeds 148022-148025 only (df 3)" + tag)
result["P1_action_4"] = seed_reading(ratios("act", NEW))
result["P2_flow_4"] = seed_reading(ratios("flow", NEW))
line("P1 sampled-action error", result["P1_action_4"])
line("P2 flow MSE", result["P2_flow_4"])

print("\n== SECONDARY, five seeds")
result["flow_robustness"] = seed_reading(ratios("rob", SEEDS))
line("flow MSE, ROBUSTNESS rows", result["flow_robustness"])
result["tfilm_vs_plain_flow"] = seed_reading(ratios("flow", SEEDS, ref="plain_flow"))
result["tfilm_vs_seen_action"] = seed_reading(ratios("act", SEEDS, ref="seen_act"))
result["tfilm_vs_seen_flow"] = seed_reading(ratios("flow", SEEDS, ref="seen_flow"))
line("tfilm_nl vs plain_nl, flow MSE", result["tfilm_vs_plain_flow"])
line("tfilm_nl vs trained head, action error", result["tfilm_vs_seen_action"])
line("tfilm_nl vs trained head, flow MSE", result["tfilm_vs_seen_flow"])


def stored_reading(a, b):
  rows = [np.array([math.log(STORED[s][a][t] / STORED[s][b][t]) for t in TASKS]) for s in SEEDS]
  return seed_reading(rows)


line("film_nl vs plain_nl, flow (paper -19.9)", stored_reading("flow", "plain_flow"))
line("film_nl vs trained head, action (paper +430)", stored_reading("act", "seen_act"))
line("film_nl vs trained head, flow (paper +214)", stored_reading("flow", "seen_flow"))
if complete:
  gaps = []
  for s in SEEDS:
    b, a, top = (np.mean([d[t] for t in TASKS]) for d in (STORED[s]["plain_flow"], TF[s]["flow"], STORED[s]["seen_flow"]))
    f = np.mean([STORED[s]["flow"][t] for t in TASKS])
    gaps.append(((b - a) / (b - top), (b - f) / (b - top)))
  result["gap_closed"] = dict(tfilm_nl=[g[0] for g in gaps], film_nl=[g[1] for g in gaps])
  print("  share of the flow gap plain_nl -> trained head closed, per seed: tfilm_nl %s | film_nl %s" % (
    " ".join("%.2f" % g[0] for g in gaps), " ".join("%.2f" % g[1] for g in gaps)))
  print("  means over 205 folds: tfilm_nl flow %.4f action %.4f | film_nl flow %.4f action %.4f" % (
    np.mean([TF[s]["flow"][t] for s in SEEDS for t in TASKS]), np.mean([TF[s]["act"][t] for s in SEEDS for t in TASKS]),
    np.mean([STORED[s]["flow"][t] for s in SEEDS for t in TASKS]), np.mean([STORED[s]["act"][t] for s in SEEDS for t in TASKS])))

print("\n== SECOND VIEW: bootstrap over the 41 tasks (per-task mean over the seeds that have the task)")
for name, metric in (("P1 sampled-action error", "act"), ("P2 flow MSE", "flow")):
  per_task = []
  for t in TASKS:
    vals = [math.log(TF[s][metric][t] / REF[s][metric][t]) for s in SEEDS if t in TF[s][metric] and t in REF[s][metric]]
    if vals:
      per_task.append(np.mean(vals))
  per_task = np.array(per_task)
  rng = np.random.default_rng(0)
  boot = np.exp(rng.choice(per_task, size=(20000, len(per_task)), replace=True).mean(axis=1)) - 1
  lo, hi = 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)
  reading = "better" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
  result["tasks_" + metric] = dict(pct=pct(per_task.mean()), lo=lo, hi=hi, reading=reading,
    tasks_won=int((per_task < 0).sum()), tasks=len(per_task))
  print("  %-42s %+7.1f%% [%+7.1f, %+7.1f]  %-19s tasks won %d/%d" % (name, pct(per_task.mean()), lo, hi, reading,
    int((per_task < 0).sum()), len(per_task)))

print("\n== CHECK by machine, seeds 148022-148025 (the machines also differ in tasks: ws2 = first 21, ws3/ws4 = last 20)")
first21 = set(t for t in TASKS if SRC[NEW[0]].get(t) == "ws2")
for name, metric in (("sampled-action error", "act"), ("flow MSE", "flow")):
  for where in ("ws2", "nodes"):
    rows = ratios(metric, NEW, where=where)
    if all(len(r) for r in rows):
      line("%s, %s folds" % (name, "ws2" if where == "ws2" else "ws3/ws4"), seed_reading(rows))
  seed1 = [np.array([math.log(TF[148021][metric][t] / REF[148021][metric][t]) for t in TASKS if (t in first21) == flag])
    for flag in (True, False)]
  print("    seed 148021 (all on ws2) on the same two task groups: first 21 %+.1f%%, last 20 %+.1f%%" % (
    pct(seed1[0].mean()), pct(seed1[1].mean())))

print("\n== DESCRIPTIVE: without the four outlier tasks (37 tasks), five seeds")
rest = set(t for t in TASKS if t not in OUTLIERS)
result["P1_action_5_no_outliers"] = seed_reading(ratios("act", SEEDS, tasks=rest))
result["P2_flow_5_no_outliers"] = seed_reading(ratios("flow", SEEDS, tasks=rest))
line("P1 sampled-action error", result["P1_action_5_no_outliers"])
line("P2 flow MSE", result["P2_flow_5_no_outliers"])
if len(sys.argv) > 3:
  Path(sys.argv[3]).write_text(json.dumps(result, indent=1))
