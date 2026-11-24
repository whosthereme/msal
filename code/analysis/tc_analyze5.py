import json
import math
import sys
from pathlib import Path

import numpy as np

OUT, RAW = Path(sys.argv[1]), Path(sys.argv[2])
SEEDS = (148021, 148022, 148023, 148024, 148025)
NEW = SEEDS[1:]
OUTLIERS = (1, 2, 31, 43)
T975 = {3: 3.182446305284263, 4: 2.7764451051977987}


def read(path):
  return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def table(path, key, task_key="held_task"):
  return dict((int(r[task_key]), float(r[key])) for r in read(path))


def stored(seed):
  n1 = RAW / "n1n2"
  ideal = RAW / "ideal"
  out = dict()
  out["film_flow"] = table(n1 / ("blind_nl_film_nl_seed%d" % seed) / "folds.jsonl", "main_selection")
  out["film_rob"] = table(n1 / ("blind_nl_film_nl_seed%d" % seed) / "folds.jsonl", "main_robustness")
  out["plain_flow"] = table(n1 / ("blind_nl_plain_nl_seed%d" % seed) / "folds.jsonl", "main_selection")
  out["film_act"] = table(ideal / ("zero_seed%d.jsonl" % seed), "action_selection", "task")
  out["seen_flow"] = table(ideal / ("seen_seed%d.jsonl" % seed), "flow_selection", "task")
  out["seen_act"] = table(ideal / ("seen_seed%d.jsonl" % seed), "action_selection", "task")
  return out


def new(seed):
  path = OUT / ("tfilm_nl_seed%d.jsonl" % seed)
  out = dict()
  out["tf_flow"] = table(path, "flow_selection")
  out["tf_rob"] = table(path, "flow_robustness")
  out["tf_act"] = table(path, "action_selection")
  return out


DATA = dict()
for s in SEEDS:
  DATA[s] = stored(s)
  DATA[s].update(new(s))
TASKS = sorted(DATA[SEEDS[0]]["film_flow"])
assert len(TASKS) == 41
for s in SEEDS:
  for key, values in DATA[s].items():
    assert sorted(values) == TASKS, (s, key, len(values))
    assert all(math.isfinite(v) and v > 0 for v in values.values()), (s, key)


def log_ratios(a, b, seeds, tasks):
  return np.array([[math.log(DATA[s][a][t] / DATA[s][b][t]) for t in tasks] for s in seeds])


def seed_reading(a, b, seeds, tasks=TASKS):
  lr = log_ratios(a, b, seeds, tasks)
  per_seed = lr.mean(axis=1)
  m, sd, n = per_seed.mean(), per_seed.std(ddof=1), len(seeds)
  half = T975[n - 1] * sd / math.sqrt(n)
  lo, hi = m - half, m + half
  reading = "better" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
  pct = lambda v: 100 * (math.exp(v) - 1)
  return dict(pct=pct(m), lo=pct(lo), hi=pct(hi), reading=reading, seeds=n, df=n - 1,
    per_seed_pct=[pct(v) for v in per_seed], folds_won=int((lr < 0).sum()), folds=int(lr.size),
    won_per_seed=[int(v) for v in (lr < 0).sum(axis=1)])


def task_reading(a, b, seeds, tasks=TASKS):
  per_task = log_ratios(a, b, seeds, tasks).mean(axis=0)
  rng = np.random.default_rng(0)
  boot = np.exp(rng.choice(per_task, size=(20000, len(per_task)), replace=True).mean(axis=1)) - 1
  lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
  reading = "better" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
  return dict(pct=100 * (math.exp(per_task.mean()) - 1), lo=100 * lo, hi=100 * hi, reading=reading,
    tasks_won=int((per_task < 0).sum()), tasks=len(per_task))


def line(title, c):
  extra = ""
  if "per_seed_pct" in c:
    extra = "  folds won %d/%d  per seed: %s" % (c["folds_won"], c["folds"],
      " ".join("%+.1f" % v for v in c["per_seed_pct"]))
  else:
    extra = "  tasks won %d/%d" % (c["tasks_won"], c["tasks"])
  print("  %-44s %+7.1f%% [%+7.1f, %+7.1f]  %-19s%s" % (title, c["pct"], c["lo"], c["hi"], c["reading"], extra))


def closure(key, seed):
  d = DATA[seed]
  b, a, top = (np.mean([d[k][t] for t in TASKS]) for k in ("plain_flow", key, "seen_flow"))
  return float((b - a) / (b - top))


result = dict()
print("means over 205 folds: film_nl flow %.4f action %.4f | tfilm_nl flow %.4f action %.4f | trained head flow %.4f"
  " action %.4f" % tuple(np.mean([np.mean(list(DATA[s][k].values())) for s in SEEDS])
  for k in ("film_flow", "film_act", "tf_flow", "tf_act", "seen_flow", "seen_act")))

print("\n== PRIMARY, five seeds (t interval over seeds, df 4): tfilm_nl vs stored film_nl")
result["P1_action_5"] = seed_reading("tf_act", "film_act", SEEDS)
result["P2_flow_5"] = seed_reading("tf_flow", "film_flow", SEEDS)
line("P1 sampled-action error", result["P1_action_5"])
line("P2 flow MSE", result["P2_flow_5"])

print("\n== CONFIRMATION, seeds 148022-148025 only (df 3)")
result["P1_action_4"] = seed_reading("tf_act", "film_act", NEW)
result["P2_flow_4"] = seed_reading("tf_flow", "film_flow", NEW)
line("P1 sampled-action error", result["P1_action_4"])
line("P2 flow MSE", result["P2_flow_4"])

print("\n== SECONDARY, five seeds")
result["flow_robustness"] = seed_reading("tf_rob", "film_rob", SEEDS)
result["tfilm_vs_plain_flow"] = seed_reading("tf_flow", "plain_flow", SEEDS)
result["film_vs_plain_flow"] = seed_reading("film_flow", "plain_flow", SEEDS)
result["tfilm_vs_seen_action"] = seed_reading("tf_act", "seen_act", SEEDS)
result["film_vs_seen_action"] = seed_reading("film_act", "seen_act", SEEDS)
result["tfilm_vs_seen_flow"] = seed_reading("tf_flow", "seen_flow", SEEDS)
result["film_vs_seen_flow"] = seed_reading("film_flow", "seen_flow", SEEDS)
line("flow MSE, ROBUSTNESS rows", result["flow_robustness"])
line("tfilm_nl vs plain_nl, flow MSE", result["tfilm_vs_plain_flow"])
line("film_nl vs plain_nl, flow MSE (paper -19.9)", result["film_vs_plain_flow"])
line("tfilm_nl vs trained head, action error", result["tfilm_vs_seen_action"])
line("film_nl vs trained head, action (paper +430)", result["film_vs_seen_action"])
line("tfilm_nl vs trained head, flow MSE", result["tfilm_vs_seen_flow"])
line("film_nl vs trained head, flow (paper +214)", result["film_vs_seen_flow"])
result["gap_closed"] = dict(tfilm_nl=[closure("tf_flow", s) for s in SEEDS], film_nl=[closure("film_flow", s) for s in SEEDS])
print("  share of the flow gap plain_nl -> trained head closed, per seed: tfilm_nl %s | film_nl %s" % (
  " ".join("%.2f" % v for v in result["gap_closed"]["tfilm_nl"]),
  " ".join("%.2f" % v for v in result["gap_closed"]["film_nl"])))

print("\n== SECOND VIEW: bootstrap over the 41 tasks (per-task mean over the five seeds)")
result["P1_action_tasks"] = task_reading("tf_act", "film_act", SEEDS)
result["P2_flow_tasks"] = task_reading("tf_flow", "film_flow", SEEDS)
line("P1 sampled-action error", result["P1_action_tasks"])
line("P2 flow MSE", result["P2_flow_tasks"])

print("\n== DESCRIPTIVE: without the four outlier tasks (37 tasks), five seeds")
rest = [t for t in TASKS if t not in OUTLIERS]
result["P1_action_5_no_outliers"] = seed_reading("tf_act", "film_act", SEEDS, rest)
result["P2_flow_5_no_outliers"] = seed_reading("tf_flow", "film_flow", SEEDS, rest)
line("P1 sampled-action error", result["P1_action_5_no_outliers"])
line("P2 flow MSE", result["P2_flow_5_no_outliers"])
print("  outlier tasks, flow MSE (mean over seeds) tfilm_nl / film_nl: " + ", ".join("%d: %.3f / %.3f" % (
  t, np.mean([DATA[s]["tf_flow"][t] for s in SEEDS]), np.mean([DATA[s]["film_flow"][t] for s in SEEDS]))
  for t in OUTLIERS))

anchors = [r for s in NEW for r in read(OUT / ("film_nl_seed%d.jsonl" % s))]
print("\nanchors (film_nl, held task 0): " + ", ".join("%d flow %s action %s" % (
  r["seed"], r["anchor_flow_bit_exact"], r["anchor_action_bit_exact"]) for r in anchors))
if len(sys.argv) > 3:
  Path(sys.argv[3]).write_text(json.dumps(result, indent=1))
