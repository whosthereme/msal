import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1])
RAW = Path(sys.argv[2])
SEED = 148021
ARMS = ("tin_nl", "tfilm_nl", "tboth_nl")
OUTLIERS = (1, 2, 31, 43)
LABEL = dict(tin_nl="description into trunk input", tfilm_nl="FiLM inside trunk", tboth_nl="FiLM in trunk + MSAL head")


def read(path):
  return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def by_task(records, key, task_key="held_task"):
  return dict((int(r[task_key]), float(r[key])) for r in records)


film_flow = by_task(read(RAW / "n1n2" / ("blind_nl_film_nl_seed%d" % SEED) / "folds.jsonl"), "main_selection")
film_rob = by_task(read(RAW / "n1n2" / ("blind_nl_film_nl_seed%d" % SEED) / "folds.jsonl"), "main_robustness")
plain_flow = by_task(read(RAW / "n1n2" / ("blind_nl_plain_nl_seed%d" % SEED) / "folds.jsonl"), "main_selection")
zero = read(RAW / "ideal" / ("zero_seed%d.jsonl" % SEED))
seen = read(RAW / "ideal" / ("seen_seed%d.jsonl" % SEED))
film_act = by_task(zero, "action_selection", "task")
seen_flow = by_task(seen, "flow_selection", "task")
seen_act = by_task(seen, "action_selection", "task")
TASKS = sorted(film_flow)
assert len(TASKS) == 41


def compare(a, b, tasks):
  lr = np.array([math.log(a[t] / b[t]) for t in tasks])
  rng = np.random.default_rng(0)
  boot = np.exp(rng.choice(lr, size=(20000, len(lr)), replace=True).mean(axis=1)) - 1
  lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
  reading = "better" if hi < 0 else ("worse" if lo > 0 else "not distinguishable")
  return dict(pct=100 * (math.exp(lr.mean()) - 1), lo=100 * lo, hi=100 * hi, won=int((lr < 0).sum()), n=len(lr),
    reading=reading)


def show(title, c):
  print("  %-34s %+6.1f%% [%+6.1f, %+6.1f]  folds won %2d/%d  %s" % (
    title, c["pct"], c["lo"], c["hi"], c["won"], c["n"], c["reading"]))


def closure(arm_scores, base, top, tasks):
  b, a, s = (np.mean([d[t] for t in tasks]) for d in (base, arm_scores, top))
  return float((b - a) / (b - s))


result = dict(seed=SEED, arms=dict())
anchor = read(OUT / ("film_nl_seed%d.jsonl" % SEED))
print("anchor film_nl: " + ", ".join("task %d flow %s action %s" % (
  r["held_task"], r["anchor_flow_bit_exact"], r["anchor_action_bit_exact"]) for r in anchor))
print("references (arithmetic mean over 41 tasks, seed %d): film_nl flow %.4f action %.4f | plain_nl flow %.4f |"
  " trained head flow %.4f action %.4f" % (SEED, np.mean(list(film_flow.values())), np.mean(list(film_act.values())),
  np.mean(list(plain_flow.values())), np.mean(list(seen_flow.values())), np.mean(list(seen_act.values()))))
print("film_nl closes %.2f of the flow gap plain_nl -> trained head at this seed"
  % closure(film_flow, plain_flow, seen_flow, TASKS))

for arm in ARMS:
  path = OUT / ("%s_seed%d.jsonl" % (arm, SEED))
  if not path.exists():
    print("\n== %s: no results" % arm)
    continue
  recs = read(path)
  flow, rob, act = by_task(recs, "flow_selection"), by_task(recs, "flow_robustness"), by_task(recs, "action_selection")
  tasks = [t for t in TASKS if t in flow]
  bad = [t for t in tasks if not (math.isfinite(flow[t]) and math.isfinite(act[t]))]
  print("\n== %s (%s): %d/41 folds, non-finite %s, task-conditioned weights %d, mean train time %.0f s" % (
    arm, LABEL[arm], len(tasks), bad, recs[0]["task_conditioned_parameters"],
    np.mean([r["train_seconds"] for r in recs])))
  print("  mean flow MSE %.4f | mean action error %.4f" % (np.mean([flow[t] for t in tasks]),
    np.mean([act[t] for t in tasks])))
  entry = dict(folds=len(tasks))
  entry["flow_vs_film_nl"] = compare(flow, film_flow, tasks)
  entry["action_vs_film_nl"] = compare(act, film_act, tasks)
  entry["robustness_vs_film_nl"] = compare(rob, film_rob, tasks)
  entry["flow_vs_plain_nl"] = compare(flow, plain_flow, tasks)
  entry["action_vs_trained_head"] = compare(act, seen_act, tasks)
  rest = [t for t in tasks if t not in OUTLIERS]
  entry["flow_vs_film_nl_without_outliers"] = compare(flow, film_flow, rest)
  entry["action_vs_film_nl_without_outliers"] = compare(act, film_act, rest)
  entry["gap_closed_flow"] = closure(flow, plain_flow, seen_flow, tasks)
  entry["mean_flow"] = float(np.mean([flow[t] for t in tasks]))
  entry["mean_action"] = float(np.mean([act[t] for t in tasks]))
  show("flow MSE vs film_nl (primary)", entry["flow_vs_film_nl"])
  show("action error vs film_nl (primary)", entry["action_vs_film_nl"])
  show("flow MSE, ROBUSTNESS rows", entry["robustness_vs_film_nl"])
  show("flow MSE vs plain_nl", entry["flow_vs_plain_nl"])
  show("action error vs trained head", entry["action_vs_trained_head"])
  show("flow vs film_nl, 37 tasks", entry["flow_vs_film_nl_without_outliers"])
  show("action vs film_nl, 37 tasks", entry["action_vs_film_nl_without_outliers"])
  print("  share of the flow gap plain_nl -> trained head closed: %.2f" % entry["gap_closed_flow"])
  print("  outlier tasks, flow MSE arm / film_nl: " + ", ".join(
    "%d: %.3f / %.3f" % (t, flow[t], film_flow[t]) for t in OUTLIERS if t in flow))
  result["arms"][arm] = entry

print("\nfilm_nl action error vs trained head at this seed:")
show("", compare(film_act, seen_act, TASKS))
if len(sys.argv) > 3:
  Path(sys.argv[3]).write_text(json.dumps(result, indent=1))
