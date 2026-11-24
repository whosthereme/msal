import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

OUT, N1N2, IDEAL = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEEDS = (148021, 148022, 148023, 148024, 148025)
GROUP_A = ("film_nl", "plain_nl", "film", "plain", "tin_nl")
GROUP_B = ("sal_nlb", "sal_nl", "moe_desc", "sal", "ctx_add", "bilinear")
LABEL = dict(plain="No task input (linear)", ctx_add="Additive", film="FiLM", bilinear="Bilinear", sal="SAL",
  plain_nl="No task input (nonlinear)", moe_desc="Softmax mixture of experts", sal_nl="SAL, nonlinear atoms",
  sal_nlb="SAL, nonlinear base and atoms", film_nl="FiLM + MLP (MSAL head)", tin_nl="Input concatenation")
ORDER = ("plain", "ctx_add", "film", "bilinear", "sal", "plain_nl", "moe_desc", "sal_nl", "sal_nlb", "film_nl", "tin_nl")
METRICS = (("sampled chunk", "action_selection"), ("flow MSE", "flow_selection"))


def read(path):
  if not Path(path).exists():
    return []
  return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


REC, DUP = {}, 0
for p in sys.argv[4:]:
  for r in read(p):
    if r.get("train_steps") != 12800:
      continue
    k = (int(r["seed"]), r["arm"], int(r["held_task"]))
    if k in REC:
      same = REC[k]["flow_selection"] == r["flow_selection"] and REC[k]["action_selection"] == r["action_selection"]
      assert same, ("duplicate fold with different scores", k, REC[k]["lane"], r["lane"])
      DUP += 1
      continue
    REC[k] = r
STORED_FLOW, STORED_ACT = {}, {}
for p in sorted(N1N2.glob("*/folds.jsonl")):
  if "anchor" in p.parent.name or p.parent.name.startswith("lang"):
    continue
  for r in read(p):
    if r.get("feature") == "blind":
      STORED_FLOW[(int(r["seed"]), r["arm"], int(r["held_task"]))] = float(r["main_selection"])
for s in SEEDS:
  for r in read(IDEAL / ("zero_seed%d.jsonl" % s)):
    STORED_ACT[(s, "film_nl", int(r["task"]))] = float(r["action_selection"])
TASKS = sorted(set(k[2] for k in STORED_FLOW if k[1] == "film_nl"))
assert len(TASKS) == 41
REPORT = dict(comparisons={}, table={}, checks={})


def machine(s, t):
  r = REC.get((s, "plain_nl", t))
  return None if r is None else r["lane"][:3]


def complete(arms_needed):
  return all((s, a, t) in REC for s in SEEDS for t in TASKS for a in arms_needed if not (a == "film_nl" and machine(s, t) == "ws2"))


def err(arm, field, s, t):
  if arm == "film_nl" and machine(s, t) == "ws2":
    return STORED_ACT[(s, arm, t)] if field == "action_selection" else STORED_FLOW[(s, arm, t)]
  r = REC[(s, arm, t)]
  assert r["lane"][:3] == machine(s, t) or r["lane"].startswith("ws2anchor"), (s, arm, t, r["lane"])
  return float(r[field])


def summarize(lr, boot):
  means = np.array([lr[s].mean() for s in sorted(lr)])
  h = stats.t.ppf(0.975, len(means) - 1) * means.std(ddof=1) / math.sqrt(len(means))
  per_task = np.mean([lr[s] for s in sorted(lr)], axis=0)
  won = int(sum((lr[s] < 0).sum() for s in lr))
  out = dict(pct=pct(means.mean()), lo=pct(means.mean() - h), hi=pct(means.mean() + h), per_seed=[pct(x) for x in means],
    folds_lower=won, folds=int(sum(len(lr[s]) for s in lr)), tasks_lower=int((per_task < 0).sum()))
  out["reading"] = "lower" if out["hi"] < 0 else ("higher" if out["lo"] > 0 else "not distinguishable")
  if boot:
    b = per_task[np.random.default_rng(0).integers(0, len(per_task), size=(20000, len(per_task)))].mean(axis=1)
    out["task_boot"] = [pct(np.percentile(b, 2.5)), pct(np.percentile(b, 97.5))]
  return out


def log_ratio(a, b, field, seeds):
  return dict((s, np.array([math.log(err(a, field, s, t) / err(b, field, s, t)) for t in TASKS])) for s in seeds)


def compare(name, a, b, boot=False, seeds=SEEDS):
  REPORT["comparisons"][name] = {}
  for label, field in METRICS:
    r = summarize(log_ratio(a, b, field, seeds), boot)
    REPORT["comparisons"][name][field] = r
    extra = "" if not boot else " | task bootstrap %+.1f..%+.1f" % tuple(r["task_boot"])
    print("  %-34s %-13s %+7.1f%% [%+7.1f, %+7.1f]  %-19s folds lower %3d/%d  per seed %s%s" % (name, label, r["pct"],
      r["lo"], r["hi"], r["reading"], r["folds_lower"], r["folds"], " ".join("%+.1f" % x for x in r["per_seed"]), extra))


HAVE_A, HAVE_B = complete(GROUP_A), complete(GROUP_A) and complete(GROUP_B)
print("records %d (duplicates seen %d); group A complete: %s; group B complete: %s" % (len(REC), DUP, HAVE_A, HAVE_B))
REPORT["complete"] = dict(group_a=HAVE_A, group_b=HAVE_B)
if not HAVE_A:
  n = dict((a, sum((s, a, t) in REC for s in SEEDS for t in TASKS)) for a in ORDER)
  print("records per arm:", n)
  REPORT["records_per_arm"] = n
  OUT.write_text(json.dumps(REPORT, indent=1))
  sys.exit(0)

units = dict(ws2=0, ws4=0)
for s in SEEDS:
  for t in TASKS:
    units[machine(s, t)] += 1
REPORT["units_per_machine"] = units
print("units per machine:", units)
ARMS = [a for a in ORDER if a in GROUP_A or HAVE_B]

print("== checks")
for a in ARMS:
  keys = [(s, a, t) for s in SEEDS for t in TASKS if (s, a, t) in REC]
  ws2 = [k for k in keys if REC[k]["lane"][:3] == "ws2"]
  flags = [REC[k]["flow_bit_exact"] for k in ws2 if REC[k]["flow_bit_exact"] is not None]
  aflags = [REC[k]["action_bit_exact"] for k in ws2 if REC[k]["action_bit_exact"] is not None]
  off = [k for k in keys if REC[k]["lane"][:3] == "ws4" and k in STORED_FLOW]
  noise = [math.log(REC[k]["flow_selection"] / STORED_FLOW[k]) for k in off]
  c = dict(ws2_folds=len(ws2), ws2_flow_exact=int(sum(flags)), ws2_flow_checked=len(flags), ws2_action_exact=int(sum(aflags)),
    ws2_action_checked=len(aflags), server_folds=len(off),
    server_vs_stored_pct=pct(float(np.mean(noise))) if noise else None, server_fold_sd_pct=100 * float(np.std(noise)) if noise else None)
  REPORT["checks"][a] = c
  print("  %-9s ws2 folds %3d: flow bit-exact %d/%d, sampled chunk bit-exact %d/%d | server folds %3d: flow vs stored %s, per-fold SD %s"
    % (a, c["ws2_folds"], c["ws2_flow_exact"], c["ws2_flow_checked"], c["ws2_action_exact"], c["ws2_action_checked"], c["server_folds"],
       "n/a" if not noise else "%+.1f%%" % c["server_vs_stored_pct"], "n/a" if not noise else "%.1f%%" % c["server_fold_sd_pct"]))

print("== table: each head against FiLM and against MSAL's head")
for a in ARMS:
  row = dict(label=LABEL[a])
  for label, field in METRICS:
    row["mean_" + field] = float(np.mean([err(a, field, s, t) for s in SEEDS for t in TASKS]))
    for ref in ("film", "film_nl"):
      row["%s_vs_%s" % (field, ref)] = None if a == ref else summarize(log_ratio(a, ref, field, SEEDS), False)
  REPORT["table"][a] = row

  def cell(x):
    return "       ---            " if x is None else "%+6.1f [%+6.1f, %+6.1f]" % (x["pct"], x["lo"], x["hi"])

  print("  %-30s chunk %.4f vs FiLM %s vs MSAL %s | flow %.4f vs FiLM %s vs MSAL %s" % (LABEL[a],
    row["mean_action_selection"], cell(row["action_selection_vs_film"]), cell(row["action_selection_vs_film_nl"]),
    row["mean_flow_selection"], cell(row["flow_selection_vs_film"]), cell(row["flow_selection_vs_film_nl"])))

print("== comparisons")
compare("H1 film_nl vs plain_nl", "film_nl", "plain_nl", True)
compare("H2 film vs plain", "film", "plain", True)
REPORT["comparisons"]["H3 H1 minus H2"] = {}
for label, field in METRICS:
  one, two = log_ratio("film_nl", "plain_nl", field, SEEDS), log_ratio("film", "plain", field, SEEDS)
  r = summarize(dict((s, one[s] - two[s]) for s in SEEDS), True)
  REPORT["comparisons"]["H3 H1 minus H2"][field] = r
  print("  %-34s %-13s %+7.1f%% [%+7.1f, %+7.1f]  %-19s per seed %s | task bootstrap %+.1f..%+.1f" % ("H3 H1 minus H2", label,
    r["pct"], r["lo"], r["hi"], r["reading"], " ".join("%+.1f" % x for x in r["per_seed"]), r["task_boot"][0], r["task_boot"][1]))
compare("H4 film_nl vs film", "film_nl", "film", True)
compare("H5 plain_nl vs film", "plain_nl", "film", True)
compare("C1 tin_nl vs film_nl", "tin_nl", "film_nl", True)
compare("C1 tin_nl vs film_nl, four new seeds", "tin_nl", "film_nl", True, SEEDS[1:])
compare("tin_nl vs plain_nl", "tin_nl", "plain_nl")
if HAVE_B:
  for a in ("sal_nlb", "sal_nl", "moe_desc"):
    compare("H6 %s vs plain_nl" % a, a, "plain_nl")
  compare("H7 film_nl vs sal_nlb", "film_nl", "sal_nlb")
  for a in ("sal", "ctx_add", "bilinear"):
    compare("H8 %s vs film" % a, a, "film")

C = REPORT["comparisons"]
field = "action_selection"
others = [a for a in ARMS if a != "film_nl"]
not_higher = [a for a in others if REPORT["table"][a]["%s_vs_film_nl" % field]["reading"] != "higher"]
REPORT["readings"] = dict(
  description_helps_more_before_nonlinear_readout=C["H3 H1 minus H2"][field]["reading"] == "lower",
  readout_alone_beats_film=C["H5 plain_nl vs film"][field]["reading"],
  msal_head_most_accurate=(not not_higher) and HAVE_B, arms_not_higher_than_msal_head=not_higher,
  arms_compared=others, input_concatenation_vs_msal_head=C["C1 tin_nl vs film_nl"][field]["reading"],
  input_concatenation_four_new_seeds=C["C1 tin_nl vs film_nl, four new seeds"][field]["reading"])
print("readings (sampled chunk):", json.dumps(REPORT["readings"]))
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
