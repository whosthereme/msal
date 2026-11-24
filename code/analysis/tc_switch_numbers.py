import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

RES, RAW, PREFIX = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
SEEDS = (148021, 148022, 148023, 148024, 148025)
WS2 = (0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 27)
FAMILY = {148022: "ws34", 148025: "ws34", 148023: "local", 148024: "local"}
METRIC = dict(act="action_selection", flow="flow_selection")
LANES = ("out_q_ws3a", "out_q_ws3b", "out_q_ws4a", "out_q_ws4b", "out_ws3", "out_ws4", "out_ws4adup")
LINES, REPORT = [], {}


def say(s=""):
  LINES.append(s)
  print(s)


def read(path):
  path = Path(path)
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def summarize(lr, tasks_n):
  seeds = sorted(lr)
  means = np.array([lr[s].mean() for s in seeds])
  h = stats.t.ppf(0.975, len(means) - 1) * means.std(ddof=1) / math.sqrt(len(means))
  per_task = np.mean([lr[s] for s in seeds], axis=0)
  b = per_task[np.random.default_rng(0).integers(0, len(per_task), size=(20000, len(per_task)))].mean(axis=1)
  return dict(pct=pct(means.mean()), lo=pct(means.mean() - h), hi=pct(means.mean() + h),
    per_seed=[round(pct(x), 1) for x in means], folds_lower=int(sum((lr[s] < 0).sum() for s in seeds)),
    folds=int(sum(len(lr[s]) for s in seeds)), tasks_lower=int((per_task < 0).sum()), tasks=tasks_n,
    task_boot=[pct(np.percentile(b, 2.5)), pct(np.percentile(b, 97.5))])


def line(name, r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  say("  %-62s %+7.1f%% [%+7.1f, %+7.1f] %-6s folds %3d/%d tasks %d/%d  per seed %s" % (name, r["pct"], r["lo"], r["hi"], word,
    r["folds_lower"], r["folds"], r["tasks_lower"], r["tasks"], r["per_seed"]))
  REPORT[name] = r


def table(path, key, task_key="held_task"):
  return dict((int(r[task_key]), float(r[key])) for r in read(path))


STORED = {}
for s in SEEDS:
  STORED[s] = dict(flow=table(RAW / "n1n2" / ("blind_nl_film_nl_seed%d" % s) / "folds.jsonl", "main_selection"),
    act=table(RAW / "ideal" / ("zero_seed%d.jsonl" % s), "action_selection", "task"))
TASKS = sorted(STORED[148021]["flow"])
REST = [t for t in TASKS if t not in WS2]
TF, TF_RAW = {}, {}
for s in SEEDS:
  for r in read(RES / ("tfilm_nl_seed%d.jsonl" % s)):
    t = int(r["held_task"])
    for m, k in METRIC.items():
      TF[(s, t, m)] = math.log(float(r[k]) / STORED[s][m][t])
      TF_RAW[(s, t, m)] = float(r[k])
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
      TF_RAW[(s, t, m)] = float(got["tfilm_nl"][t][k])
assert all((s, t, m) in TF for s in SEEDS for t in TASKS for m in METRIC)

REC = {}
for p in sorted((RES / "heads5" / "full").glob("folds_*.jsonl")):
  for r in read(p):
    if r.get("train_steps") == 12800:
      REC.setdefault((int(r["seed"]), r["arm"], int(r["held_task"])), r)
STORED_FLOW = {}
for p in sorted((RAW / "n1n2").glob("*/folds.jsonl")):
  if "anchor" in p.parent.name or p.parent.name.startswith("lang"):
    continue
  for r in read(p):
    if r.get("feature") == "blind":
      STORED_FLOW[(int(r["seed"]), r["arm"], int(r["held_task"]))] = float(r["main_selection"])


def machine(s, t):
  return REC[(s, "plain_nl", t)]["lane"][:3]


def rerun_lr(arm, s, t, m):
  k = METRIC[m]
  film = STORED[s][m][t] if machine(s, t) == "ws2" else float(REC[(s, "film_nl", t)][k])
  return math.log(float(REC[(s, arm, t)][k]) / film)


ARMS = ("plain", "ctx_add", "film", "bilinear", "sal", "plain_nl", "moe_desc", "sal_nl", "sal_nlb", "tin_nl")
say("== A  development family, zero-shot, trunk modulation (tfilm_nl); 41 tasks, five seeds")
say("  mean errors of tfilm_nl over the 205 folds: flow %.4f, sampled chunk %.5f" % (
  float(np.mean([TF_RAW[(s, t, "flow")] for s in SEEDS for t in TASKS])),
  float(np.mean([TF_RAW[(s, t, "act")] for s in SEEDS for t in TASKS]))))
REPORT["A means"] = dict(flow=float(np.mean([TF_RAW[(s, t, "flow")] for s in SEEDS for t in TASKS])),
  act=float(np.mean([TF_RAW[(s, t, "act")] for s in SEEDS for t in TASKS])))
for m in ("act", "flow"):
  line("A tfilm vs film_nl (registered, 5SEED), %s" % m,
    summarize(dict((s, np.array([TF[(s, t, m)] for t in TASKS])) for s in SEEDS), 41))
say(" chained through film_nl of the HEADS5 rerun (sampled chunk and flow of the rerun)")
for arm in ARMS:
  for m in ("act", "flow"):
    line("A tfilm vs %s, %s (chained, rerun)" % (arm, m),
      summarize(dict((s, np.array([TF[(s, t, m)] - rerun_lr(arm, s, t, m) for t in TASKS])) for s in SEEDS), 41))
say(" flow chained through the stored original runs on the workstation (the flow columns of Table I)")
for arm in ARMS[:-1]:
  lr = dict((s, np.array([TF[(s, t, "flow")] - math.log(STORED_FLOW[(s, arm, t)] / STORED_FLOW[(s, "film_nl", t)])
    for t in TASKS])) for s in SEEDS)
  line("A tfilm vs %s, flow (chained, stored original)" % arm, summarize(lr, 41))

AD = {}
for p in sorted((RES / "adapt").rglob("folds.jsonl")) + sorted((RES / "adapt5").rglob("folds.jsonl")):
  for r in read(p):
    AD.setdefault((int(r["seed"]), r["arm"], int(r["held_task"])), r)
TIN = defaultdict(dict)
for p in sorted((RES / "tin").glob("outt_*.jsonl")):
  for r in read(p):
    TIN[(r["family"], r["arm"])].setdefault((int(r["seed"]), int(r["held_task"])), r)
assert all((s, a, t) in AD for s in SEEDS for a in ("tfilm_nl", "film_nl") for t in TASKS), "adaptation records incomplete"


def fl(r, stage, k=None):
  if stage == "zero":
    return r["zero_shot"]["selection"] if "zero_shot" in r else r["score_forward"]["selection"]
  return r[stage][str(k)]["selection"]


def lr_of(a, b, tasks):
  return dict((s, np.array([math.log(a(s, t) / b(s, t)) for t in tasks])) for s in SEEDS)


say("== B  development family, adaptation on the trunk-modulated head (ADAPT_TRUNK, ADAPT_TRUNK5 records)")
A_ = lambda arm, f: (lambda s, t: f(AD[(s, arm, t)]))
for k in (1, 16):
  line("B trunk: inversion vs own zero-shot, flow, k=%d" % k, summarize(lr_of(A_("tfilm_nl", lambda r, k=k: fl(r, "inversion", k)),
    A_("tfilm_nl", lambda r: fl(r, "zero")), TASKS), 41))
  line("B trunk: inversion vs own zero-shot, chunk, k=%d" % k, summarize(lr_of(A_("tfilm_nl", lambda r, k=k: r["action"]["inversion_%d" % k]),
    A_("tfilm_nl", lambda r: r["action"]["zero_shot"]), TASKS), 41))
  line("B trunk: method vs own zero-shot, flow, k=%d" % k, summarize(lr_of(A_("tfilm_nl", lambda r, k=k: fl(r, "combination", k)),
    A_("tfilm_nl", lambda r: fl(r, "zero")), TASKS), 41))
  line("B trunk: method vs own zero-shot, chunk, k=%d" % k, summarize(lr_of(A_("tfilm_nl", lambda r, k=k: r["action"]["combination_%d" % k]),
    A_("tfilm_nl", lambda r: r["action"]["zero_shot"]), TASKS), 41))
  line("B trunk: code beyond inversion, flow, k=%d" % k, summarize(lr_of(A_("tfilm_nl", lambda r, k=k: fl(r, "combination", k)),
    A_("tfilm_nl", lambda r, k=k: fl(r, "inversion", k)), TASKS), 41))
say(" against input concatenation with the same adaptation (TIN records; chained through film_nl: the tin record is")
say(" paired with the film_nl record of this study, which ran in one process with tfilm_nl)")
T_ = lambda f: (lambda s, t: f(TIN[("dev", "tin_nl")][(s, t)]))
assert all((s, t) in TIN[("dev", "tin_nl")] for s in SEEDS for t in TASKS)
line("B trunk vs concatenation, zero-shot, chunk", summarize(lr_of(A_("tfilm_nl", lambda r: r["action"]["zero_shot"]),
  T_(lambda r: r["action"]["zero_shot"]), TASKS), 41))
line("B trunk vs concatenation, zero-shot, flow", summarize(lr_of(A_("tfilm_nl", lambda r: fl(r, "zero")),
  T_(lambda r: fl(r, "zero")), TASKS), 41))
for k in (1, 16):
  line("B trunk vs concatenation, method, chunk, k=%d" % k, summarize(lr_of(A_("tfilm_nl", lambda r, k=k: r["action"]["combination_%d" % k]),
    T_(lambda r, k=k: r["action"]["combination_%d" % k]), TASKS), 41))
  line("B trunk vs concatenation, method, flow, k=%d" % k, summarize(lr_of(A_("tfilm_nl", lambda r, k=k: fl(r, "combination", k)),
    T_(lambda r, k=k: fl(r, "combination", k)), TASKS), 41))
shift = dict((arm, float(np.mean([AD[(s, arm, t)]["combination"]["16"]["descriptor_shift"] for s in SEEDS for t in TASKS])))
  for arm in ("tfilm_nl", "film_nl"))
say("  mean descriptor shift by inversion at k=16: trunk %.2f, h %.2f" % (shift["tfilm_nl"], shift["film_nl"]))
REPORT["B shift"] = shift

L40 = {}
for p in sorted((RES / "l40").glob("out_*.jsonl")):
  for r in read(p):
    L40.setdefault((int(r["seed"]), r["arm"], int(r["held_task"])), r)
LT = list(range(1000, 1040))
assert all((s, a, t) in L40 for s in SEEDS for a in ("tfilm_nl", "film_nl") for t in LT)
assert all((s, t) in TIN[("l40", a)] for s in SEEDS for t in LT for a in ("tin_nl", "plain_nl"))
L_ = lambda arm, f: (lambda s, t: f(L40[(s, arm, t)]))
N_ = lambda arm, f: (lambda s, t: f(TIN[("l40", arm)][(s, t)]))
say("== C  LIBERO40 (L40_TRUNK and TIN records; cross-study pairs are chained through film_nl as above)")
for name, a, b in (
    ("C trunk vs h, zero-shot, chunk", L_("tfilm_nl", lambda r: r["action"]["zero_shot"]), L_("film_nl", lambda r: r["action"]["zero_shot"])),
    ("C trunk vs none (nonlinear readout), zero-shot, chunk", L_("tfilm_nl", lambda r: r["action"]["zero_shot"]), N_("plain_nl", lambda r: r["action"]["zero_shot"])),
    ("C trunk vs none (nonlinear readout), zero-shot, flow", L_("tfilm_nl", lambda r: fl(r, "zero")), N_("plain_nl", lambda r: fl(r, "zero"))),
    ("C h vs none, zero-shot, chunk (TIN L2 again)", L_("film_nl", lambda r: r["action"]["zero_shot"]), N_("plain_nl", lambda r: r["action"]["zero_shot"])),
    ("C trunk vs concatenation, zero-shot, chunk", L_("tfilm_nl", lambda r: r["action"]["zero_shot"]), N_("tin_nl", lambda r: r["action"]["zero_shot"])),
    ("C trunk vs concatenation, zero-shot, flow", L_("tfilm_nl", lambda r: fl(r, "zero")), N_("tin_nl", lambda r: fl(r, "zero"))),
    ("C trunk vs concatenation, method, chunk, k=1", L_("tfilm_nl", lambda r: r["action"]["combination_1"]), N_("tin_nl", lambda r: r["action"]["combination_1"])),
    ("C trunk vs concatenation, method, chunk, k=16", L_("tfilm_nl", lambda r: r["action"]["combination_16"]), N_("tin_nl", lambda r: r["action"]["combination_16"])),
    ("C trunk vs concatenation, method, flow, k=1", L_("tfilm_nl", lambda r: fl(r, "combination", 1)), N_("tin_nl", lambda r: fl(r, "combination", 1))),
    ("C trunk vs concatenation, method, flow, k=16", L_("tfilm_nl", lambda r: fl(r, "combination", 16)), N_("tin_nl", lambda r: fl(r, "combination", 16))),
    ("C trunk: method vs none, chunk, k=1", L_("tfilm_nl", lambda r: r["action"]["combination_1"]), N_("plain_nl", lambda r: r["action"]["zero_shot"])),
    ("C trunk: method vs none, chunk, k=16", L_("tfilm_nl", lambda r: r["action"]["combination_16"]), N_("plain_nl", lambda r: r["action"]["zero_shot"])),
    ("C trunk: inversion vs own zero-shot, flow, k=1", L_("tfilm_nl", lambda r: fl(r, "inversion", 1)), L_("tfilm_nl", lambda r: fl(r, "zero"))),
    ("C trunk: inversion vs own zero-shot, flow, k=16", L_("tfilm_nl", lambda r: fl(r, "inversion", 16)), L_("tfilm_nl", lambda r: fl(r, "zero"))),
    ("C trunk: method vs own zero-shot, flow, k=1", L_("tfilm_nl", lambda r: fl(r, "combination", 1)), L_("tfilm_nl", lambda r: fl(r, "zero"))),
    ("C trunk: method vs own zero-shot, flow, k=16", L_("tfilm_nl", lambda r: fl(r, "combination", 16)), L_("tfilm_nl", lambda r: fl(r, "zero"))),
    ("C trunk: method vs own zero-shot, chunk, k=1", L_("tfilm_nl", lambda r: r["action"]["combination_1"]), L_("tfilm_nl", lambda r: r["action"]["zero_shot"])),
    ("C trunk: method vs own zero-shot, chunk, k=16", L_("tfilm_nl", lambda r: r["action"]["combination_16"]), L_("tfilm_nl", lambda r: r["action"]["zero_shot"]))):
  line(name, summarize(lr_of(a, b, LT), 40))
shift = dict((arm, float(np.mean([L40[(s, arm, t)]["combination"]["16"]["descriptor_shift"] for s in SEEDS for t in LT])))
  for arm in ("tfilm_nl", "film_nl"))
say("  mean descriptor shift by inversion at k=16: trunk %.2f, h %.2f" % (shift["tfilm_nl"], shift["film_nl"]))
REPORT["C shift"] = shift

say("== D  cluttered closed-loop family: K0 against stored policies (two-level bootstrap of lc_analyze3, seed")
say("      [20261121, index]; indices 1-3 and 21-23 are the plan's H7-H9; 4-6 and 24-26 below are post hoc)")
held = []
for d, pols in ((RES / "clh" / "stored" / "results", ("C0", "N0")), (RES / "clh" / "results2", None), (RES / "clt" / "results3", None)):
  for p in sorted(d.glob("held_*.jsonl")):
    for r in read(p):
      if pols is None or r["policy"] in pols:
        held.append(r)


def cl_table(key):
  t = defaultdict(lambda: defaultdict(dict))
  for r in held:
    t[r["policy"]][r["task"]][(r["seed"], r["episode"])] = float(r[key])
  return t


def paired(a, b, key, index, stat="mean"):
  t = cl_table(key)
  tasks = sorted(set(t[a]) & set(t[b]))
  units = {k: sorted(set(t[a][k]) & set(t[b][k])) for k in tasks}
  A = [np.array([t[a][k][u] for u in units[k]]) for k in tasks]
  Bv = [np.array([t[b][k][u] for u in units[k]]) for k in tasks]
  f = np.mean if stat == "mean" else np.median

  def est(ti, ui):
    return float(np.mean([f(A[i][u]) - f(Bv[i][u]) for i, u in zip(ti, ui)]))

  point = est(range(len(tasks)), [np.arange(len(units[k])) for k in tasks])
  rng = np.random.default_rng([20261121, index])
  boot = np.empty(10_000)
  for j in range(10_000):
    ti = rng.integers(len(tasks), size=len(tasks))
    boot[j] = est(ti, [rng.integers(len(A[i]), size=len(A[i])) for i in ti])
  lo, hi = np.percentile(boot, [2.5, 97.5])
  return dict(a=a, b=b, key=key, a_mean=float(np.mean([f(x) for x in A])), b_mean=float(np.mean([f(x) for x in Bv])), diff=point,
    ci=[float(lo), float(hi)], targets=len(tasks), units=int(sum(len(u) for u in units.values())),
    a_tasks_better=int(sum(f(x) > f(y) for x, y in zip(A, Bv))), a_tasks_worse=int(sum(f(x) < f(y) for x, y in zip(A, Bv))))


for i, b in ((1, "T0"), (2, "C0"), (3, "N0"), (4, "F0"), (5, "M0"), (6, "G0")):
  v = paired("K0", b, "success", i)
  d = paired("K0", b, "final_dist", 20 + i, stat="median")
  REPORT["D K0 vs %s" % b] = dict(success=v, dist=d)
  say("  K0 vs %s: success %.1f vs %.1f, diff %+.1f [%+.1f, %+.1f] %-19s targets better/worse %d/%d | median distance %.1f vs %.1f cm, "
      "diff %+.1f [%+.1f, %+.1f]%s" % (b, 100 * v["a_mean"], 100 * v["b_mean"], 100 * v["diff"], 100 * v["ci"][0], 100 * v["ci"][1],
      "better" if v["ci"][0] > 0 else ("worse" if v["ci"][1] < 0 else "not distinguishable"), v["a_tasks_better"], v["a_tasks_worse"],
      100 * d["a_mean"], 100 * d["b_mean"], 100 * d["diff"], 100 * d["ci"][0], 100 * d["ci"][1], "" if i <= 3 else "  (post hoc)"))
for i, (a_, b_) in ((7, ("T0", "F0")), (8, ("T0", "M0"))):
  v = paired(a_, b_, "success", i)
  REPORT["D %s vs %s" % (a_, b_)] = dict(success=v)
  say("  %s vs %s: success %.1f vs %.1f, diff %+.1f [%+.1f, %+.1f]  (post hoc)" % (a_, b_, 100 * v["a_mean"], 100 * v["b_mean"],
    100 * v["diff"], 100 * v["ci"][0], 100 * v["ci"][1]))
Path(PREFIX + ".json").write_text(json.dumps(REPORT, indent=1))
Path(PREFIX + ".txt").write_text("\n".join(LINES) + "\n")
