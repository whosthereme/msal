import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

RES, RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SEEDS = (148021, 148022, 148023, 148024, 148025)
NEW = SEEDS[1:]
WS2 = (0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 27)
OUTLIERS = (1, 2, 31, 43)
FAMILY = {148022: "ws34", 148025: "ws34", 148023: "local", 148024: "local"}
METRIC = dict(flow="flow_selection", act="action_selection", rob="flow_robustness")
NODES = RES / "final_nodes"
LANES = ("out_q_ws3a", "out_q_ws3b", "out_q_ws4a", "out_q_ws4b", "out_ws3", "out_ws4", "out_ws4adup")
REPORT = dict()


def read(path):
  path = Path(path)
  if not path.exists():
    return []
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(values):
  v = np.asarray(values, dtype=float)
  m = float(v.mean())
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(n=len(v), mean_log=m, pct=pct(m), lo=pct(m - h), hi=pct(m + h), sd_pct=100 * float(v.std(ddof=1)))


def boot(values):
  v = np.asarray(values, dtype=float)
  idx = np.random.default_rng(0).integers(0, len(v), size=(20000, len(v)))
  means = v[idx].mean(axis=1)
  return dict(n=len(v), mean_log=float(v.mean()), pct=pct(v.mean()), lo=pct(np.percentile(means, 2.5)),
    hi=pct(np.percentile(means, 97.5)))


def fmt(r):
  word = "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "n.d.")
  return "%+6.1f%% [%+6.1f, %+6.1f] %s" % (r["pct"], r["lo"], r["hi"], word)


def table(path, key, task_key="held_task"):
  return dict((int(r[task_key]), float(r[key])) for r in read(path))


def collect(paths, label, dups):
  out = dict()
  for path in paths:
    for r in read(path):
      t = int(r["held_task"])
      vals = dict((m, float(r[k])) for m, k in METRIC.items())
      if t in out:
        dups.append(dict(what=label, task=t, identical=all(out[t][m] == vals[m] for m in METRIC)))
      else:
        out[t] = vals
  return out


def node_paths(arm, s):
  return [NODES / lane / ("%s_seed%d.jsonl" % (arm, s)) for lane in LANES]


def local_paths(arm, s):
  return [RES / "local" / "out" / ("%s_seed%d.jsonl" % (arm, s))]


STORED = dict()
for s in SEEDS:
  film = RAW / "n1n2" / ("blind_nl_film_nl_seed%d" % s) / "folds.jsonl"
  plain = RAW / "n1n2" / ("blind_nl_plain_nl_seed%d" % s) / "folds.jsonl"
  zero = RAW / "ideal" / ("zero_seed%d.jsonl" % s)
  seen = RAW / "ideal" / ("seen_seed%d.jsonl" % s)
  STORED[s] = dict(flow=table(film, "main_selection"), rob=table(film, "main_robustness"),
    act=table(zero, "action_selection", "task"), plain_flow=table(plain, "main_selection"),
    plain_rob=table(plain, "main_robustness"), seen_flow=table(seen, "flow_selection", "task"),
    seen_act=table(seen, "action_selection", "task"))
TASKS = sorted(STORED[148021]["flow"])
REST = [t for t in TASKS if t not in WS2]
assert len(TASKS) == 41 and len(REST) == 20
for s in SEEDS:
  assert all(sorted(STORED[s][k]) == TASKS for k in STORED[s]), s

print("== ws2 anchors (film_nl on held task 0 through the runner, bit-exact with the stored scores)")
for s in SEEDS:
  flags = [(r.get("anchor_flow_bit_exact"), r.get("anchor_action_bit_exact")) for r in read(RES / ("film_nl_seed%d.jsonl" % s))]
  print("  seed %d: %s" % (s, flags))
REPORT["anchors"] = "printed"

DUPS = []
RATIO = dict()
SOURCE = dict()
MISSING = []
OFF = dict()
for s in SEEDS:
  RATIO[s] = dict((m, dict()) for m in METRIC)
  SOURCE[s] = dict()
  for r in read(RES / ("tfilm_nl_seed%d.jsonl" % s)):
    t = int(r["held_task"])
    assert s == 148021 or t in WS2, (s, t)
    for m, k in METRIC.items():
      RATIO[s][m][t] = math.log(float(r[k]) / STORED[s][m][t])
    SOURCE[s][t] = "ws2"
for s in NEW:
  paths = node_paths if FAMILY[s] == "ws34" else local_paths
  tf = collect(paths("tfilm_nl", s), "tfilm_nl %d" % s, DUPS)
  fn = collect(paths("film_nl", s), "film_nl %d" % s, DUPS)
  OFF[s] = dict(tfilm=tf, film=fn)
  for t in REST:
    if t not in tf or t not in fn:
      MISSING.append((s, t))
      continue
    for m in METRIC:
      RATIO[s][m][t] = math.log(tf[t][m] / fn[t][m])
    SOURCE[s][t] = FAMILY[s]
print("== folds per seed:", dict((s, len(RATIO[s]["act"])) for s in SEEDS), "missing pairs:", MISSING)
print("== duplicated folds within a build family:", DUPS)
REPORT["duplicates"] = DUPS
REPORT["missing"] = MISSING
assert not MISSING
COMPLETE = [s for s in SEEDS if len(RATIO[s]["act"]) == 41]


def seed_reading(metric, seeds, tasks):
  per_seed = [float(np.mean([RATIO[s][metric][t] for t in tasks])) for s in seeds]
  wins = sum(RATIO[s][metric][t] < 0 for s in seeds for t in tasks)
  r = tci(per_seed)
  r.update(per_seed_pct=[pct(v) for v in per_seed], won=int(wins), folds=len(seeds) * len(tasks))
  return r


def task_view(metric, seeds, tasks):
  r = boot([float(np.mean([RATIO[s][metric][t] for s in seeds])) for t in tasks])
  r["tasks_won"] = int(sum(np.mean([RATIO[s][metric][t] for s in seeds]) < 0 for t in tasks))
  return r


def show(name, r, extra=""):
  won = (" won %d/%d" % (r["won"], r["folds"])) if "won" in r else ""
  won = won + ((" tasks won %d/%d" % (r["tasks_won"], r["n"])) if "tasks_won" in r else "")
  print("  %-58s %s%s %s" % (name, fmt(r), won, extra))
  REPORT[name] = r


NAMES = dict(act="sampled-action error", flow="flow MSE (SELECTION)", rob="flow MSE (ROBUSTNESS)")
print("\n== PLAN_5SEED: tfilm_nl vs film_nl (paired log ratio; t interval over seeds)")
print("   seeds complete on all 41 tasks:", COMPLETE)
show("P1 action, 41 tasks, 5 seeds (df 4)", seed_reading("act", COMPLETE, TASKS))
show("P2 flow, 41 tasks, 5 seeds (df 4)", seed_reading("flow", COMPLETE, TASKS))
show("P1 confirm, seeds 148022-25 (df 3)", seed_reading("act", NEW, TASKS))
show("P2 confirm, seeds 148022-25 (df 3)", seed_reading("flow", NEW, TASKS))
show("P1' action, 21 ws2 tasks, 5 seeds", seed_reading("act", SEEDS, WS2))
show("P2' flow, 21 ws2 tasks, 5 seeds", seed_reading("flow", SEEDS, WS2))
show("secondary: ROBUSTNESS flow, 5 seeds", seed_reading("rob", SEEDS, TASKS))
show("second view: action, bootstrap over 41 tasks", task_view("act", SEEDS, TASKS))
show("second view: flow, bootstrap over 41 tasks", task_view("flow", SEEDS, TASKS))
show("second view: ROBUSTNESS, bootstrap over 41 tasks", task_view("rob", SEEDS, TASKS))
KEEP = [t for t in TASKS if t not in OUTLIERS]
show("descriptive: action without tasks 1 2 31 43", seed_reading("act", SEEDS, KEEP))
show("descriptive: flow without tasks 1 2 31 43", seed_reading("flow", SEEDS, KEEP))
show("check: action, ws2 half (21 tasks)", seed_reading("act", SEEDS, WS2))
show("check: action, off-ws2 half (20 tasks), new seeds", seed_reading("act", NEW, REST))
show("check: flow, off-ws2 half (20 tasks), new seeds", seed_reading("flow", NEW, REST))
for s in SEEDS:
  a = np.mean([RATIO[s]["act"][t] for t in TASKS])
  f = np.mean([RATIO[s]["flow"][t] for t in TASKS])
  print("  seed %d: action %+.1f%%  flow %+.1f%%" % (s, pct(a), pct(f)))

print("\n== secondary (chained: same-build ratio x stored ws2 ratio)")
CHAIN = dict()
for s in SEEDS:
  st = STORED[s]
  f_vs_plain = dict((t, math.log(st["flow"][t] / st["plain_flow"][t])) for t in TASKS)
  f_vs_seen_flow = dict((t, math.log(st["flow"][t] / st["seen_flow"][t])) for t in TASKS)
  f_vs_seen_act = dict((t, math.log(st["act"][t] / st["seen_act"][t])) for t in TASKS)
  plain_vs_seen = np.mean([math.log(st["plain_flow"][t] / st["seen_flow"][t]) for t in TASKS])
  CHAIN[s] = dict(
    tf_plain=np.mean([RATIO[s]["flow"][t] + f_vs_plain[t] for t in TASKS]),
    fn_plain=np.mean(list(f_vs_plain.values())),
    tf_seen_flow=np.mean([RATIO[s]["flow"][t] + f_vs_seen_flow[t] for t in TASKS]),
    fn_seen_flow=np.mean(list(f_vs_seen_flow.values())),
    tf_seen_act=np.mean([RATIO[s]["act"][t] + f_vs_seen_act[t] for t in TASKS]),
    fn_seen_act=np.mean(list(f_vs_seen_act.values())),
    gap=plain_vs_seen)
for key, label in (("tf_plain", "tfilm_nl vs plain_nl, flow"), ("fn_plain", "film_nl vs plain_nl, flow (stored)"),
    ("tf_seen_flow", "tfilm_nl vs trained head, flow"), ("fn_seen_flow", "film_nl vs trained head, flow"),
    ("tf_seen_act", "tfilm_nl vs trained head, action"), ("fn_seen_act", "film_nl vs trained head, action")):
  show(label, tci([CHAIN[s][key] for s in SEEDS]))
share_f = [(-CHAIN[s]["fn_plain"]) / CHAIN[s]["gap"] for s in SEEDS]
share_t = [(-CHAIN[s]["tf_plain"]) / CHAIN[s]["gap"] for s in SEEDS]
print("  share of the plain_nl -> trained-head flow gap closed, per seed:")
print("    film_nl ", ["%.2f" % v for v in share_f], "mean %.2f" % np.mean(share_f))
print("    tfilm_nl", ["%.2f" % v for v in share_t], "mean %.2f" % np.mean(share_t))
REPORT["gap_share"] = dict(film_nl=share_f, tfilm_nl=share_t)

print("\n== retrained film_nl vs stored ws2 score (cross-build retraining noise)")
CAL = dict()
for fam, paths, seeds in (("ws34", node_paths, SEEDS), ("local", local_paths, (148022, 148023, 148024))):
  rows = dict(flow=[], act=[])
  for s in seeds:
    for t, v in collect(paths("film_nl", s), "cal %s %d" % (fam, s), []).items():
      rows["flow"].append(math.log(v["flow"] / STORED[s]["flow"][t]))
      rows["act"].append(math.log(v["act"] / STORED[s]["act"][t]))
  for m in ("flow", "act"):
    r = tci(rows[m])
    r["max_abs_pct"] = max(abs(pct(x)) for x in rows[m])
    r["bit_identical"] = int(sum(x == 0.0 for x in rows[m]))
    show("calibration %s %s (%d folds)" % (fam, m, len(rows[m])), r,
      "SD %.1f%% max %.1f%% identical %d" % (r["sd_pct"], r["max_abs_pct"], r["bit_identical"]))
    CAL[fam + "_" + m] = r

print("\n== PLAN_ACTION (seed 148021, ws3 / ws4, exploratory): bootstrap over 41 tasks")
P = dict()
STORED_SAME = dict()
for arm in ("plain_nl", "film_nl", "film"):
  P[arm] = collect(node_paths(arm, 148021), "action %s" % arm, DUPS)
  STORED_SAME[arm] = dict()
  for lane in LANES:
    for r in read(NODES / lane / ("%s_seed148021.jsonl" % arm)):
      if r.get("stored_same_arm_flow") is not None:
        STORED_SAME[arm][int(r["held_task"])] = float(r["stored_same_arm_flow"])
  if arm == "film_nl":
    STORED_SAME[arm] = dict(STORED[148021]["flow"])
  if arm == "film":
    g1 = read(RAW / "n1n2" / "blind_g1_seed148021" / "folds.jsonl")
    STORED_SAME[arm] = dict((int(r["held_task"]), float(r["main_selection"])) for r in g1 if r["arm"] == "film")
  assert sorted(P[arm]) == TASKS and sorted(STORED_SAME[arm]) == TASKS, arm


def paired(a, b, metric):
  lr = [math.log(P[a][t][metric] / P[b][t][metric]) for t in TASKS]
  r = boot(lr)
  r["tasks_won"] = int(sum(x < 0 for x in lr))
  return r


def stored_pair(a, b):
  lr = [math.log(STORED_SAME[a][t] / STORED_SAME[b][t]) for t in TASKS]
  r = boot(lr)
  r["tasks_won"] = int(sum(x < 0 for x in lr))
  return r


show("A1 film_nl vs plain_nl, action (retrained)", paired("film_nl", "plain_nl", "act"))
show("A1 film_nl vs plain_nl, flow (retrained)", paired("film_nl", "plain_nl", "flow"))
show("A1 film_nl vs plain_nl, flow (stored ws2)", stored_pair("film_nl", "plain_nl"))
show("A2 film_nl vs film, action (retrained)", paired("film_nl", "film", "act"))
show("A2 film_nl vs film, flow (retrained)", paired("film_nl", "film", "flow"))
show("A2 film_nl vs film, flow (stored ws2)", stored_pair("film_nl", "film"))
for arm in ("plain_nl", "film_nl", "film"):
  lr = [math.log(P[arm][t]["flow"] / STORED_SAME[arm][t]) for t in TASKS]
  r = tci(lr)
  show("check: %s retrained vs stored flow" % arm, r, "SD %.1f%%" % r["sd_pct"])
chain = [RATIO[148021]["act"][t] + math.log(P["film_nl"][t]["act"] / P["plain_nl"][t]["act"]) for t in TASKS]
show("post hoc: tfilm_nl vs plain_nl, action (chained)", boot(chain))
seen_ratio = [math.log(P["plain_nl"][t]["act"] / STORED[148021]["seen_act"][t]) for t in TASKS]
show("context: plain_nl (ws3/4) vs trained head (ws2), action", boot(seen_ratio))

print("\n== PLAN_LODO (seed 148021, this computer, exploratory): bootstrap over 16 directions")
L = dict()
for arm in ("lodo_plain", "lodo_sat", "lodo_xy6", "lodo_h9", "lodo_txy6"):
  recs = read(RES / "local" / "out_lodo" / ("%s_seed148021.jsonl" % arm))
  L[arm] = dict((int(r["held_task"]), r) for r in recs)
  assert sorted(L[arm]) == TASKS, arm
DIR = dict((t, int(L["lodo_plain"][t]["held_direction"])) for t in TASKS)
DIRS = sorted(set(DIR.values()))


def cluster(lr):
  lr = np.asarray(lr, dtype=float)
  sums = np.array([lr[[i for i, t in enumerate(TASKS) if DIR[t] == d]].sum() for d in DIRS])
  counts = np.array([sum(DIR[t] == d for t in TASKS) for d in DIRS], dtype=float)
  idx = np.random.default_rng(0).integers(0, len(DIRS), size=(20000, len(DIRS)))
  means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
  return dict(n=len(lr), mean_log=float(lr.mean()), pct=pct(lr.mean()), lo=pct(np.percentile(means, 2.5)),
    hi=pct(np.percentile(means, 97.5)), tasks_won=int((lr < 0).sum()),
    dirs_won=int(sum(s < 0 for s in sums / counts)), dirs=len(DIRS))


def lodo_value(arm, t, metric):
  return float(L[arm][t][METRIC[metric]])


def lodo_show(a, b, metric):
  lr = [math.log(lodo_value(a, t, metric) / lodo_value(b, t, metric)) for t in TASKS]
  r = cluster(lr)
  show("%s vs %s, %s" % (a, b, metric), r, "directions won %d/%d" % (r["dirs_won"], r["dirs"]))


for a, b in (("lodo_xy6", "lodo_sat"), ("lodo_xy6", "lodo_plain"), ("lodo_sat", "lodo_plain"),
    ("lodo_h9", "lodo_xy6"), ("lodo_txy6", "lodo_xy6"), ("lodo_txy6", "lodo_plain"), ("lodo_h9", "lodo_plain")):
  for metric in ("act", "flow"):
    lodo_show(a, b, metric)
print("  context: unseen direction (LODO, this computer) vs leave-one-task-out (stored ws2), same arm")
ctx = (("lodo_sat", "flow", "flow"), ("lodo_sat", "act", "act"), ("lodo_plain", "flow", "plain_flow"))
for arm, metric, key in ctx:
  lr = [math.log(lodo_value(arm, t, metric) / STORED[148021][key][t]) for t in TASKS]
  r = cluster(lr)
  show("context: %s vs LOTO %s" % (arm, key), r)
for arm in ("lodo_plain", "lodo_sat", "lodo_xy6", "lodo_h9", "lodo_txy6"):
  for metric, key in (("act", "seen_act"), ("flow", "seen_flow")):
    lr = [math.log(lodo_value(arm, t, metric) / STORED[148021][key][t]) for t in TASKS]
    show("context: %s vs trained head, %s" % (arm, metric), cluster(lr))

OUT.write_text(json.dumps(dict(report=REPORT, ratio=dict((str(s), RATIO[s]) for s in SEEDS),
  source=dict((str(s), SOURCE[s]) for s in SEEDS)), indent=1, default=float))
print("\nwritten", OUT)
