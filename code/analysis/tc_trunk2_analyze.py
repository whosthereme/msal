import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

args = sys.argv[1:]
RES, PREFIX = Path(args[0]), args[1]
groups, cur = dict(a=[], b=[], bench=[], units4=[]), None
for x in args[2:]:
  if x.startswith("--"):
    cur = x[2:]
  else:
    groups[cur].append(Path(x))
SEEDS = (148021, 148022, 148023, 148024, 148025)
LINES, REPORT = [], {}


def say(s=""):
  LINES.append(s)
  print(s)


def read(path):
  path = Path(path)
  return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(means):
  v = np.asarray(means, dtype=float)
  m = float(v.mean())
  if len(v) < 2:
    return dict(seeds=len(v), pct=pct(m), lo=float("nan"), hi=float("nan"))
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(seeds=len(v), pct=pct(m), lo=pct(m - h), hi=pct(m + h), per_seed=[round(pct(x), 1) for x in v])


def word(r):
  if r["lo"] != r["lo"]:
    return "one seed"
  return "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "not distinguishable")


A = {}
for p in groups["a"]:
  for r in read(p):
    if r.get("train_steps") != 12800:
      continue
    k = (int(r["seed"]), r["arm"], int(r["held_task"]))
    assert k not in A, k
    A[k] = r
ARMS = ("lodo_plain", "lodo_tsat", "lodo_txy6", "lodo_txy6id05", "lodo_txy6id1", "lodo_txy6id2")
TASKS = sorted(set(k[2] for k in A))
DIRECTION = dict((k[2], int(r["held_direction"])) for k, r in A.items())
FIELD = dict(act="action_selection", flow="flow_selection", rob="flow_robustness")


def a_value(arm, metric, proj=False):
  key = ("proj_" if proj else "") + FIELD[metric]
  return lambda s, t: float(A[(s, arm, t)][key])


def stored_value(table, arm, metric):
  return lambda s, t: float(table[(s, arm, t)][FIELD[metric]])


def reading(name, a, b, seeds, tasks):
  lr = dict((s, np.array([math.log(a(s, t) / b(s, t)) for t in tasks])) for s in seeds)
  r = tci([lr[s].mean() for s in seeds])
  per_task = np.mean([lr[s] for s in seeds], axis=0)
  dirs = sorted(set(DIRECTION[t] for t in tasks))
  per_dir = np.array([per_task[[i for i, t in enumerate(tasks) if DIRECTION[t] == d]].mean() for d in dirs])
  boot = per_dir[np.random.default_rng(0).integers(0, len(per_dir), size=(20000, len(per_dir)))].mean(axis=1)
  r.update(tasks_lower=int((per_task < 0).sum()), tasks=len(tasks), directions_lower=int((per_dir < 0).sum()),
    directions=len(dirs), folds_lower=int(sum((lr[s] < 0).sum() for s in seeds)), folds=len(seeds) * len(tasks),
    direction_boot=[pct(np.percentile(boot, 2.5)), pct(np.percentile(boot, 97.5))], reading=word(r))
  REPORT[name] = r
  say("  %-58s %+7.1f%% [%+7.1f, %+7.1f] %-19s tasks %2d/%d dirs %2d/%d boot %+.1f..%+.1f per seed %s" % (name, r["pct"],
    r["lo"], r["hi"], r["reading"], r["tasks_lower"], r["tasks"], r["directions_lower"], r["directions"],
    r["direction_boot"][0], r["direction_boot"][1], r.get("per_seed")))
  return r


if A:
  done = [s for s in SEEDS if all((s, a, t) in A for a in ARMS for t in TASKS)]
  say("== Part A: held directions, MSAL's head. records %d; tasks %d; complete seeds %s" % (len(A), len(TASKS), done))
  hosts = defaultdict(int)
  for (s, a, t), r in A.items():
    if a == "lodo_plain":
      hosts[r.get("host", "?")] += 1
  say("  seed-task units per host: %s" % dict(hosts))
  REPORT["A units per host"] = dict(hosts)
  for a in ARMS[1:]:
    io = [A[k]["iota"] for k in A if k[1] == a]
    say("  mean span share %-14s %.3f (%.3f to %.3f)" % (a, np.mean(io), min(io), max(io)))
    REPORT["A iota " + a] = dict(mean=float(np.mean(io)), lo=float(min(io)), hi=float(max(io)))
  if done:
    for metric in ("act", "flow"):
      m = "sampled chunk" if metric == "act" else "flow"
      reading("A1 harmonic vs none, %s" % m, a_value("lodo_tsat", metric), a_value("lodo_plain", metric), done, TASKS)
      reading("A2 coordinates vs none, %s" % m, a_value("lodo_txy6", metric), a_value("lodo_plain", metric), done, TASKS)
      reading("A3 coordinates vs harmonic, %s" % m, a_value("lodo_txy6", metric), a_value("lodo_tsat", metric), done, TASKS)
      for arm, al in (("lodo_txy6id2", "2"), ("lodo_txy6id1", "1"), ("lodo_txy6id05", "0.5")):
        reading("A4 identifier alpha=%s vs coordinates, %s" % (al, m), a_value(arm, metric), a_value("lodo_txy6", metric), done, TASKS)
      reading("A4 identifier alpha=2 vs none, %s" % m, a_value("lodo_txy6id2", metric), a_value("lodo_plain", metric), done, TASKS)
      reading("A5 harmonic in-span vs whole, %s" % m, a_value("lodo_tsat", metric, True), a_value("lodo_tsat", metric), done, TASKS)
      reading("A5 identifier in-span vs whole, %s" % m, a_value("lodo_txy6id2", metric, True), a_value("lodo_txy6id2", metric), done, TASKS)
      reading("A5 harmonic in-span vs none, %s" % m, a_value("lodo_tsat", metric, True), a_value("lodo_plain", metric), done, TASKS)
      reading("A5 harmonic in-span vs coordinates, %s" % m, a_value("lodo_tsat", metric, True), a_value("lodo_txy6", metric), done, TASKS)
      reading("A5 identifier in-span vs coordinates, %s" % m, a_value("lodo_txy6id2", metric, True), a_value("lodo_txy6", metric), done, TASKS)
    order = [REPORT["A4 identifier alpha=%s vs coordinates, sampled chunk" % al]["pct"] for al in ("0.5", "1", "2")]
    say("  A4 order of alpha 0.5 < 1 < 2 in sampled chunk: %s (%s)" % (order[0] < order[1] < order[2], [round(x, 1) for x in order]))
    REPORT["A4 monotone"] = bool(order[0] < order[1] < order[2])
    raw_ok = [A[k]["raw_matches_stored_scorer"] for k in A if "raw_matches_stored_scorer" in A[k]]
    say("  projected arms: raw flow equals the stored scorer on %d of %d records" % (sum(raw_ok), len(raw_ok)))
    STORED = {}
    for d in (RES / "local" / "out_lodo", RES / "final_nodes" / "out_lodo"):
      for arm in ("lodo_plain", "lodo_xy6", "lodo_sat", "lodo_txy6"):
        for s in SEEDS:
          for r in read(d / ("%s_seed%d.jsonl" % (arm, s))):
            STORED.setdefault((s, arm, int(r["held_task"])), dict(r, _src=d.parent.name))
    for p in sorted((RES / "span" / "b").glob("out*.jsonl")):
      for r in read(p):
        if r.get("train_steps") == 12800:
          STORED.setdefault((int(r["seed"]), r["arm"], int(r["held_task"])), dict(r, _src="span"))
    say(" anchors: new record against the stored record of the same arm, seed and task")
    for arm in ("lodo_plain", "lodo_txy6"):
      keys = [(s, arm, t) for s in done for t in TASKS if (s, arm, t) in STORED]
      same = sum(A[k]["flow_selection"] == STORED[k]["flow_selection"] and A[k]["action_selection"] == STORED[k]["action_selection"] for k in keys)
      by = defaultdict(lambda: [0, 0])
      for k in keys:
        e = by[(A[k].get("host", "?"), STORED[k]["_src"])]
        e[0] += 1
        e[1] += int(A[k]["flow_selection"] == STORED[k]["flow_selection"] and A[k]["action_selection"] == STORED[k]["action_selection"])
      lr = [math.log(A[k]["action_selection"] / STORED[k]["action_selection"]) for k in keys]
      say("  %-11s pairs %3d, bit-identical %3d, mean log ratio of sampled chunk %+.2f%% (SD %.1f%%); by (new host, stored source): %s" % (
        arm, len(keys), same, pct(float(np.mean(lr))) if lr else float("nan"), 100 * float(np.std(lr)) if lr else float("nan"),
        dict((str(k), v) for k, v in by.items())))
      REPORT["A anchor " + arm] = dict(pairs=len(keys), identical=int(same))
    say(" descriptive: MSAL's head against the stored MSAL-h record of the same descriptor (other process)")
    for new, old, label in (("lodo_tsat", "lodo_sat", "harmonic"), ("lodo_txy6", "lodo_xy6", "coordinates"),
        ("lodo_txy6id05", "lodo_xy6id05", "identifier 0.5"), ("lodo_txy6id1", "lodo_xy6id1", "identifier 1"),
        ("lodo_txy6id2", "lodo_xy6id2", "identifier 2")):
      if all((s, old, t) in STORED for s in done for t in TASKS):
        for metric in ("act", "flow"):
          reading("MSAL vs MSAL-h, %s, %s" % (label, "sampled chunk" if metric == "act" else "flow"), a_value(new, metric),
            stored_value(STORED, old, metric), done, TASKS)
      else:
        say("  MSAL vs MSAL-h, %s: stored records incomplete" % label)

BALL = defaultdict(dict)
for p in groups["b"]:
  for r in read(p):
    if r.get("train_steps") == 12800:
      m = "ws2" if r["host"].startswith("workstation") else "a800"
      k = (int(r["seed"]), int(r["held_task"]))
      assert m not in BALL[k], (k, m)
      BALL[k][m] = r
U4 = set()
for p in groups["units4"]:
  for line in p.read_text().splitlines():
    if line.strip():
      s_, t_ = line.split()
      U4.add((int(s_), int(t_)))
B = {}
for k, v in BALL.items():
  want = "a800" if (U4 and k in U4) else "ws2"
  if want in v:
    B[k] = v[want]
  elif not U4:
    B[k] = next(iter(v.values()))
if B:
  WORDS, NONE = {}, {}
  for p in sorted((RES / "l40").glob("out_*.jsonl")):
    for r in read(p):
      if r["arm"] == "tfilm_nl":
        lane = r.get("lane", "")
        WORDS[(int(r["seed"]), int(r["held_task"]))] = dict(r, _m="ws2" if lane.startswith("ws2") else ("local" if lane == "local" else "a800"))
  for p in sorted((RES / "tin").glob("outt_l40_*.jsonl")):
    for r in read(p):
      if r["arm"] == "plain_nl":
        NONE[(int(r["seed"]), int(r["held_task"]))] = dict(r, _m="ws2" if r["host"].startswith("workstation") else "a800")
  LT = list(range(1000, 1040))
  done = [s for s in SEEDS if all((s, t) in B for t in LT)]
  say("== Part B: text embedding on MSAL's head, LIBERO40. records %d; complete seeds %s" % (len(B), done))

  def mach(r):
    return "ws2" if r["host"].startswith("workstation") else "a800"

  def b_reading(name, a, b, same=None):
    out = {}
    for label, pick in (("all", lambda k: True), ("same", same)):
      if pick is None:
        continue
      lr = defaultdict(list)
      for s in done:
        for t in LT:
          if pick((s, t)):
            lr[s].append(math.log(a((s, t)) / b((s, t))))
      if lr:
        out[label] = tci([float(np.mean(v)) for v in lr.values()])
        out[label].update(pairs=int(sum(len(v) for v in lr.values())), lower=int(sum(sum(1 for z in v if z < 0) for v in lr.values())))
    REPORT[name] = out
    r = out["all"]
    say("  %-52s %+6.1f%% [%+6.1f, %+6.1f] %-19s folds %d/%d per seed %s" % (name, r["pct"], r["lo"], r["hi"], word(r),
      r["lower"], r["pairs"], r.get("per_seed")))
    if "same" in out:
      r = out["same"]
      say("  %-52s %+6.1f%% [%+6.1f, %+6.1f] %-19s folds %d/%d (%d seeds)" % ("    same-machine pairs", r["pct"], r["lo"],
        r["hi"], word(r), r["lower"], r["pairs"], r["seeds"]))

  if U4:
    rep_ = [k for k in BALL if len(BALL[k]) == 2]
    say("  amendment 2: %d units of the server's list, %d read from the server; %d units run on both machines" % (len(U4),
      sum(1 for k in U4 if k in B), len(rep_)))
    for f in ("action_selection", "flow_selection"):
      lr = [math.log(BALL[k]["ws2"][f] / BALL[k]["a800"][f]) for k in rep_]
      if lr:
        say("    descriptive repeat, ws2 against server, %s: mean %+.1f%%, sd %.1f%%, max abs %.1f%%" % (f, pct(float(np.mean(lr))),
          100 * float(np.std(lr)), 100 * max(abs(z) for z in lr)))
        REPORT["B repeat " + f] = dict(pairs=len(lr), pct=pct(float(np.mean(lr))))
  if done:
    io = [B[(s, t)]["iota"] for s in done for t in LT]
    say("  span share of a held descriptor: mean %.3f (%.3f to %.3f); units per machine %s" % (np.mean(io), min(io), max(io),
      dict((m, sum(mach(B[(s, t)]) == m for s in done for t in LT)) for m in ("ws2", "a800"))))
    for metric, field, wfield in (("sampled chunk", "action_selection", None), ("flow", "flow_selection", None)):
      e = lambda k, f=field: float(B[k][f])
      if field == "action_selection":
        n = lambda k: float(NONE[k]["action"]["zero_shot"])
        w = lambda k: float(WORDS[k]["action"]["zero_shot"])
      else:
        n = lambda k: float(NONE[k]["zero_shot"]["selection"] if "zero_shot" in NONE[k] else NONE[k]["score_forward"]["selection"])
        w = lambda k: float(WORDS[k]["zero_shot"]["selection"])
      b_reading("T1' embedding vs none, %s" % metric, e, n, lambda k: mach(B[k]) == NONE[k]["_m"])
      b_reading("T2' embedding vs words (same head), %s" % metric, e, w, lambda k: mach(B[k]) == WORDS[k]["_m"])
      b_reading("T3' in-span vs whole, %s" % metric, lambda k, f=field: float(B[k]["proj_" + f]), e)
      b_reading("T4' whole vs constant-only, %s" % metric, e, lambda k, f=field: float(B[k]["mean_" + f]))
      b_reading("    words vs none (stored, same head), %s" % metric, w, n, lambda k: WORDS[k]["_m"] == NONE[k]["_m"])

for p in groups["bench"]:
  say("== Part D: CPU time of adaptation (four threads; median of three)")
  rows = read(p)
  for arm, label in (("film_nl", "MSAL-h"), ("tfilm_nl", "MSAL")):
    for r in sorted((x for x in rows if x["arm"] == arm), key=lambda x: x["k"]):
      say("  %-7s k=%2d rows %4d median %.2f s (runs %s), peak extra memory %.2f GB" % (label, r["k"], r["rows"],
        r["median_seconds"], [round(v, 2) for v in r["seconds"]], r["peak_extra_mb"] / 1024))
  REPORT["D"] = rows
Path(PREFIX + ".json").write_text(json.dumps(REPORT, indent=1))
Path(PREFIX + ".txt").write_text("\n".join(LINES) + "\n")
