import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

args = sys.argv[1:]
prefix = args[0]
groups, cur = dict(text=[], tin=[], l40=[]), None
for a in args[1:]:
  if a.startswith("--"):
    cur = a[2:]
  else:
    groups[cur].append(Path(a))


def read(paths):
  out = []
  for p in paths:
    for line in p.read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        r["_file"] = str(p).replace("\\", "/")
        out.append(r)
  return out


def machine_new(r):
  return "ws2" if r["host"].startswith("workstation") else "a800"


def machine_stored(r):
  f, lane = r["_file"], r.get("lane", "")
  if "ws2" in f.split("/")[-1] or lane.startswith("ws2"):
    return "ws2"
  if "local" in f or lane == "local":
    return "local"
  return "a800"


E = dict(((int(r["seed"]), int(r["held_task"])), r) for r in read(groups["text"]))
N = dict(((int(r["seed"]), int(r["held_task"])), r) for r in read(groups["tin"]) if r["arm"] == "plain_nl")
W = dict(((int(r["seed"]), int(r["held_task"])), r) for r in read(groups["l40"])
  if r["arm"] == "film_nl" and "anchor" not in r["_file"])
for r in read(groups["l40"]):
  if r["arm"] == "film_nl" and "anchor" in r["_file"]:
    W.setdefault((int(r["seed"]), int(r["held_task"])), r)
LINES, REPORT = [], {}


def say(s=""):
  LINES.append(s)
  print(s)


def pct(v):
  return 100.0 * (math.exp(v) - 1.0)


def tci(means):
  v = np.asarray(means, dtype=float)
  m = float(v.mean())
  if len(v) < 2:
    return dict(n=len(v), pct=pct(m), lo=float("nan"), hi=float("nan"))
  h = float(stats.t.ppf(0.975, len(v) - 1)) * float(v.std(ddof=1)) / math.sqrt(len(v))
  return dict(n=len(v), pct=pct(m), lo=pct(m - h), hi=pct(m + h))


def word(r):
  if r["lo"] != r["lo"]:
    return "one seed"
  return "lower" if r["hi"] < 0 else ("higher" if r["lo"] > 0 else "not distinguishable")


def reading(name, a, b, same=None, keys=None):
  rows = [(k, a(k), b(k)) for k in sorted(keys if keys is not None else E)]
  rows = [(k, x, y) for k, x, y in rows if x is not None and y is not None]
  entry = {}
  for label, pick in (("all", lambda k: True), ("same", same)):
    if pick is None:
      continue
    lr = {}
    for k, x, y in rows:
      if pick(k):
        lr.setdefault(k[0], []).append(math.log(x / y))
    if lr:
      entry[label] = tci([float(np.mean(v)) for v in lr.values()])
      entry[label].update(pairs=int(sum(len(v) for v in lr.values())), won=int(sum(sum(1 for z in v if z < 0) for v in lr.values())))
  if "all" not in entry:
    say("  %-52s no pairs yet" % name)
    return None
  REPORT[name] = entry
  for label in entry:
    r = entry[label]
    say("  %-52s %+6.1f%% [%+6.1f, %+6.1f] %-19s (%d seeds, folds %d/%d)" % (name if label == "all" else "    same-machine pairs",
      r["pct"], r["lo"], r["hi"], word(r), r["n"], r["won"], r["pairs"]))
  return rows


get = lambda src, f: (lambda k: None if src.get(k) is None else f(src[k]))
sameN = lambda k: k in N and machine_new(E[k]) == machine_new(N[k])
sameW = lambda k: k in W and machine_new(E[k]) == machine_stored(W[k])
say("text records %d (machines %s); plain_nl records %d; stored word-descriptor records %d" % (len(E),
  dict((m, sum(1 for r in E.values() if machine_new(r) == m)) for m in ("ws2", "a800")), len(N), len(W)))
if E:
  io = np.array([r["iota"] for r in E.values()])
  say("span share of the held descriptor: mean %.3f, min %.3f, max %.3f; scorer check %d/%d" % (io.mean(), io.min(), io.max(),
    sum(bool(r["raw_matches_stored_scorer"]) for r in E.values()), len(E)))
say("T1: text embedding vs no task input")
t1 = reading("T1 sampled chunk", get(E, lambda r: r["action_selection"]), get(N, lambda r: r["action"]["zero_shot"]), sameN)
reading("T1 flow", get(E, lambda r: r["flow_selection"]), get(N, lambda r: r["score_forward"]["selection"]), sameN)
say("T2: text embedding vs bag of words (stored MSAL head)")
reading("T2 sampled chunk", get(E, lambda r: r["action_selection"]), get(W, lambda r: r["action"]["zero_shot"]), sameW)
reading("T2 flow", get(E, lambda r: r["flow_selection"]), get(W, lambda r: r["zero_shot"]["selection"]), sameW)
say("T3: in-span part only vs whole descriptor (same model)")
reading("T3 sampled chunk", get(E, lambda r: r["proj_action_selection"]), get(E, lambda r: r["action_selection"]))
reading("T3 flow", get(E, lambda r: r["proj_flow_selection"]), get(E, lambda r: r["flow_selection"]))
say("T4: whole descriptor vs constant-only descriptor (same model)")
reading("T4 sampled chunk", get(E, lambda r: r["action_selection"]), get(E, lambda r: r["mean_action_selection"]))
reading("T4 flow", get(E, lambda r: r["flow_selection"]), get(E, lambda r: r["mean_flow_selection"]))
say("reference: bag of words vs no task input (stored head, new plain_nl)")
sameWN = lambda k: k in W and k in N and machine_new(N[k]) == machine_stored(W[k])
reading("W vs N sampled chunk", get(W, lambda r: r["action"]["zero_shot"]), get(N, lambda r: r["action"]["zero_shot"]), sameWN)
reading("W vs N flow", get(W, lambda r: r["zero_shot"]["selection"]), get(N, lambda r: r["score_forward"]["selection"]), sameWN)

tasks = sorted(set(k[1] for k in E))
for metric, fe, fn in (("sampled chunk", lambda r: r["action_selection"], lambda r: r["action"]["zero_shot"]),
    ("flow", lambda r: r["flow_selection"], lambda r: r["score_forward"]["selection"])):
  x, y = [], []
  for t in tasks:
    ks = [k for k in E if k[1] == t and k in N]
    if ks:
      x.append(float(np.mean([E[k]["iota"] for k in ks])))
      y.append(float(np.mean([math.log(fe(E[k]) / fn(N[k])) for k in ks])))
  if len(x) > 5:
    x, y = np.array(x), np.array(y)
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(20_000):
      i = rng.integers(len(x), size=len(x))
      bs.append(stats.spearmanr(x[i], y[i]).statistic)
    lo, hi = float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))
    REPORT["T5 " + metric] = dict(rho=rho, lo=lo, hi=hi, tasks=len(x))
    say("T5 %s: Spearman of span share and log(E/N) over %d tasks: %+.2f [%+.2f, %+.2f] %s" % (metric, len(x), rho, lo, hi,
      "supported" if hi < 0 else "not supported"))
    med = float(np.median(x))
    hi_t = set(t for t, v in zip([t for t in tasks if any(k[1] == t and k in N for k in E)], x) if v >= med)
    for label, pick in (("share at or above the median %.2f" % med, lambda k: k[1] in hi_t), ("share below the median", lambda k: k[1] not in hi_t)):
      reading("T1 %s, %s" % (metric, label), get(E, fe), get(N, fn), keys=[k for k in E if pick(k)])
      reading("T3 %s, %s" % (metric, label), get(E, lambda r: r["proj_action_selection"] if metric == "sampled chunk" else r["proj_flow_selection"]),
        get(E, fe), keys=[k for k in E if pick(k)])
Path(prefix + ".json").write_text(json.dumps(REPORT, indent=1))
Path(prefix + ".txt").write_text("\n".join(LINES) + "\n")
