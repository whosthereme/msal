import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

args = sys.argv[1:]
prefix = args[0]
groups, cur = dict(new=[], dev=[], l40=[]), None
for a in args[1:]:
  if a.startswith("--"):
    cur = a[2:]
  else:
    groups[cur].append(Path(a))
SEEDS = (148021, 148022, 148023, 148024, 148025)


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


NEW, ANCHOR, STORED = {}, {}, {}
for r in read(groups["new"]):
  key = (r["family"], int(r["seed"]), int(r["held_task"]))
  (ANCHOR if r["arm"] == "film_nl" else NEW).setdefault(r["arm"], {})[key] = r
for fam in ("dev", "l40"):
  for r in read(groups[fam]):
    if r["arm"] == "film_nl" and "anchor" not in r["_file"]:
      STORED[(fam, int(r["seed"]), int(r["held_task"]))] = r
  for r in read(groups[fam]):
    if r["arm"] == "film_nl" and "anchor" in r["_file"]:
      STORED.setdefault((fam, int(r["seed"]), int(r["held_task"])), r)
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


def fmt(r):
  return "%+6.1f%% [%+6.1f, %+6.1f] %-19s (%d seeds)" % (r["pct"], r["lo"], r["hi"], word(r), r["n"])


def flow(r, stage, k=None):
  if stage == "zero":
    return r["zero_shot"]["selection"] if "zero_shot" in r else r["score_forward"]["selection"]
  return r[stage][str(k)]["selection"]


def act(r, key):
  return r["action"][key]


def reading(name, fam, a, b, same=None):
  keys = sorted(k for k in NEW.get("tin_nl", {}) if k[0] == fam)
  rows = [(k, a(k), b(k)) for k in keys]
  rows = [(k, x, y) for k, x, y in rows if x is not None and y is not None]
  entry = dict(pairs=len(rows))
  for label, pick in (("all", lambda k: True), ("same", same)):
    if pick is None:
      continue
    lr = {}
    for k, x, y in rows:
      if pick(k):
        lr.setdefault(k[1], []).append(math.log(x / y))
    if lr:
      entry[label] = tci([float(np.mean(v)) for v in lr.values()])
      entry[label]["pairs"] = int(sum(len(v) for v in lr.values()))
      entry[label]["won"] = int(sum(sum(1 for z in v if z < 0) for v in lr.values()))
  if "all" not in entry:
    say("  %-58s no pairs yet" % name)
    return
  use = entry["all"]
  if "same" in entry and entry["same"]["pct"] * entry["all"]["pct"] < 0:
    use = entry["same"]
  entry["verdict"] = word(use)
  REPORT["%s %s" % (fam, name)] = entry
  say("  %-58s %s folds %d/%d" % (name, fmt(entry["all"]), entry["all"]["won"], entry["all"]["pairs"]))
  if "same" in entry:
    say("  %-58s %s folds %d/%d   -> %s" % ("    same-machine pairs", fmt(entry["same"]), entry["same"]["won"],
      entry["same"]["pairs"], entry["verdict"]))


for fam, title in (("l40", "LIBERO40"), ("dev", "development family")):
  tin = NEW.get("tin_nl", {})
  n = sum(1 for k in tin if k[0] == fam)
  if not n:
    continue
  say("== %s: %d tin_nl units; machines %s" % (title, n, dict((m, sum(1 for k, r in tin.items() if k[0] == fam and
    machine_new(r) == m)) for m in ("ws2", "a800"))))
  T = lambda k: tin.get(k)
  S = lambda k: STORED.get(k)
  P = lambda k: NEW.get("plain_nl", {}).get(k)
  same = lambda k: S(k) is not None and machine_new(tin[k]) == machine_stored(S(k))
  g = lambda src, f: (lambda k: None if src(k) is None else f(src(k)))
  say(" zero-shot (D1 / L1): input concatenation vs MSAL's head")
  reading("sampled chunk", fam, g(T, lambda r: act(r, "zero_shot")), g(S, lambda r: act(r, "zero_shot")), same)
  reading("flow", fam, g(T, lambda r: flow(r, "zero")), g(S, lambda r: flow(r, "zero")), same)
  if fam == "l40":
    say(" what the description adds (L2), against the same readout without task input")
    reading("input concatenation vs none, sampled chunk", fam, g(T, lambda r: act(r, "zero_shot")), g(P, lambda r: act(r, "zero_shot")))
    reading("input concatenation vs none, flow", fam, g(T, lambda r: flow(r, "zero")), g(P, lambda r: flow(r, "zero")))
    reading("MSAL's head vs none, sampled chunk", fam, g(S, lambda r: act(r, "zero_shot")), g(P, lambda r: act(r, "zero_shot")),
      lambda k: S(k) is not None and P(k) is not None and machine_new(P(k)) == machine_stored(S(k)))
    reading("MSAL's head vs none, flow", fam, g(S, lambda r: flow(r, "zero")), g(P, lambda r: flow(r, "zero")),
      lambda k: S(k) is not None and P(k) is not None and machine_new(P(k)) == machine_stored(S(k)))
  say(" complete method (D2 / L3): concatenation + inversion + residual code vs MSAL")
  for k_ in (1, 16):
    reading("sampled chunk, k=%d" % k_, fam, g(T, lambda r, k_=k_: act(r, "combination_%d" % k_)),
      g(S, lambda r, k_=k_: act(r, "combination_%d" % k_)), same)
    reading("flow, k=%d" % k_, fam, g(T, lambda r, k_=k_: flow(r, "combination", k_)),
      g(S, lambda r, k_=k_: flow(r, "combination", k_)), same)
  say(" on the concatenation head (D3 / L4 and descriptive)")
  for k_ in (1, 16):
    reading("residual code beyond inversion, flow, k=%d" % k_, fam, g(T, lambda r, k_=k_: flow(r, "combination", k_)),
      g(T, lambda r, k_=k_: flow(r, "inversion", k_)))
    reading("residual code beyond inversion, sampled chunk, k=%d" % k_, fam,
      g(T, lambda r, k_=k_: act(r, "combination_%d" % k_)), g(T, lambda r, k_=k_: act(r, "inversion_%d" % k_)))
    reading("method vs own zero-shot, sampled chunk, k=%d" % k_, fam, g(T, lambda r, k_=k_: act(r, "combination_%d" % k_)),
      g(T, lambda r: act(r, "zero_shot")))
    reading("method vs own zero-shot, flow, k=%d" % k_, fam, g(T, lambda r, k_=k_: flow(r, "combination", k_)),
      g(T, lambda r: flow(r, "zero")))
    reading("MSAL vs own zero-shot (stored), sampled chunk, k=%d" % k_, fam,
      g(S, lambda r, k_=k_: act(r, "combination_%d" % k_)), g(S, lambda r: act(r, "zero_shot")))
  for k_ in (1, 16):
    reading("inversion: concatenation vs MSAL's head, sampled chunk, k=%d" % k_, fam,
      g(T, lambda r, k_=k_: act(r, "inversion_%d" % k_)), g(S, lambda r, k_=k_: act(r, "inversion_%d" % k_)), same)
  shifts = [r["combination"]["16"]["descriptor_shift"] for k, r in tin.items() if k[0] == fam]
  sh0 = [STORED[k]["combination"]["16"]["descriptor_shift"] for k in tin if k[0] == fam and k in STORED]
  means = dict(tin_chunk=float(np.mean([act(r, "zero_shot") for k, r in tin.items() if k[0] == fam])),
    tin_flow=float(np.mean([flow(r, "zero") for k, r in tin.items() if k[0] == fam])),
    shift_tin=float(np.mean(shifts)), shift_msal=float(np.mean(sh0)) if sh0 else None)
  REPORT["%s means" % fam] = means
  say("  mean zero-shot sampled chunk %.5f, flow %.5f; descriptor shift by inversion at k=16: concatenation %.2f, MSAL %s" % (
    means["tin_chunk"], means["tin_flow"], means["shift_tin"], "%.2f" % means["shift_msal"] if sh0 else "n/a"))

say("== anchors: film_nl retrained in the new runs vs its stored record")
for key, r in sorted(ANCHOR.get("film_nl", {}).items()):
  s = STORED.get(key)
  if s is None:
    say("  %s seed %d task %d: no stored record" % key)
    continue
  exact = flow(r, "zero") == flow(s, "zero") and act(r, "combination_16") == act(s, "combination_16")
  say("  %s seed %d task %d: new on %s, stored on %s, zero-shot flow %.6f vs %.6f, bit-exact %s" % (key[0], key[1], key[2],
    machine_new(r), machine_stored(s), flow(r, "zero"), flow(s, "zero"), exact))
  REPORT.setdefault("anchors", []).append(dict(key=list(key), new=machine_new(r), stored=machine_stored(s), exact=bool(exact)))
Path(prefix + ".json").write_text(json.dumps(REPORT, indent=1))
Path(prefix + ".txt").write_text("\n".join(LINES) + "\n")
