import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

a = sys.argv[1:]
old_dir, mid_dir, k_dir, new_dir, off_k, off_j, off_a, prefix = [Path(x) for x in a[:7]] + [a[7]]
B, SEED = 10_000, 20261122


def read_dir(d, pattern="*.jsonl"):
  return [json.loads(x) for p in sorted(d.glob(pattern)) for x in p.read_text().splitlines() if x.strip()]


def read(p):
  return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


new = read_dir(new_dir)
held = ([r for r in read_dir(old_dir, "held_*.jsonl") if r["policy"] in ("C0", "C1", "N0")] + read_dir(mid_dir, "held_*.jsonl")
  + read_dir(k_dir, "held_*.jsonl") + [r for r in new if r["set"] == "held"])
seen = [r for r in new if r["set"] == "seen"] + [dict(r, policy="C0") for r in read_dir(old_dir) if r["set"] == "seen" and r["policy"] == "C0"]


def table(rows, key):
  t = defaultdict(lambda: defaultdict(dict))
  for r in rows:
    t[r["policy"]][r["task"]][(r["seed"], r["episode"])] = float(r[key])
  return t


def paired(a, b, key, index, stat="mean"):
  t = table(held, key)
  tasks = sorted(set(t[a]) & set(t[b]))
  units = {k: sorted(set(t[a][k]) & set(t[b][k])) for k in tasks}
  A = [np.array([t[a][k][u] for u in units[k]]) for k in tasks]
  Bv = [np.array([t[b][k][u] for u in units[k]]) for k in tasks]
  f = np.mean if stat == "mean" else np.median

  def est(ti, ui):
    return float(np.mean([f(A[i][u]) - f(Bv[i][u]) for i, u in zip(ti, ui)]))

  point = est(range(len(tasks)), [np.arange(len(units[k])) for k in tasks])
  rng = np.random.default_rng([SEED, index])
  boot = np.empty(B)
  for j in range(B):
    ti = rng.integers(len(tasks), size=len(tasks))
    boot[j] = est(ti, [rng.integers(len(A[i]), size=len(A[i])) for i in ti])
  lo, hi = np.percentile(boot, [2.5, 97.5])
  return dict(a=a, b=b, key=key, stat=stat, tasks=len(tasks), units=int(sum(len(u) for u in units.values())),
    a_mean=float(np.mean([f(x) for x in A])), b_mean=float(np.mean([f(x) for x in Bv])), diff=point,
    ci=[float(lo), float(hi)], reading="better" if lo > 0 else ("worse" if hi < 0 else "not distinguishable"))


def unpaired(policy, key, index):
  ta, tb = table(held, key)[policy], table(seen, key)[policy]
  A = [np.array(list(ta[k].values())) for k in sorted(ta)]
  Bv = [np.array(list(tb[k].values())) for k in sorted(tb)]
  rng = np.random.default_rng([SEED, index])

  def two_level(G):
    ti = rng.integers(len(G), size=len(G))
    return float(np.mean([np.mean(G[i][rng.integers(len(G[i]), size=len(G[i]))]) for i in ti]))

  point = float(np.mean([x.mean() for x in A]) - np.mean([x.mean() for x in Bv]))
  boot = np.array([two_level(A) - two_level(Bv) for _ in range(B)])
  lo, hi = np.percentile(boot, [2.5, 97.5])
  return dict(a=policy + " held", b=policy + " trained", key=key, tasks=[len(A), len(Bv)], diff=point,
    units=[int(sum(len(x) for x in A)), int(sum(len(x) for x in Bv))],
    a_mean=float(np.mean([x.mean() for x in A])), b_mean=float(np.mean([x.mean() for x in Bv])),
    ci=[float(lo), float(hi)], reading="better" if lo > 0 else ("worse" if hi < 0 else "not distinguishable"))


out = dict(counts_held={p: sum(r["policy"] == p for r in held) for p in sorted(set(r["policy"] for r in held))},
  counts_trained={p: sum(r["policy"] == p for r in seen) for p in sorted(set(r["policy"] for r in seen))})
lines = ["held %s" % out["counts_held"], "trained %s" % out["counts_trained"]]
READ = (("H10", "J0", "K0", 1), ("H11", "J0", "N0", 2), ("H12", "K1", "K0", 3))
have = set(out["counts_held"])
for name, x, y, i in READ:
  if x in have and y in have:
    out[name] = paired(x, y, "success", i)
    out[name + "_dist"] = paired(x, y, "final_dist", 20 + i, stat="median")
    for suffix in ("", "_dist"):
      v = out[name + suffix]
      scale = 100.0
      lines.append("%-9s %s vs %s (%s %s): %.1f vs %.1f, diff %+.1f [%+.1f, %+.1f] -> %s  (targets %d, units %d)" % (
        name + suffix, v["a"], v["b"], v["key"], v["stat"], scale * v["a_mean"], scale * v["b_mean"], scale * v["diff"],
        scale * v["ci"][0], scale * v["ci"][1],
        v["reading"] if not suffix else ("closer" if v["ci"][1] < 0 else ("farther" if v["ci"][0] > 0 else "n.d.")),
        v["tasks"], v["units"]))
if "K0" in have and "K0" in out["counts_trained"]:
  v = out["H13"] = unpaired("K0", "success", 4)
  lines.append("H13       K0 held vs trained targets: %.1f vs %.1f, diff %+.1f [%+.1f, %+.1f] -> %s (targets %s, units %s)" % (
    100 * v["a_mean"], 100 * v["b_mean"], 100 * v["diff"], 100 * v["ci"][0], 100 * v["ci"][1], v["reading"], v["tasks"], v["units"]))
  if "C0" in out["counts_trained"]:
    w = out["S3_again"] = unpaired("C0", "success", 5)
    lines.append("  (stored C0 held vs trained, this seed stream: %.1f vs %.1f, diff %+.1f [%+.1f, %+.1f])" % (100 * w["a_mean"],
      100 * w["b_mean"], 100 * w["diff"], 100 * w["ci"][0], 100 * w["ci"][1]))
for i, (x, y) in enumerate((("K1", "C1"), ("K1", "T0"), ("J0", "G0"), ("K1", "N0")), 6):
  if x in have and y in have:
    v = out["post hoc %s vs %s" % (x, y)] = paired(x, y, "success", i)
    lines.append("post hoc  %s vs %s: %.1f vs %.1f, diff %+.1f [%+.1f, %+.1f] -> %s" % (x, y, 100 * v["a_mean"], 100 * v["b_mean"],
      100 * v["diff"], 100 * v["ci"][0], 100 * v["ci"][1], v["reading"]))

pol = {}
for group, rows in (("held", held), ("trained", seen)):
  for p in sorted(set(r["policy"] for r in rows)):
    rr = [r for r in rows if r["policy"] == p]
    pol["%s %s" % (p, group)] = d = dict(n=len(rr), success=float(np.mean([r["success"] for r in rr])),
      median_final_dist=float(np.median([r["final_dist"] for r in rr])),
      by_fold={f: float(np.mean([r["success"] for r in rr if r["fold"] == f])) for f in range(4)},
      by_seed={s: float(np.mean([r["success"] for r in rr if r["seed"] == s])) for s in (148021, 148022, 148023)},
      by_ring={g: float(np.mean([r["success"] for r in rr if r["ring"] == g])) for g in (0, 1, 2)},
      probe_max=max((r.get("probe_max_abs") or 0.0) for r in rr))
    lines.append("%-10s n=%4d success %5.1f%% median dist %.3f folds %s seeds %s rings %s probe %.1e" % (p + " " + group, d["n"],
      100 * d["success"], d["median_final_dist"], [round(100 * x, 1) for x in d["by_fold"].values()],
      [round(100 * x, 1) for x in d["by_seed"].values()], [round(100 * x, 1) for x in d["by_ring"].values()], d["probe_max"]))
out["policies"] = pol

pc = lambda v: 100.0 * (math.exp(v) - 1.0)


def tci(m):
  m = np.asarray(m, dtype=float)
  h = float(stats.t.ppf(0.975, len(m) - 1)) * float(m.std(ddof=1)) / math.sqrt(len(m)) if len(m) > 1 else float("nan")
  return dict(pct=pc(m.mean()), lo=pc(m.mean() - h), hi=pc(m.mean() + h), seeds=len(m))


K = dict(((r["seed"], r["task"]), r) for r in read(off_k))
J = dict(((r["seed"], r["task"]), r) for r in read(off_j) if r["arm"] == "lc_tfilm_h20")
AD = dict(((r["seed"], r["task"]), r) for r in read(off_a) if r["held"])
off = {}
if J and K:
  keys = sorted(set(J) & set(K))
  seeds = sorted(set(k[0] for k in keys))
  for label, get in (("sampled chunk", lambda r: r["action_selection"]), ("flow", lambda r: r["zero_shot"]["selection"])):
    off["harmonic vs coordinates, %s" % label] = dict(tci([np.mean([math.log(get(J[k]) / get(K[k])) for k in keys if k[0] == s])
      for s in seeds]), pairs=len(keys))
  off["harmonic heads"] = dict(n=len(J), flow=float(np.mean([r["zero_shot"]["selection"] for r in J.values()])),
    chunk=float(np.mean([r["action_selection"] for r in J.values()])), iota=float(np.mean([r["iota"] for r in J.values()])),
    iota_min=float(min(r["iota"] for r in J.values())), iota_max=float(max(r["iota"] for r in J.values())))
if AD:
  keys = sorted(AD)
  seeds = sorted(set(k[0] for k in keys))
  for label, a_, b_ in (("adapted (k=1) vs zero-shot, flow", lambda r: r["msal_k1"]["selection"], lambda r: r["zero_shot"]["selection"]),
      ("inversion (k=1) vs zero-shot, flow", lambda r: r["inversion_k1"]["selection"], lambda r: r["zero_shot"]["selection"]),
      ("adapted (k=1) vs zero-shot, sampled chunk", lambda r: r["action"]["msal_k1"], lambda r: r["action"]["zero_shot"])):
    off[label] = dict(tci([np.mean([math.log(a_(AD[k]) / b_(AD[k])) for k in keys if k[0] == s]) for s in seeds]), pairs=len(keys))
  off["adapted heads"] = dict(n=len(AD), descriptor_shift=float(np.mean([r["msal_k1"]["descriptor_shift"] for r in AD.values()])),
    code_norm=float(np.mean([r["msal_k1"]["code_norm"] for r in AD.values()])))
  if K:
    same = [AD[k]["zero_shot"]["selection"] == K[k]["zero_shot"]["selection"] for k in keys if k in K]
    off["adapted heads"]["zero_shot_flow_equals_stored"] = [int(sum(same)), len(same)]
out["offline"] = off
for k, v in off.items():
  if "pct" in v:
    lines.append("offline %-44s %+6.1f%% [%+6.1f, %+6.1f] (%d seeds, %d pairs)" % (k, v["pct"], v["lo"], v["hi"], v["seeds"], v["pairs"]))
  else:
    lines.append("offline %-44s %s" % (k, v))
Path(prefix + ".json").write_text(json.dumps(out, indent=1))
Path(prefix + ".txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
