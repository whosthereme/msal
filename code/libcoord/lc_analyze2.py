import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

args = sys.argv[1:]
cut = args.index("--")
old_dir, new_dir, offline, prefix = Path(args[0]), Path(args[1]), [Path(p) for p in args[2:cut]], args[cut + 1]
B, SEED = 10_000, 20261119


def read(d):
  return [json.loads(x) for p in sorted(d.glob("held_*.jsonl")) for x in p.read_text().splitlines() if x.strip()]


held = [r for r in read(old_dir) if r["policy"] in ("C0", "N0", "H0")] + read(new_dir)


def table(key):
  t = defaultdict(lambda: defaultdict(dict))
  for r in held:
    t[r["policy"]][r["task"]][(r["seed"], r["episode"])] = float(r[key])
  return t


def paired(a, b, key, index, stat="mean"):
  t = table(key)
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


out = dict(records=len(held), counts={p: sum(r["policy"] == p for r in held) for p in sorted(set(r["policy"] for r in held))})
READ = (("H1", "C0", "F0"), ("H2", "C0", "M0"), ("H3", "T0", "C0"), ("H4_F", "F0", "N0"), ("H4_M", "M0", "N0"),
  ("H4_T", "T0", "N0"), ("H5", "G0", "C0"), ("H6", "G0", "N0"))
for i, (name, a, b) in enumerate(READ, 1):
  out[name] = paired(a, b, "success", i)
  out[name + "_dist"] = paired(a, b, "final_dist", 20 + i, stat="median")

iota = {}
off = defaultdict(lambda: defaultdict(list))
for p in offline:
  for x in p.read_text().splitlines():
    if x.strip():
      r = json.loads(x)
      off[r["arm"]][(r["seed"], r["task"])].append(r)
      if r["arm"] == "lc_h20":
        iota[r["task"]] = r["iota"]
g = defaultdict(list)
for r in held:
  if r["policy"] == "G0":
    g[r["task"]].append(float(r["success"]))
tasks = sorted(set(g) & set(iota))
if len(tasks) > 3:
  x, y = np.array([iota[t] for t in tasks]), np.array([np.mean(g[t]) for t in tasks])
  rho = float(stats.spearmanr(x, y).statistic)
  rng = np.random.default_rng([SEED, 99])
  bs = []
  for _ in range(20_000):
    i = rng.integers(len(tasks), size=len(tasks))
    if len(set(x[i])) > 1 and len(set(y[i])) > 1:
      bs.append(stats.spearmanr(x[i], y[i]).statistic)
  out["G0_success_vs_share"] = dict(rho=rho, ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
    tasks=len(tasks), share_mean=float(x.mean()), share_min=float(x.min()), share_max=float(x.max()))

desc = {}
for arm, recs in off.items():
  rows = [r[-1] for r in recs.values()]
  desc[arm] = dict(n=len(rows), flow=float(np.mean([r["zero_shot"]["selection"] for r in rows])),
    chunk=float(np.mean([r["action_selection"] for r in rows])), share=float(np.mean([r["iota"] for r in rows])))
out["offline"] = desc
flow = defaultdict(dict)
for p in sorted(old_dir.parent.glob("heads/offline*.jsonl")):
  for x in p.read_text().splitlines():
    if x.strip():
      r = json.loads(x)
      if r.get("held"):
        flow[r["arm"]][(r["seed"], r["task"])] = r["zero_shot"]["selection"]
for arm, recs in off.items():
  for key, rr in recs.items():
    flow[arm][key] = rr[-1]["zero_shot"]["selection"]
ratio = {}
for arm in sorted(off):
  for ref in ("lc_xy6", "lc_plain"):
    keys = sorted(set(flow[arm]) & set(flow[ref]))
    seeds = sorted(set(k[0] for k in keys))
    if len(seeds) < 2:
      continue
    m = np.array([np.mean([math.log(flow[arm][k] / flow[ref][k]) for k in keys if k[0] == s]) for s in seeds])
    h = float(stats.t.ppf(0.975, len(m) - 1)) * float(m.std(ddof=1)) / math.sqrt(len(m))
    pc = lambda v: 100.0 * (math.exp(v) - 1.0)
    ratio["%s vs %s" % (arm, ref)] = dict(pct=pc(m.mean()), lo=pc(m.mean() - h), hi=pc(m.mean() + h), seeds=len(m),
      pairs=len(keys))
out["offline_flow"] = ratio
pol = {}
for p in sorted(out["counts"]):
  rows = [r for r in held if r["policy"] == p]
  pol[p] = dict(n=len(rows), success=float(np.mean([r["success"] for r in rows])),
    median_final_dist=float(np.median([r["final_dist"] for r in rows])),
    by_fold={f: float(np.mean([r["success"] for r in rows if r["fold"] == f])) for f in range(4)},
    probe_max=max((r.get("probe_max_abs") or 0.0) for r in rows))
out["policies"] = pol
Path(prefix + ".json").write_text(json.dumps(out, indent=1))
lines = ["records %d %s" % (out["records"], out["counts"])]
for name, a, b in READ:
  for suffix in ("", "_dist"):
    v = out[name + suffix]
    lines.append("%-9s %s vs %s (%s %s): %.3f vs %.3f, diff %+.3f [%+.3f, %+.3f] -> %s  (targets %d, units %d)" % (
      name + suffix, v["a"], v["b"], v["key"], v["stat"], v["a_mean"], v["b_mean"], v["diff"], v["ci"][0], v["ci"][1],
      v["reading"] if not suffix else ("closer" if v["ci"][1] < 0 else ("farther" if v["ci"][0] > 0 else "n.d.")),
      v["tasks"], v["units"]))
if "G0_success_vs_share" in out:
  v = out["G0_success_vs_share"]
  lines.append("G0 success vs span share: rho %+.2f [%+.2f, %+.2f] over %d targets (share mean %.2f, %.2f to %.2f)" % (
    v["rho"], v["ci"][0], v["ci"][1], v["tasks"], v["share_mean"], v["share_min"], v["share_max"]))
for p, d in pol.items():
  lines.append("%-3s n=%d success %.3f median dist %.3f folds %s probe %.1e" % (p, d["n"], d["success"],
    d["median_final_dist"], {f: round(x, 3) for f, x in d["by_fold"].items()}, d["probe_max"]))
for a, d in desc.items():
  lines.append("offline %-12s n=%d flow %.4f sampled chunk %.4f share %.3f" % (a, d["n"], d["flow"], d["chunk"], d["share"]))
for k, v in ratio.items():
  lines.append("offline flow %-24s %+6.1f%% [%+6.1f, %+6.1f] (%d seeds, %d pairs)" % (k, v["pct"], v["lo"], v["hi"],
    v["seeds"], v["pairs"]))
Path(prefix + ".txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
