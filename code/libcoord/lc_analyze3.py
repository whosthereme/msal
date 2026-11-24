import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

old_dir, mid_dir, new_dir, off_t, off_k, prefix = [Path(a) for a in sys.argv[1:6]] + [sys.argv[6]]
B, SEED = 10_000, 20261121


def read(d):
  return [json.loads(x) for p in sorted(d.glob("held_*.jsonl")) for x in p.read_text().splitlines() if x.strip()]


held = [r for r in read(old_dir) if r["policy"] in ("C0", "N0")] + read(mid_dir) + read(new_dir)


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
READ = (("H7", "K0", "T0"), ("H8", "K0", "C0"), ("H9", "K0", "N0"))
for i, (name, a, b) in enumerate(READ, 1):
  out[name] = paired(a, b, "success", i)
  out[name + "_dist"] = paired(a, b, "final_dist", 20 + i, stat="median")

off = defaultdict(dict)
for p in (off_t, off_k):
  for x in p.read_text().splitlines():
    if x.strip():
      r = json.loads(x)
      off[r["arm"]][(r["seed"], r["task"])] = r
ratio = {}
for ref in ("lc_tin_xy6", "lc_film_xy6", "lc_moe_xy6"):
  keys = sorted(set(off["lc_tfilm_xy6"]) & set(off[ref]))
  seeds = sorted(set(k[0] for k in keys))
  for label, get in (("chunk", lambda r: r["action_selection"]), ("flow", lambda r: r["zero_shot"]["selection"])):
    m = np.array([np.mean([math.log(get(off["lc_tfilm_xy6"][k]) / get(off[ref][k])) for k in keys if k[0] == s]) for s in seeds])
    h = float(stats.t.ppf(0.975, len(m) - 1)) * float(m.std(ddof=1)) / math.sqrt(len(m)) if len(m) > 1 else float("nan")
    pc = lambda v: 100.0 * (math.exp(v) - 1.0)
    ratio["lc_tfilm_xy6 vs %s, %s" % (ref, label)] = dict(pct=pc(m.mean()), lo=pc(m.mean() - h), hi=pc(m.mean() + h),
      seeds=len(m), pairs=len(keys))
out["offline"] = ratio
pol = {}
for p in sorted(out["counts"]):
  rows = [r for r in held if r["policy"] == p]
  pol[p] = dict(n=len(rows), success=float(np.mean([r["success"] for r in rows])),
    median_final_dist=float(np.median([r["final_dist"] for r in rows])),
    by_fold={f: float(np.mean([r["success"] for r in rows if r["fold"] == f])) for f in range(4)},
    by_seed={s: float(np.mean([r["success"] for r in rows if r["seed"] == s])) for s in (148021, 148022, 148023)},
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
for k, v in ratio.items():
  lines.append("offline %-40s %+6.1f%% [%+6.1f, %+6.1f] (%d seeds, %d pairs)" % (k, v["pct"], v["lo"], v["hi"], v["seeds"], v["pairs"]))
for p, d in pol.items():
  lines.append("%-3s n=%d success %.3f median dist %.3f folds %s seeds %s probe %.1e" % (p, d["n"], d["success"],
    d["median_final_dist"], {f: round(x, 3) for f, x in d["by_fold"].items()}, {s: round(x, 3) for s, x in d["by_seed"].items()},
    d["probe_max"]))
Path(prefix + ".txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
