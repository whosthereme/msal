import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 10_000
SEED = 20261116
src, prefix = Path(sys.argv[1]), sys.argv[2]
R = [json.loads(x) for p in sorted(src.glob("*.jsonl")) for x in p.read_text().splitlines() if x.strip()]
held = [r for r in R if r["set"] == "held"]
seen = [r for r in R if r["set"] == "seen"]


def table(rows, key):
  t = defaultdict(lambda: defaultdict(dict))
  for r in rows:
    t[r["policy"]][r["task"]][(r["seed"], r["episode"])] = float(r[key])
  return t


def paired(rows, a, b, key, index, stat="mean"):
  t = table(rows, key)
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


def unpaired(rows_a, rows_b, policy, key, index):
  ta, tb = table(rows_a, key)[policy], table(rows_b, key)[policy]
  A = [np.array(list(ta[k].values())) for k in sorted(ta)]
  Bv = [np.array(list(tb[k].values())) for k in sorted(tb)]
  rng = np.random.default_rng([SEED, index])

  def two_level(G):
    ti = rng.integers(len(G), size=len(G))
    return float(np.mean([np.mean(G[i][rng.integers(len(G[i]), size=len(G[i]))]) for i in ti]))

  point = float(np.mean([x.mean() for x in A]) - np.mean([x.mean() for x in Bv]))
  boot = np.array([two_level(A) - two_level(Bv) for _ in range(B)])
  lo, hi = np.percentile(boot, [2.5, 97.5])
  return dict(a=policy + " held", b=policy + " seen", key=key, tasks=[len(A), len(Bv)], diff=point,
    a_mean=float(np.mean([x.mean() for x in A])), b_mean=float(np.mean([x.mean() for x in Bv])),
    ci=[float(lo), float(hi)], reading="better" if lo > 0 else ("worse" if hi < 0 else "not distinguishable"))


out = dict(records=len(R), held_records=len(held), seen_records=len(seen))
out["P1"] = paired(held, "C0", "N0", "success", 1)
out["P2"] = paired(held, "C0", "H0", "success", 2)
out["S1"] = paired(held, "C1", "C0", "success", 3)
out["S2"] = paired(held, "H1", "H0", "success", 4)
out["S3"] = unpaired(held, seen, "C0", "success", 5) if seen else None
for name, (a, b), i in (("S4_P1", ("C0", "N0"), 6), ("S4_P2", ("C0", "H0"), 7)):
  out[name] = paired(held, a, b, "final_dist", i, stat="median")

pol = sorted(set(r["policy"] for r in held))
desc = {}
for p in pol:
  rows = [r for r in held if r["policy"] == p]
  desc[p] = dict(n=len(rows), success=float(np.mean([r["success"] for r in rows])),
    median_final_dist=float(np.median([r["final_dist"] for r in rows])),
    by_ring={g: float(np.mean([r["success"] for r in rows if r["ring"] == g])) for g in (0, 1, 2)},
    by_fold={f: float(np.mean([r["success"] for r in rows if r["fold"] == f])) for f in range(4)})
script_ok = {(r["task"], r["episode"]) for r in held if r["policy"] == "script" and r["success"]}
desc["script"] = dict(n=sum(1 for r in held if r["policy"] == "script"), solved=len(script_ok))
for p in pol:
  rows = [r for r in held if r["policy"] == p and (r["task"], r["episode"]) in script_ok]
  if rows:
    desc[p]["success_on_script_feasible"] = float(np.mean([r["success"] for r in rows]))
by_vessel = defaultdict(list)
for r in held:
  if r["policy"] == "C0":
    by_vessel[r["vessel"]].append(r["success"])
desc["C0_by_vessel"] = {k: [float(np.mean(v)), len(v)] for k, v in sorted(by_vessel.items())}
for p in sorted(set(r["policy"] for r in seen)):
  rows = [r for r in seen if r["policy"] == p]
  desc["seen_" + p] = dict(n=len(rows), success=float(np.mean([r["success"] for r in rows])),
    median_final_dist=float(np.median([r["final_dist"] for r in rows])))
out["descriptive"] = desc
Path(prefix + ".json").write_text(json.dumps(out, indent=1))
lines = []
for k in ("P1", "P2", "S1", "S2", "S3", "S4_P1", "S4_P2"):
  v = out[k]
  if v:
    lines.append("%-6s %s vs %s (%s %s): %.3f vs %.3f, diff %+.3f [%+.3f, %+.3f] -> %s" % (k, v["a"], v["b"], v["key"],
      v.get("stat", "mean"), v["a_mean"], v["b_mean"], v["diff"], v["ci"][0], v["ci"][1], v["reading"]))
for p in pol:
  d = desc[p]
  if "n" not in d or "success" not in d:
    continue
  lines.append("%-7s n=%d success %.3f median dist %.3f rings %s folds %s feasible %s" % (p, d["n"], d["success"],
    d["median_final_dist"], {g: round(x, 3) for g, x in d["by_ring"].items()},
    {f: round(x, 3) for f, x in d["by_fold"].items()}, round(d.get("success_on_script_feasible", float("nan")), 3)))
Path(prefix + ".txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
