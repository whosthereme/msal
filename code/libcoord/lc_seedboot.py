import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

old_dir, mid_dir, k_dir, new_dir = [Path(x) for x in sys.argv[1:5]]
prefix = sys.argv[5]
B, GEN = 10_000, 20261123
SEEDS = (148021, 148022, 148023)


def read_dir(d, pattern="*.jsonl"):
  return [json.loads(x) for p in sorted(d.glob(pattern)) for x in p.read_text().splitlines() if x.strip()]


held = ([r for r in read_dir(old_dir, "held_*.jsonl") if r["policy"] in ("C0", "C1", "N0")] + read_dir(mid_dir, "held_*.jsonl")
  + read_dir(k_dir, "held_*.jsonl") + [r for r in read_dir(new_dir) if r["set"] == "held"])
T = defaultdict(lambda: defaultdict(dict))
for r in held:
  T[r["policy"]][(r["task"], r["seed"])][r["episode"]] = float(r["success"])

COMP = (
  ("H8 (reg.)", "K0", "C0", "MSAL vs MSAL-h, coordinates"),
  ("H9 (reg.)", "K0", "N0", "MSAL coordinates vs no description"),
  ("H7 (reg.)", "K0", "T0", "MSAL vs input concatenation"),
  ("post hoc", "K0", "F0", "MSAL vs FiLM, linear readout"),
  ("post hoc", "K0", "M0", "MSAL vs mixture of experts"),
  ("CL_HEADS", "T0", "C0", "concatenation vs MSAL-h"),
  ("post hoc", "T0", "F0", "concatenation vs FiLM, linear readout"),
  ("post hoc", "T0", "M0", "concatenation vs mixture of experts"),
  ("CL_HEADS", "F0", "C0", "FiLM linear vs MSAL-h"),
  ("CL_HEADS", "M0", "C0", "mixture vs MSAL-h"),
  ("CL_COORD", "C0", "N0", "MSAL-h coordinates vs none"),
  ("H10 (reg.)", "J0", "K0", "MSAL harmonic vs coordinates"),
  ("H11 (reg.)", "J0", "N0", "MSAL harmonic vs none"),
  ("CL_HEADS", "G0", "C0", "MSAL-h harmonic vs coordinates"),
  ("CL_HEADS", "G0", "N0", "MSAL-h harmonic vs none"),
  ("H12 (reg.)", "K1", "K0", "MSAL one demonstration vs zero-shot"),
  ("CL_COORD", "C1", "C0", "MSAL-h one demonstration vs zero-shot"),
  ("post hoc", "K1", "T0", "MSAL one demonstration vs concatenation"),
)
lines = ["seed-level view of the closed-loop comparisons (success, points); post hoc",
  "%-11s %-42s %6s %6s | %-34s | %-28s | %s" % ("plan", "comparison", "a", "b", "per seed (a-b)", "seed t (df 2)", "3-level bootstrap")]
out = {}
for idx, (tag, a, b, label) in enumerate(COMP):
  cells = sorted(set(T[a]) & set(T[b]))
  tasks = sorted(set(c[0] for c in cells))
  assert all((t, s) in T[a] and (t, s) in T[b] for t in tasks for s in SEEDS), (a, b)
  D = {}
  for t in tasks:
    for s in SEEDS:
      eps = sorted(set(T[a][(t, s)]) & set(T[b][(t, s)]))
      D[(t, s)] = np.array([T[a][(t, s)][e] - T[b][(t, s)][e] for e in eps])
  per_seed = np.array([np.mean([D[(t, s)].mean() for t in tasks]) for s in SEEDS])
  m = per_seed.mean()
  h = float(stats.t.ppf(0.975, 2)) * per_seed.std(ddof=1) / np.sqrt(3)
  rng = np.random.default_rng([GEN, idx])
  boot = np.empty(B)
  for j in range(B):
    ss = rng.integers(3, size=3)
    tt = rng.integers(len(tasks), size=len(tasks))
    v = 0.0
    for ti in tt:
      for si in ss:
        x = D[(tasks[ti], SEEDS[si])]
        v += x[rng.integers(len(x), size=len(x))].mean()
    boot[j] = v / (len(tt) * len(ss))
  lo, hi = np.percentile(boot, [2.5, 97.5])
  rd = lambda l, u: "better" if l > 0 else ("worse" if u < 0 else "n.d.")
  out["%s %s-%s" % (tag, a, b)] = dict(label=label, a=a, b=b, per_seed=(100 * per_seed).tolist(), mean=100 * m,
    t_ci=[100 * (m - h), 100 * (m + h)], boot_ci=[100 * lo, 100 * hi], t_reading=rd(m - h, m + h), boot_reading=rd(lo, hi))
  pa = 100 * np.mean([np.mean(list(T[a][(t, s)].values())) for t in tasks for s in SEEDS])
  pb = 100 * np.mean([np.mean(list(T[b][(t, s)].values())) for t in tasks for s in SEEDS])
  lines.append("%-11s %-42s %6.1f %6.1f | %-34s | %+5.1f [%+5.1f, %+5.1f] %-5s | [%+5.1f, %+5.1f] %s" % (tag, label, pa, pb,
    " ".join("%+5.1f" % v for v in 100 * per_seed), 100 * m, 100 * (m - h), 100 * (m + h), rd(m - h, m + h),
    100 * lo, 100 * hi, rd(lo, hi)))
Path(prefix + ".txt").write_text("\n".join(lines) + "\n")
Path(prefix + ".json").write_text(json.dumps(out, indent=1))
print("\n".join(lines))
