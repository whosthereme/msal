import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STORED, NEW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
N_BOOT, BOOT_SEED = 10_000, 20261115
COMPARISONS = [("T1", "tmsal_k16", "msal_k16", "primary"), ("T2", "tmsal_k1", "msal_k1", "primary"),
  ("T3", "tzero", "zero", "secondary"), ("T4", "tmsal_k16", "tzero", "secondary"),
  ("T5", "tmsal_k1", "tzero", "secondary")]
DESCRIPTIVE = [("tmsal_k16", "ft_k16"), ("tmsal_k1", "ft_k1"), ("tmsal_k16", "seen"), ("tmsal_k1", "seen")]


def read(paths):
  out = []
  for p in paths:
    out += [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
  return out


recs = read(sorted(STORED.glob("B_*.jsonl"))) + read(sorted(NEW.glob("B_*.jsonl")))
t = defaultdict(dict)
for r in recs:
  t[r["policy"]].setdefault(r["task"], {})[r["episode"]] = int(r["success"])
probe = [r["probe_max_abs"] for r in read(sorted(NEW.glob("B_*.jsonl")))]


def compare(a, b, index):
  tasks = sorted(set(t.get(a, {})) & set(t.get(b, {})))
  pairs = []
  for task in tasks:
    eps = sorted(set(t[a][task]) & set(t[b][task]))
    pairs.append(np.array([[t[a][task][e], t[b][task][e]] for e in eps], dtype=float))
  diff = 100 * np.mean([p[:, 0].mean() - p[:, 1].mean() for p in pairs])
  rng = np.random.default_rng([BOOT_SEED, index])
  boot = np.empty(N_BOOT)
  for i in range(N_BOOT):
    vals = []
    for j in rng.integers(0, len(pairs), len(pairs)):
      p = pairs[j]
      e = rng.integers(0, len(p), len(p))
      vals.append(p[e, 0].mean() - p[e, 1].mean())
    boot[i] = 100 * np.mean(vals)
  lo, hi = np.percentile(boot, [2.5, 97.5])
  return dict(a=a, b=b, tasks=len(tasks), episodes=int(sum(len(p) for p in pairs)), diff_pp=float(diff),
    boot95_pp=[float(lo), float(hi)], reading="better" if lo > 0 else "worse" if hi < 0 else "not distinguishable",
    tasks_a_better=int(sum(p[:, 0].mean() > p[:, 1].mean() for p in pairs)),
    tasks_b_better=int(sum(p[:, 0].mean() < p[:, 1].mean() for p in pairs)))


success = dict((p, dict((task, 100 * np.mean(list(e.values()))) for task, e in per.items())) for p, per in t.items())
print("episodes per new policy:", dict((p, sum(len(e) for e in t[p].values())) for p in ("tzero", "tmsal_k1", "tmsal_k16") if p in t))
print("probe max abs over new episodes: %.2e" % max(probe))
print("success (%%, mean over tasks):")
for p in ("zero", "tzero", "msal_k1", "tmsal_k1", "msal_k16", "tmsal_k16", "ft_k1", "ft_k16", "seen"):
  if p in success:
    v = success[p]
    print("  %-10s %5.1f   per task: %s" % (p, np.mean(list(v.values())), " ".join("%d:%.0f" % (k, v[k]) for k in sorted(v))))
REPORT = dict(success_pct=dict((p, dict(overall=float(np.mean(list(v.values()))), per_task=v)) for p, v in success.items()),
  comparisons={}, descriptive={}, probe_max_abs=max(probe))
for i, (name, a, b, role) in enumerate(COMPARISONS):
  c = compare(a, b, i)
  c["role"] = role
  REPORT["comparisons"][name] = c
  print("  %s %-10s vs %-9s %+6.1f pp [%+6.1f, %+6.1f] %-20s tasks %d/%d (%s)" % (name, a, b, c["diff_pp"],
    c["boot95_pp"][0], c["boot95_pp"][1], c["reading"], c["tasks_a_better"], c["tasks_b_better"], role))
for i, (a, b) in enumerate(DESCRIPTIVE):
  c = compare(a, b, 100 + i)
  REPORT["descriptive"]["%s vs %s" % (a, b)] = c
  print("     %-10s vs %-9s %+6.1f pp [%+6.1f, %+6.1f] %s (descriptive)" % (a, b, c["diff_pp"], c["boot95_pp"][0],
    c["boot95_pp"][1], c["reading"]))
OUT.write_text(json.dumps(REPORT, indent=1))
print("written", OUT)
