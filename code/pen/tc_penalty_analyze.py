import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = [0, 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 15, 16, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 31, 32, 33, 35, 37, 39,
  40, 41, 42, 43, 44, 45, 47, 49, 51, 52, 53, 54]
SETTINGS = ("S1", "S2", "S3", "S4")
NAMES = {"S1": "orthogonality 0", "S2": "orthogonality 0.08", "S3": "group 0", "S4": "group 0.004"}


def load(dirs):
  rows = {}
  for d in dirs:
    for line in (Path(d) / "folds.jsonl").read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        rows[(int(r["held_task"]), r["setting"])] = r
  return rows


def reading(a, b):
  l = np.log(np.asarray(b) / np.asarray(a))
  rng = np.random.default_rng(0)
  idx = rng.integers(0, len(l), size=(10000, len(l)))
  boot = l[idx].mean(axis=1)
  lo, hi = np.percentile(boot, [2.5, 97.5])
  f = lambda v: 100.0 * (np.exp(v) - 1.0)
  return dict(change=f(l.mean()), lo=f(lo), hi=f(hi), n=len(l))


def main():
  rows = load(sys.argv[1:])
  out = {"complete": {}, "failed": [], "readings": {}, "zero_shot_max_abs_diff": 0.0, "active_slots": {}}
  for s in ("S0",) + SETTINGS:
    out["complete"][s] = sum(1 for t in TASKS if (t, s) in rows and not rows[(t, s)].get("failed"))
  out["failed"] = sorted([k for k, r in rows.items() if r.get("failed")])
  tasks = [t for t in TASKS if all((t, s) in rows and not rows[(t, s)].get("failed") for s in ("S0",) + SETTINGS)]
  for t in tasks:
    z0 = rows[(t, "S0")]["zero_shot"]["selection"]
    for s in SETTINGS:
      out["zero_shot_max_abs_diff"] = max(out["zero_shot_max_abs_diff"], abs(rows[(t, s)]["zero_shot"]["selection"] - z0))
  for s in ("S0",) + SETTINGS:
    act = [sum(1 for v in rows[(t, s)]["slot_norms"] if v > 1e-3) for t in tasks]
    out["active_slots"][s] = dict(mean=float(np.mean(act)), min=int(min(act)), max=int(max(act))) if act else None
  shots = sorted(int(k) for k in rows[(tasks[0], "S0")]["combination"]) if tasks else []
  for s in SETTINGS:
    r = {}
    for k in (1, 16):
      key = "combination_%d" % k
      r["sampled_chunk_k%d" % k] = reading([rows[(t, "S0")]["action"][key] for t in tasks],
        [rows[(t, s)]["action"][key] for t in tasks])
    for k in shots:
      r["flow_k%d" % k] = reading([rows[(t, "S0")]["combination"][str(k)]["selection"] for t in tasks],
        [rows[(t, s)]["combination"][str(k)]["selection"] for t in tasks])
    out["readings"][s] = r
  out["tasks_used"] = len(tasks)
  (HERE / "penalty_analysis.json").write_text(json.dumps(out, indent=1))
  lines = ["PLAN_PENALTY analysis: %d of 41 tasks complete in all settings; failed: %s" % (len(tasks), out["failed"]),
    "zero-shot flow max |S - S0| = %.3g (must be ~0)" % out["zero_shot_max_abs_diff"],
    "active slots (norm > 1e-3): %s" % out["active_slots"]]
  for s in SETTINGS:
    lines.append("%s (%s) vs S0:" % (s, NAMES[s]))
    for name, v in out["readings"][s].items():
      lines.append("  %-18s %+6.1f%%  [%+6.1f, %+6.1f]  n=%d" % (name, v["change"], v["lo"], v["hi"], v["n"]))
  (HERE / "penalty_analysis.txt").write_text("\n".join(lines) + "\n")
  print("\n".join(lines))


if __name__ == "__main__":
  main()
