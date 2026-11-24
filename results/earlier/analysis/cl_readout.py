import json
import re
from pathlib import Path

import numpy as np

from cl_adapt import compare, read, table

ROOT = Path(__file__).resolve().parent.parent
RR = ROOT / "results_remote"
COMPARISONS = [("R1", "roft_k16", "msal_k16", "primary"), ("R2", "roft_k1", "msal_k1", "primary"),
               ("R3", "roft_k16", "ft_k16", "secondary"), ("R4", "roft_k1", "ft_k1", "secondary"),
               ("R5", "roft_k16", "zero", "secondary"), ("R6", "roft_k1", "zero", "secondary")]


def main():
    recs = read(sorted((RR / "cl_adapt").glob("B_*.jsonl")))
    files = sorted((RR / "readout_cl").glob("R_roft_k*_task*.jsonl"))
    for f in files:
        label = re.match(r"R_(roft_k\d+)_task", f.name).group(1)
        for r in read([f]):
            assert r["policy"] == "ft_k" + label.split("_k")[1] and r["artifact"].startswith("roft"), r
            recs.append({**r, "policy": label})
    t = table(recs)
    out = {"files": len(files), "success_pct": {}, "per_task_pct": {}, "comparisons": {}}
    for p in ("zero", "msal_k1", "msal_k16", "roft_k1", "roft_k16", "ft_k1", "ft_k16", "seen"):
        if p in t:
            per = {task: 100 * float(np.mean(list(e.values()))) for task, e in sorted(t[p].items())}
            out["per_task_pct"][p] = per
            out["success_pct"][p] = {"mean_over_tasks": float(np.mean(list(per.values()))), "tasks": len(per),
                                     "episodes": sum(len(e) for e in t[p].values())}
            print(f"{p:9s} {out['success_pct'][p]['mean_over_tasks']:5.1f}%  ({out['success_pct'][p]['episodes']} eps)  "
                  + " ".join(f"{task}:{v:.0f}" for task, v in per.items()))
    for i, (name, a, b, kind) in enumerate(COMPARISONS):
        c = compare(t, a, b, 600 + i)
        if c:
            out["comparisons"][name] = {**c, "kind": kind}
            print(f"{name} ({kind}) {a} vs {b}: {c['diff_pp']:+.1f} [{c['boot95_pp'][0]:+.1f}, {c['boot95_pp'][1]:+.1f}] "
                  f"{c['reading']} (tasks {c['tasks_a_better']}/{c['tasks_b_better']}, {c['episodes']} eps)")
    (ROOT / "analysis" / "cl_readout.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
