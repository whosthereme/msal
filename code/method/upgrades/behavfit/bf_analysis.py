from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("cl_adapt", ROOT / "analysis" / "cl_adapt.py")
cla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cla)

STAGE_B = ROOT / "results_remote" / "cl_adapt"
MINE = ROOT / "results_remote" / "upgrades" / "behavfit"
HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)
OURS = ("bf_k1", "bf_k16")


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main():
    recs = cla.read(sorted(STAGE_B.glob("B_*.jsonl"))) + cla.read(sorted(MINE.glob("bf_*.jsonl")))
    t = cla.table(recs)
    out = {"success_pct": {}, "comparisons": {}}
    for p in ("zero", "msal_k1", "msal_k16", "ft_k1", "ft_k16", "seen") + OURS:
        if p in t:
            per = {task: 100 * np.mean(list(eps.values())) for task, eps in t[p].items()}
            out["success_pct"][p] = {"overall": float(np.mean(list(per.values()))), "per_task": per,
                                     "episodes": int(sum(len(e) for e in t[p].values()))}
    comps = [("D1", "bf_k1", "msal_k1"), ("D2", "bf_k16", "msal_k16"), ("D3", "bf_k1", "zero"), ("D4", "bf_k16", "zero"),
             ("D5", "bf_k1", "ft_k1"), ("D6", "bf_k16", "ft_k16")]
    for i, (name, a, b) in enumerate(comps):
        if a in t:
            out["comparisons"][name] = cla.compare(t, a, b, 500 + i)
    diag_path = MINE / "diag.jsonl"
    if diag_path.exists():
        diag = [json.loads(x) for x in diag_path.read_text().splitlines() if x.strip()]
        succ = {(p, task): 100 * np.mean(list(eps.values())) for p in t for task, eps in t[p].items()}
        stageb = [r for r in diag if r["policy"] in ("zero", "msal_k1", "msal_k16", "ft_k1", "ft_k16")]
        res = {}
        for key in ("act", "act_first8", "flow", "flow_low"):
            x = [r[key] for r in stageb]
            y = [succ[(r["policy"], r["task"])] for r in stageb]
            within = []
            for task in HELD:
                rr = [r for r in stageb if r["task"] == task]
                ys = [succ[(r["policy"], task)] for r in rr]
                if len(set(ys)) > 1:
                    within.append(spearman([r[key] for r in rr], ys))
            res[key] = {"spearman_40": spearman(x, y), "mean_within_task": float(np.mean(within)),
                        "tasks_used_within": len(within)}
        out["diagnostic"] = res
        out["diagnostic_table"] = {f"{r['policy']}@{r['task']}": {k: r[k] for k in ("act", "flow", "flow_low")}
                                   for r in diag}
        mean = {}
        for p in sorted({r["policy"] for r in diag}):
            mean[p] = {k: float(np.mean([r[k] for r in diag if r["policy"] == p])) for k in ("act", "act_first8", "flow", "flow_low")}
        out["diagnostic_means"] = mean
    (MINE / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "diagnostic_table"}, indent=1))


if __name__ == "__main__":
    main()
