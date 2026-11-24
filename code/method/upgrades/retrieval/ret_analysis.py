from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
from cl_adapt import compare, read, table

STAGE_B = ROOT / "results_remote" / "cl_adapt"
MINE = ROOT / "results_remote" / "upgrades" / "retrieval"
OURS = ("ret_msal_k1", "ret_msal_k16", "ret_replay_k1")
REF = ("zero", "msal_k1", "msal_k16", "ft_k1", "ft_k16", "seen")
PAIRS = [("ret_msal_k1", "msal_k1"), ("ret_msal_k16", "msal_k16"), ("ret_msal_k1", "zero"), ("ret_msal_k16", "zero"),
         ("ret_replay_k1", "msal_k1"), ("ret_replay_k1", "zero"), ("ret_msal_k1", "ft_k1"),
         ("ret_msal_k16", "ft_k16"), ("ret_replay_k1", "ft_k1"), ("ret_msal_k1", "ret_replay_k1"),
         ("ret_msal_k16", "ret_msal_k1")]


def main() -> None:
    recs = read(sorted(STAGE_B.glob("B_*.jsonl"))) + read(sorted((MINE / "results").glob("R_*.jsonl")))
    t = table(recs)
    out = {"success_pct": {}, "success_pct_eps0_24": {}, "comparisons": {}}
    for p in OURS + REF:
        if p in t:
            per = {task: 100 * np.mean(list(eps.values())) for task, eps in sorted(t[p].items())}
            out["success_pct"][p] = {"overall": float(np.mean(list(per.values()))), "per_task": per,
                                     "episodes": int(sum(len(e) for e in t[p].values()))}
            per25 = {task: 100 * np.mean([v for e, v in eps.items() if e < 25]) for task, eps in sorted(t[p].items())}
            out["success_pct_eps0_24"][p] = {"overall": float(np.mean(list(per25.values()))), "per_task": per25}
    for i, (a, b) in enumerate(PAIRS):
        out["comparisons"][f"{a} vs {b}"] = compare(t, a, b, 500 + i)
    (MINE / "analysis.json").write_text(json.dumps(out, indent=1))
    for p, v in out["success_pct"].items():
        print(f"{p:14s} {v['overall']:5.1f}%  n={v['episodes']}  " +
              " ".join(f"{task}:{s:.0f}" for task, s in v["per_task"].items()))
    for name, c in out["comparisons"].items():
        if c:
            print(f"{name:30s} {c['diff_pp']:+6.1f} pp [{c['boot95_pp'][0]:+.1f}, {c['boot95_pp'][1]:+.1f}] "
                  f"{c['reading']} ({c['tasks_a_better']}/{c['tasks_b_better']} tasks, {c['episodes']} eps)")


if __name__ == "__main__":
    main()
