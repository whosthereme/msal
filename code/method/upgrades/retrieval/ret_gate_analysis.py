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
POLS = ("gate_k1", "gate_k16", "ret_replay_k1", "ret_replay_k16", "msal_k1", "msal_k16", "ft_k1", "ft_k16", "seen")
TESTS = [("gate_k1", "ret_replay_k1"), ("gate_k16", "ret_replay_k16"), ("gate_k1", "msal_k1"),
         ("gate_k16", "msal_k16"), ("gate_k1", "ft_k1"), ("gate_k16", "ft_k16")]


def main() -> None:
    gate_recs = read(sorted((MINE / "results").glob("R_gate_*.jsonl")))
    recs = read(sorted(STAGE_B.glob("B_*.jsonl"))) + read(sorted((MINE / "results").glob("R_*.jsonl")))
    t = table(recs)
    out = {"success_pct": {}, "head_fraction": {}, "tests": {}}
    for p in POLS:
        if p in t:
            per = {task: 100 * np.mean(list(eps.values())) for task, eps in sorted(t[p].items())}
            out["success_pct"][p] = {"overall": float(np.mean(list(per.values()))), "per_task": per,
                                     "episodes": int(sum(len(e) for e in t[p].values()))}
    for k in (1, 16):
        rs = [r for r in gate_recs if r["policy"] == f"gate_k{k}"]
        per = {}
        for task in sorted({r["task"] for r in rs}):
            rr = [r for r in rs if r["task"] == task]
            per[task] = {"head_replans_pct": 100 * sum(r["head_replans"] for r in rr) / sum(r["replans"] for r in rr),
                         "episodes_any_head_pct": 100 * np.mean([r["head_replans"] > 0 for r in rr]),
                         "head_pct_success_eps": float(100 * np.mean([r["head_replans"] / r["replans"] for r in rr if r["success"]]))
                         if any(r["success"] for r in rr) else None,
                         "head_pct_failed_eps": float(100 * np.mean([r["head_replans"] / r["replans"] for r in rr if not r["success"]]))
                         if any(not r["success"] for r in rr) else None,
                         "T": rr[0]["gate_T"]}
        if rs:
            out["head_fraction"][f"gate_k{k}"] = {
                "overall_head_replans_pct": 100 * sum(r["head_replans"] for r in rs) / sum(r["replans"] for r in rs),
                "mean_over_tasks_pct": float(np.mean([v["head_replans_pct"] for v in per.values()])), "per_task": per}
    for i, (a, b) in enumerate(TESTS):
        out["tests"][f"{a} vs {b}"] = compare(t, a, b, 900 + i)
    (MINE / "gate.json").write_text(json.dumps(out, indent=1))
    for p, v in out["success_pct"].items():
        print(f"{p:16s} {v['overall']:5.1f}  n={v['episodes']}  " +
              " ".join(f"{task}:{s:.0f}" for task, s in v["per_task"].items()))
    for p, v in out["head_fraction"].items():
        print(f"{p} head replans {v['overall_head_replans_pct']:.1f}% (mean over tasks {v['mean_over_tasks_pct']:.1f}%): " +
              " ".join(f"{task}:{x['head_replans_pct']:.0f}%/anyhead{x['episodes_any_head_pct']:.0f}%"
                       f"/succ{x['head_pct_success_eps'] if x['head_pct_success_eps'] is None else round(x['head_pct_success_eps'])}"
                       f"/fail{x['head_pct_failed_eps'] if x['head_pct_failed_eps'] is None else round(x['head_pct_failed_eps'])}"
                       for task, x in v["per_task"].items()))
    for name, c in out["tests"].items():
        if c:
            print(f"{name:28s} {c['diff_pp']:+6.1f} pp [{c['boot95_pp'][0]:+.1f}, {c['boot95_pp'][1]:+.1f}] "
                  f"{c['reading']} ({c['tasks_a_better']}/{c['tasks_b_better']} tasks, {c['episodes']} eps)")


if __name__ == "__main__":
    main()
