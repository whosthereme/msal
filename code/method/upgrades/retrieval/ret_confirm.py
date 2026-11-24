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
POLS = ("ret_replay_k1", "ret_replay_k16", "ret_replaysh_k1", "ret_msal_k1", "ret_msal_k16",
        "zero", "msal_k1", "msal_k16", "ft_k1", "ft_k16", "seen")
TESTS = [("P1 primary", "ret_replay_k1", "msal_k1", (25, 49)),
         ("P1 secondary", "ret_replay_k1", "ft_k1", (25, 49)),
         ("P2a", "ret_replay_k16", "msal_k16", (0, 49)),
         ("P2b", "ret_replay_k16", "ft_k16", (0, 49)),
         ("P2c", "ret_replay_k16", "ret_replay_k1", (0, 49)),
         ("P3", "ret_replaysh_k1", "ret_replay_k1", (0, 24)),
         ("descr: ret_replay_k1 vs msal_k1, eps 0-49", "ret_replay_k1", "msal_k1", (0, 49)),
         ("descr: ret_replay_k1 vs ft_k1, eps 0-49", "ret_replay_k1", "ft_k1", (0, 49)),
         ("descr: ret_replay_k1 vs seen, eps 0-49", "ret_replay_k1", "seen", (0, 49)),
         ("descr: ret_replay_k16 vs seen, eps 0-49", "ret_replay_k16", "seen", (0, 49)),
         ("descr: ret_replaysh_k1 vs msal_k1, eps 0-24", "ret_replaysh_k1", "msal_k1", (0, 24))]


def window(t, lo, hi):
    out = {}
    for p, per in t.items():
        for task, eps in per.items():
            sel = {e: v for e, v in eps.items() if lo <= e <= hi}
            if sel:
                out.setdefault(p, {})[task] = sel
    return out


def main() -> None:
    recs = read(sorted(STAGE_B.glob("B_*.jsonl"))) + read(sorted((MINE / "results").glob("R_*.jsonl")))
    t = table(recs)
    out = {"success_pct": {}, "tests": {}}
    for lo, hi in ((0, 24), (25, 49), (0, 49)):
        tw = window(t, lo, hi)
        key = f"eps{lo}-{hi}"
        for p in POLS:
            per = {task: 100 * np.mean(list(eps.values())) for task, eps in sorted(tw.get(p, {}).items()) if eps}
            if len(per) == 8 and all(len(tw[p][task]) == hi - lo + 1 for task in per):
                out["success_pct"].setdefault(key, {})[p] = {"overall": float(np.mean(list(per.values()))),
                                                              "per_task": per}
    for i, (name, a, b, (lo, hi)) in enumerate(TESTS):
        out["tests"][name] = {**(compare(window(t, lo, hi), a, b, 700 + i) or {}), "episodes_window": [lo, hi]}
    (MINE / "confirmation.json").write_text(json.dumps(out, indent=1))
    for key, per in out["success_pct"].items():
        print(f"== {key}")
        for p, v in per.items():
            print(f"  {p:16s} {v['overall']:5.1f}  " + " ".join(f"{task}:{s:.0f}" for task, s in v["per_task"].items()))
    for name, c in out["tests"].items():
        if "diff_pp" in c:
            print(f"{name:45s} {c['a']} vs {c['b']} eps{c['episodes_window']}: {c['diff_pp']:+6.1f} pp "
                  f"[{c['boot95_pp'][0]:+.1f}, {c['boot95_pp'][1]:+.1f}] {c['reading']} "
                  f"({c['tasks_a_better']}/{c['tasks_b_better']} tasks, {c['episodes']} eps)")


if __name__ == "__main__":
    main()
