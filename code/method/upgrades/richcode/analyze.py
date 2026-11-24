from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
from cl_adapt import compare, read, table

HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)
MINE = ROOT / "results_remote" / "upgrades" / "richcode"
COMPS = [("rich_k1", "msal_k1"), ("rich_k16", "msal_k16"), ("rich_k1", "zero"), ("rich_k16", "zero"),
         ("rich_k1", "ft_k1"), ("rich_k16", "ft_k16"), ("rich_k16", "rich_k1"), ("msal_k16", "msal_k1")]


def main() -> None:
    recs = read(sorted((ROOT / "results_remote" / "cl_adapt").glob("B_*.jsonl")))
    recs += read(sorted((MINE / "results").glob("B_rich_*.jsonl")))
    t = table(recs)
    pol = ("zero", "msal_k1", "rich_k1", "ft_k1", "msal_k16", "rich_k16", "ft_k16", "seen")
    succ = {p: {h: 100 * float(np.mean(list(t[p][h].values()))) for h in HELD if h in t.get(p, {})} for p in pol}
    eps = {p: sum(len(t[p][h]) for h in t.get(p, {}) if h in HELD) for p in pol}
    out = {"success_pct": {p: {"mean": float(np.mean(list(v.values()))) if v else None, "per_task": v,
                               "episodes": eps[p]} for p, v in succ.items()},
           "comparisons": {f"{a} vs {b}": compare(t, a, b, 300 + i) for i, (a, b) in enumerate(COMPS)}}
    try:
        import torch
        off = {}
        for h in HELD:
            c = torch.load(MINE / "codes" / f"rich{h}.pt", weights_only=False)
            for k, ad in c["adapt"].items():
                off.setdefault(f"k{k}", {})[h] = {kk: ad[kk] for kk in (
                    "selection", "msal_selection", "ft_selection", "demo_mse", "msal_demo_mse", "code_norm",
                    "z_norm", "numbers_per_task", "seconds_inversion", "seconds_code", "variant", "demo_rows")}
        out["offline"] = {k: {"per_task": v, "mean_selection": float(np.mean([x["selection"] for x in v.values()])),
                              "mean_msal_selection": float(np.mean([x["msal_selection"] for x in v.values()])),
                              "mean_ft_selection": float(np.mean([x["ft_selection"] for x in v.values()])),
                              "tasks_lower_than_msal": int(sum(x["selection"] < x["msal_selection"] for x in v.values()))}
                          for k, v in off.items()}
    except Exception as exc:
        out["offline"] = f"unavailable: {exc}"
    (MINE / "analysis.json").write_text(json.dumps(out, indent=1))
    for p in pol:
        s = out["success_pct"][p]
        print(f"{p:9s} mean={s['mean']} n={s['episodes']} " + " ".join(f"{h}:{v:.0f}" for h, v in s["per_task"].items()))
    for name, c in out["comparisons"].items():
        if c:
            print(f"{name:22s} diff={c['diff_pp']:+.2f}pp CI=[{c['boot95_pp'][0]:+.2f},{c['boot95_pp'][1]:+.2f}] "
                  f"{c['reading']} tasks a>b {c['tasks_a_better']} b>a {c['tasks_b_better']} ({c['episodes']} eps)")
    if isinstance(out["offline"], dict):
        for k, v in out["offline"].items():
            print(k, {kk: vv for kk, vv in v.items() if kk != "per_task"})


if __name__ == "__main__":
    main()
