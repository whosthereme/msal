from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "analysis"))
from cl_adapt import compare, read, table

RES = ROOT / "results_remote" / "upgrades" / "meta"
HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)


def convert():
    for raw in sorted((RES / "raw").glob("R_*.jsonl")):
        recs = read([raw])
        if not recs:
            continue
        base = recs[0]["policy"]
        out = RES / f"M_meta_{base}_task{recs[0]['task']}.jsonl"
        with out.open("w") as fh:
            for r in sorted(recs, key=lambda r: r["episode"]):
                fh.write(json.dumps({**r, "policy": f"meta_{base}", "variant": "meta", "base_policy": base}) + "\n")


def main():
    convert()
    stage_b = read(sorted((ROOT / "results_remote" / "cl_adapt").glob("B_*.jsonl")))
    mine = read(sorted(RES.glob("M_meta_*.jsonl")))
    t = table(stage_b + mine)
    pols = ("meta_msal_k1", "meta_msal_k16", "meta_zero", "msal_k1", "msal_k16", "zero", "ft_k1", "ft_k16", "seen")
    success = {}
    for p in pols:
        if p in t:
            per = {h: 100 * float(np.mean(list(t[p][h].values()))) for h in HELD if h in t[p]}
            success[p] = {"overall": float(np.mean(list(per.values()))), "per_task": per,
                          "episodes": int(sum(len(t[p][h]) for h in per))}
    if "meta_zero" in t:
        t["zero_e0_24"] = {h: {e: v for e, v in eps.items() if e < 25} for h, eps in t["zero"].items()}
        per = {h: 100 * float(np.mean(list(t["zero_e0_24"][h].values()))) for h in HELD}
        success["zero_e0_24"] = {"overall": float(np.mean(list(per.values()))), "per_task": per}
    comps = [("M1", "meta_msal_k1", "msal_k1"), ("M2", "meta_msal_k16", "msal_k16"),
             ("M3", "meta_msal_k1", "zero"), ("M4", "meta_msal_k16", "zero"),
             ("M5", "meta_msal_k16", "meta_msal_k1"), ("M6", "meta_zero", "zero"),
             ("M7", "meta_msal_k1", "ft_k1"), ("M8", "meta_msal_k16", "ft_k16")]
    res = {}
    for i, (name, a, b) in enumerate(comps):
        if a in t and b in t:
            res[name] = compare(t, a, b, 200 + i)
    out = {"success_pct": success, "comparisons": res}
    (RES / "compare.json").write_text(json.dumps(out, indent=1))
    for p, v in success.items():
        print(f"{p:15s} {v['overall']:6.2f}  " + " ".join(f"{h}:{v['per_task'].get(h, float('nan')):5.1f}" for h in HELD))
    for name, c in res.items():
        if c:
            print(f"{name} {c['a']} - {c['b']}: {c['diff_pp']:+.2f} pp [{c['boot95_pp'][0]:+.2f}, {c['boot95_pp'][1]:+.2f}] "
                  f"{c['reading']} (tasks a>b {c['tasks_a_better']}, b>a {c['tasks_b_better']}, eps {c['episodes']})")


if __name__ == "__main__":
    main()
