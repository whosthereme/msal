import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 148021
FT_KEYS = {1: "k1_all_lr0.0001_s100", 16: "k16_all_lr0.0001_s400"}


def records(pattern):
    out = {}
    for p in sorted((ROOT / "results_remote").glob(pattern)):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["held_task"], r)
    return out


def rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "cl_offline_link.json")
    args = ap.parse_args()
    l40 = records(f"l40/seed{SEED}_*/folds.jsonl")
    ft = records(f"ft/*_l40_*_s{SEED}/folds.jsonl")
    succ = defaultdict(dict)
    for p in sorted(args.results.glob("B_*.jsonl")):
        eps = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if eps:
            succ[eps[0]["policy"]][eps[0]["task"]] = 100 * np.mean([e["success"] for e in eps])
    rows = []
    for h in sorted(succ.get("zero", {})):
        offline = {"zero": l40[h]["zero_shot"]["selection"]}
        for k in (1, 16):
            offline[f"msal_k{k}"] = l40[h]["combination"][str(k)]["selection"]
            offline[f"ft_k{k}"] = ft[h]["finetune"][FT_KEYS[k]]["selection"]
        for pol, mse in offline.items():
            if h in succ.get(pol, {}):
                rows.append({"task": h, "policy": pol, "offline_mse": mse,
                             "offline_change_pct": 100 * (mse / offline["zero"] - 1), "success_pct": succ[pol][h]})
    pols = sorted({r["policy"] for r in rows}, key=["zero", "msal_k1", "msal_k16", "ft_k1", "ft_k16"].index)
    per_policy = {p: {"offline_change_pct_mean": float(np.mean([r["offline_change_pct"] for r in rows if r["policy"] == p])),
                      "success_pct_mean": float(np.mean([r["success_pct"] for r in rows if r["policy"] == p]))}
                  for p in pols}
    within = []
    for h in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == h]
        within.append(spearman([r["offline_mse"] for r in sub], [r["success_pct"] for r in sub]))
    out = {"note": "post hoc, seed 148021 offline selection-row MSE vs Stage B success",
           "per_policy": per_policy,
           "spearman_pooled_change_vs_success": spearman([r["offline_change_pct"] for r in rows],
                                                         [r["success_pct"] for r in rows]),
           "spearman_within_task": within, "spearman_within_task_mean": float(np.nanmean(within)),
           "rows": rows}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
