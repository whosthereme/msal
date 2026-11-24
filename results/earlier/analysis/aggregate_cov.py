import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("COV_RESULTS", ROOT / "results_remote"))
SEEDS = (148021, 148022, 148023, 148024, 148025)
LEVELS = (8, 16, 24, 32, 40)
FOLDS = 41


def load():
    out = defaultdict(dict)
    for path in sorted((RUNS / "cov").glob("*/folds.jsonl")):
        if path.parent.name == "anchor" or path.parent.name.endswith("_SMOKE"):
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["seed"], r["level"])][r["held_task"]] = r
    for seed in SEEDS:
        path = RUNS / "combo" / f"seed{seed}" / "folds.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            if line.strip():
                r = json.loads(line)
                r["hybrid"] = {"16": r["hybrid_16"]}
                out[(seed, 40)][r["held_task"]] = r
    return out


COMPARISONS = {
    "V1 MSAL vs inversion, k=16": (lambda r: r["combination"]["16"]["selection"], lambda r: r["inversion"]["16"]["selection"]),
    "V2 MSAL vs inversion, k=1": (lambda r: r["combination"]["1"]["selection"], lambda r: r["inversion"]["1"]["selection"]),
    "V3 library only vs zero-shot, k=16": (lambda r: r["hybrid"]["16"]["selection"], lambda r: r["zero_shot"]["selection"]),
    "V4 MSAL vs zero-shot, k=16": (lambda r: r["combination"]["16"]["selection"], lambda r: r["zero_shot"]["selection"]),
}


def across(per_seed):
    x = np.asarray(per_seed, dtype=float)
    m = x.mean()
    if len(x) < 2:
        return {"n_seeds": len(x), "pct": 100 * (math.exp(m) - 1)}
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    lo, hi = m - half, m + half
    reading = "better" if hi < 0 else "worse" if lo > 0 else "not distinguishable"
    return {"n_seeds": len(x), "mean": m, "ci95": [lo, hi], "pct": 100 * (math.exp(m) - 1),
            "pct_ci95": [100 * (math.exp(lo) - 1), 100 * (math.exp(hi) - 1)], "reading": reading}


def main():
    runs = load()
    result = {"folds": {f"{s}_n{n}": len(runs.get((s, n), {})) for s in SEEDS for n in LEVELS}, "comparisons": {}}
    complete = all(len(runs.get((s, n), {})) == FOLDS for s in SEEDS for n in LEVELS)
    fold_means = {}
    for name, (num, den) in COMPARISONS.items():
        result["comparisons"][name] = {}
        for n in LEVELS:
            per_seed, wins, total = [], 0, 0
            for s in SEEDS:
                recs = runs.get((s, n), {})
                if not recs:
                    continue
                logs = np.array([math.log(num(r) / den(r)) for r in recs.values()])
                per_seed.append(logs.mean())
                fold_means[(name, n, s)] = logs.mean()
                wins += int((logs < 0).sum())
                total += len(logs)
            if per_seed:
                result["comparisons"][name][str(n)] = {**across(per_seed), "folds_won": wins, "folds": total}
    slopes = []
    for s in SEEDS:
        pts = [(math.log2(n), fold_means[("V1 MSAL vs inversion, k=16", n, s)]) for n in LEVELS
               if ("V1 MSAL vs inversion, k=16", n, s) in fold_means]
        if len(pts) >= 3:
            xs, ys = np.array(pts).T
            slopes.append(float(np.polyfit(xs, ys, 1)[0]))
    if slopes:
        t = across(slopes)
        t["reading"] = ("the library's gain grows with coverage" if t.get("ci95", [0, 0])[1] < 0 else
                        "the library's gain shrinks with coverage" if t.get("ci95", [0, 0])[0] > 0 else
                        "no distinguishable trend")
        result["trend_V1_slope_per_doubling"] = {**t, "per_seed": slopes}
    rel = {}
    for n in LEVELS[:-1]:
        per_seed = []
        for s in SEEDS:
            a, b = runs.get((s, n), {}), runs.get((s, 40), {})
            common = sorted(set(a) & set(b))
            if common:
                per_seed.append(np.mean([math.log(a[h]["zero_shot"]["selection"] / b[h]["zero_shot"]["selection"])
                                         for h in common]))
        if per_seed:
            rel[str(n)] = across(per_seed)
    result["zero_shot_vs_n40"] = rel
    out = (RUNS if "COV_RESULTS" in os.environ else ROOT / "analysis") / ("cov_final.json" if complete else "cov_interim.json")
    out.write_text(json.dumps(result, indent=1))
    for name, levels in result["comparisons"].items():
        print(name)
        for n, v in levels.items():
            ci = v.get("pct_ci95")
            print(f"  n={n:>2}: {v['pct']:+6.1f}%" + (f" [{ci[0]:+.1f}, {ci[1]:+.1f}] {v['reading']}" if ci else "")
                  + f"  won {v['folds_won']}/{v['folds']}")
    if "trend_V1_slope_per_doubling" in result:
        t = result["trend_V1_slope_per_doubling"]
        print("trend V1 slope per doubling of n:", round(t["mean"], 4) if "mean" in t else t, t.get("reading"))
    print("zero-shot vs n=40:", {n: round(v["pct"], 1) for n, v in rel.items()})
    print("written", out, "(complete)" if complete else "(interim)")


if __name__ == "__main__":
    main()
