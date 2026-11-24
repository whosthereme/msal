import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("FT_RESULTS", ROOT / "results_remote"))
SEEDS = (148021, 148022, 148023, 148024, 148025)
SHOTS = (1, 2, 4, 8, 16)
FOLDS = {"dev": 41, "l40": 40}
KEY = re.compile(r"k(\d+)_(readout|all)_lr([0-9.e-]+)_s(\d+)$")


def read_jsonl(paths):
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def load():
    ft = defaultdict(dict)
    for r in read_jsonl(p for p in sorted((RUNS / "ft").glob("*/folds.jsonl"))
                        if not p.parent.name.startswith("anchor") and not p.parent.name.endswith("_SMOKE")):
        ft[(r["family"], r["seed"])][r["held_task"]] = r
    msal = defaultdict(dict)
    for r in read_jsonl(sorted((RUNS / "combo").glob("seed*/folds.jsonl"))):
        msal[("dev", r["seed"])][r["held_task"]] = r
    for r in read_jsonl(sorted((RUNS / "l40").glob("seed*/folds.jsonl"))):
        msal[("l40", r["seed"])][r["held_task"]] = r
    return ft, msal


def across(per_seed):
    x = np.asarray(per_seed, dtype=float)
    m = x.mean()
    if len(x) < 2:
        return {"n_seeds": len(x), "pct": 100 * (math.exp(m) - 1)}
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    lo, hi = m - half, m + half
    return {"n_seeds": len(x), "mean": m, "ci95": [lo, hi], "pct": 100 * (math.exp(m) - 1),
            "pct_ci95": [100 * (math.exp(lo) - 1), 100 * (math.exp(hi) - 1)],
            "reading": "better" if hi < 0 else "worse" if lo > 0 else "not distinguishable"}


def compare(ft, msal, family, num, den):
    per_seed, wins, total = [], 0, 0
    for s in SEEDS:
        a, b = ft.get((family, s), {}), msal.get((family, s), {})
        common = sorted(set(a) & set(b))
        if not common:
            continue
        logs = np.array([math.log(num(a[h], b[h]) / den(a[h], b[h])) for h in common])
        per_seed.append(logs.mean())
        wins += int((logs < 0).sum())
        total += len(logs)
    return {**across(per_seed), "folds_won": wins, "folds": total} if per_seed else None


def settings_for(ft, family, k):
    keys = set()
    for recs in (v for (f, _), v in ft.items() if f == family):
        for r in recs.values():
            keys |= {key for key in r["finetune"] if KEY.match(key) and int(KEY.match(key).group(1)) == k}
    return sorted(keys)


def pooled_log_ratio(ft, family, key):
    vals = [math.log(r["finetune"][key]["selection"] / r["zero_shot"]["selection"])
            for (f, _), recs in ft.items() if f == family for r in recs.values() if key in r["finetune"]]
    return float(np.mean(vals)) if vals else float("inf")


def main():
    ft, msal = load()
    complete = all(len(ft.get((f, s), {})) == n for f, n in FOLDS.items() for s in SEEDS)
    out = {"folds": {f"{f}_{s}": len(ft.get((f, s), {})) for f in FOLDS for s in SEEDS},
           "anchors_bit_exact": {f: sum(r["anchor_bit_exact"] for (g, _), recs in ft.items() if g == f
                                        for r in recs.values()) for f in FOLDS},
           "selected": {}, "comparisons": {}, "settings_vs_zero_shot": {}, "cost_seconds": {}}
    for k in SHOTS:
        keys = settings_for(ft, "dev", k)
        if not keys:
            continue
        sel = min(keys, key=lambda key: pooled_log_ratio(ft, "dev", key))
        out["selected"][str(k)] = sel
        l40_best = min(settings_for(ft, "l40", k) or [sel], key=lambda key: pooled_log_ratio(ft, "l40", key))
        sc = lambda key: (lambda f, m: f["finetune"][key]["selection"])
        msal_k = lambda f, m, k=k: m["combination"][str(k)]["selection"]
        inv_k = lambda f, m, k=k: m["inversion"][str(k)]["selection"]
        zero = lambda f, m: f["zero_shot"]["selection"]
        for family in FOLDS:
            c = out["comparisons"].setdefault(family, {}).setdefault(str(k), {})
            c["MSAL vs selected fine-tuning"] = compare(ft, msal, family, msal_k, sc(sel))
            c["selected fine-tuning vs zero-shot"] = compare(ft, msal, family, sc(sel), zero)
            c["selected fine-tuning vs inversion"] = compare(ft, msal, family, sc(sel), inv_k)
            if family == "l40":
                c["exploratory: MSAL vs LIBERO40-best fine-tuning"] = compare(ft, msal, family, msal_k, sc(l40_best))
                c["exploratory: LIBERO40-best setting"] = l40_best
            out["settings_vs_zero_shot"].setdefault(family, {})[str(k)] = {
                key: compare(ft, msal, family, sc(key), zero) for key in settings_for(ft, family, k)}
            secs = [r["finetune"][sel].get("training_seconds") for (f, _), recs in ft.items() if f == family
                    for r in recs.values() if sel in r["finetune"]]
            secs = [s for s in secs if s is not None]
            if secs:
                out["cost_seconds"].setdefault(family, {})[str(k)] = {"selected_setting_median": float(np.median(secs))}
    path = (RUNS if "FT_RESULTS" in os.environ else ROOT / "analysis") / ("ft_final.json" if complete else "ft_interim.json")
    path.write_text(json.dumps(out, indent=1))
    print("anchors bit-exact:", out["anchors_bit_exact"], "| folds:", out["folds"])
    for k, sel in out["selected"].items():
        print(f"k={k:>2} selected on dev: {sel}")
        for family in FOLDS:
            c = out["comparisons"].get(family, {}).get(k, {})
            for name in ("MSAL vs selected fine-tuning", "selected fine-tuning vs zero-shot",
                         "selected fine-tuning vs inversion", "exploratory: MSAL vs LIBERO40-best fine-tuning"):
                v = c.get(name)
                if v and "pct_ci95" in v:
                    print(f"    {family:3s} {name:48s} {v['pct']:+6.1f}% [{v['pct_ci95'][0]:+.1f}, {v['pct_ci95'][1]:+.1f}] "
                          f"{v['reading']}, won {v['folds_won']}/{v['folds']}")
    print("written", path, "(complete)" if complete else "(interim)")


if __name__ == "__main__":
    main()
