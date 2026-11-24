import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

T975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105, 5: 2.570581836}
REF = "blind/sal"
COMPARISONS = [
    "blind/ctx_add", "blind/film", "blind/bilinear", "blind/sal_noreg",
    "blind/plain", "lang/plain", "lang/sal", "blind/sal_word",
    "blind/sal_tied",
]
EXTRA_PAIRS = [
    ("lang/plain", "blind/plain"),
    ("lang/sal", "lang/plain"),
    ("blind/sal_word", "blind/sal"),
    ("blind/film", "blind/ctx_add"),
    ("blind/bilinear", "blind/film"),
    ("blind/sal_noreg", "blind/bilinear"),
    ("blind/plain", "blind/ctx_add"),
    ("blind/sal_tied", "blind/film"),
]


def load(runs_dir: Path):
    records = []
    for path in sorted(runs_dir.glob("*/folds.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["key"] = f"{r['feature']}/{r['arm']}"
                records.append(r)
    return records


def table(records):
    by = defaultdict(dict)
    for r in records:
        by[r["key"]][(r["seed"], r["held_task"])] = r
    return by


def seed_summary(values):
    n = len(values)
    mean = st.mean(values)
    if n < 2:
        return {"n_seeds": n, "mean": mean}
    sd = st.stdev(values)
    half = T975[n - 1] * sd / math.sqrt(n)
    return {"n_seeds": n, "mean": mean, "sd": sd, "ci95": [mean - half, mean + half],
            "pct": 100 * (math.exp(mean) - 1), "pct_ci95": [100 * (math.exp(mean - half) - 1), 100 * (math.exp(mean + half) - 1)]}


def paired(by, a, b, score="main_selection", require_complete=41):
    per_seed = {}
    for seed in sorted({s for s, _ in by.get(a, {})} & {s for s, _ in by.get(b, {})}):
        keys = [k for k in by[a] if k[0] == seed and k in by[b]]
        if len(keys) < require_complete:
            continue
        logs = [math.log(by[a][k][score] / by[b][k][score]) for k in keys]
        per_seed[seed] = {
            "folds": len(keys),
            "mean_log_ratio": st.mean(logs),
            "pct_mean": 100 * (math.exp(st.mean(logs)) - 1),
            "median_pct": 100 * (math.exp(st.median(logs)) - 1),
            "wins": sum(x < 0 for x in logs),
            "matches_within_1pct": sum(by[a][k][score] <= 1.01 * by[b][k][score] for k in keys),
            "worst_pct": 100 * (math.exp(max(logs)) - 1),
            "best_pct": 100 * (math.exp(min(logs)) - 1),
            "mean_mse_a": st.mean(by[a][k][score] for k in keys),
            "mean_mse_b": st.mean(by[b][k][score] for k in keys),
        }
    across = seed_summary([v["mean_log_ratio"] for v in per_seed.values()]) if per_seed else None
    return {"a": a, "b": b, "score": score, "per_seed": per_seed, "across_seeds": across}


def reuse_baselines(by, key, score_block="selection"):
    out = {}
    for name in ("nn", "mean"):
        per_seed = {}
        for seed in sorted({s for s, _ in by.get(key, {})}):
            recs = [r for (s, _), r in by[key].items() if s == seed]
            if len(recs) < 41:
                continue
            logs = [math.log(r[score_block]["phi"] / r[score_block][name]) for r in recs]
            both = sum(r[score_block]["phi"] < min(r[score_block]["nn"], r[score_block]["mean"]) for r in recs)
            per_seed[seed] = {"mean_log_ratio": st.mean(logs), "pct_mean": 100 * (math.exp(st.mean(logs)) - 1),
                              "wins": sum(x < 0 for x in logs), "wins_both": both}
        out[name] = {"per_seed": per_seed,
                     "across_seeds": seed_summary([v["mean_log_ratio"] for v in per_seed.values()]) if per_seed else None}
    return out


def main():
    runs_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    records = load(runs_dir)
    by = table(records)
    census = {k: {str(s): sum(1 for (ss, _) in v if ss == s) for s in sorted({s for s, _ in v})} for k, v in sorted(by.items())}
    result = {"census": census, "reference": REF, "vs_reference": {}, "extra_pairs": {}, "reuse_baselines": {}, "mean_mse": {}, "parameters": {}}
    for key in COMPARISONS:
        if key in by:
            result["vs_reference"][key] = {s: paired(by, key, REF, score=s) for s in ("main_selection", "main_robustness")}
    for a, b in EXTRA_PAIRS:
        if a in by and b in by:
            result["extra_pairs"][f"{a} vs {b}"] = {s: paired(by, a, b, score=s) for s in ("main_selection", "main_robustness")}
    for key in by:
        if "phi" in next(iter(by[key].values()))["selection"]:
            result["reuse_baselines"][key] = {blk: reuse_baselines(by, key, blk) for blk in ("selection", "robustness")}
        seeds = sorted({s for s, _ in by[key]})
        result["mean_mse"][key] = {str(s): st.mean(r["main_selection"] for (ss, _), r in by[key].items() if ss == s) for s in seeds}
        any_rec = next(iter(by[key].values()))
        result["parameters"][key] = {"trainable": any_rec["trainable_parameters"], "task_conditioned": any_rec["task_conditioned_parameters"]}
        refs = [r for r in by[key].values() if "reference" in r]
        if refs:
            diffs = [abs(r["main_selection"] - (r["reference"]["phi"] if isinstance(r["reference"], dict) else r["reference"])) for r in refs]
            result.setdefault("reproduction", {})[key] = {"folds_with_reference": len(refs), "max_abs_diff": max(diffs),
                                                          "bit_exact_folds": sum(d == 0.0 for d in diffs)}
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps({"census": census, "reproduction": result.get("reproduction")}, indent=1))


if __name__ == "__main__":
    main()
