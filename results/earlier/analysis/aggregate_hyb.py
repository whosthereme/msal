import json
import math
import statistics as st
import sys
from pathlib import Path

from aggregate_n1n2 import seed_summary

SHOTS = ("1", "2", "4", "8", "16")


def reading(across):
    if not across or "pct_ci95" not in across:
        return "incomplete"
    lo, hi = across["pct_ci95"]
    return "better" if hi < 0 else "worse" if lo > 0 else "not distinguishable"


def compare(records, fa, fb):
    per_seed = {}
    for seed in sorted({r["seed"] for r in records}):
        rs = [r for r in records if r["seed"] == seed]
        logs = [math.log(fa(r) / fb(r)) for r in rs]
        per_seed[seed] = {"folds": len(rs), "mean_log_ratio": st.mean(logs),
                          "pct_mean": 100 * (math.exp(st.mean(logs)) - 1), "wins": sum(x < 0 for x in logs)}
    across = seed_summary([v["mean_log_ratio"] for v in per_seed.values()])
    return {"per_seed": per_seed, "across_seeds": across, "reading": reading(across)}


def main():
    hyb_dir, n1n2_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    records = [json.loads(x) for p in sorted(hyb_dir.glob("seed*/folds.jsonl")) for x in p.read_text().splitlines() if x.strip()]
    valid = [r for r in records if r["anchor_bit_exact"]]
    result = {"folds_total": len(records), "folds_void_anchor": len(records) - len(valid), "comparisons": {}}
    sel = lambda block: (lambda r: block(r)["selection"])
    rob = lambda block: (lambda r: block(r)["robustness"])
    zero = lambda r: r["zero_shot"]
    hyb = lambda k: (lambda r: r["hybrid"][k])
    inv = lambda k: (lambda r: r["inversion"][k])
    rand = lambda r: r["random_atoms_16"]
    plan = [
        ("H1 hybrid k=16 vs zero-shot", hyb("16"), zero),
        ("H2 hybrid k=4 vs zero-shot", hyb("4"), zero),
        ("H3 hybrid k=16 vs descriptor inversion k=16", hyb("16"), inv("16")),
        ("H4 hybrid k=16 vs untrained library k=16", hyb("16"), rand),
    ] + [(f"curve: hybrid k={k} vs zero-shot", hyb(k), zero) for k in SHOTS] \
      + [(f"curve: inversion k={k} vs zero-shot", inv(k), zero) for k in SHOTS] \
      + [(f"curve: hybrid vs inversion, k={k}", hyb(k), inv(k)) for k in SHOTS]
    for label, fa, fb in plan:
        entry = {"selection": compare(valid, sel(fa), sel(fb))}
        if label.startswith(("H1", "H2", "H3")):
            entry["robustness"] = compare(valid, rob(fa), rob(fb))
        result["comparisons"][label] = entry
        acr = entry["selection"]["across_seeds"]
        wins = ", ".join(f"{v['wins']}/{v['folds']}" for v in entry["selection"]["per_seed"].values())
        print(f"{label:48s} {acr['pct']:+6.1f}% [{acr['pct_ci95'][0]:+.1f}, {acr['pct_ci95'][1]:+.1f}] "
              f"wins per seed {wins}"
              f" -> {entry['selection']['reading']}" + (f" (rob: {entry['robustness']['reading']})" if "robustness" in entry else ""))
    stored = {}
    for p in n1n2_dir.glob("*/folds.jsonl"):
        if "anchor" in p.parent.name or "SMOKE" in p.parent.name:
            continue
        for x in p.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                stored.setdefault(r["arm"], {})[(r["seed"], int(r["held_task"]))] = r["main_selection"]
    for arm in ("sal_nlb", "sal", "film"):
        if arm in stored:
            paired = [r for r in valid if (r["seed"], r["held_task"]) in stored[arm]]
            entry = compare(paired, lambda r: r["hybrid"]["16"]["selection"], lambda r, a=arm: stored[a][(r["seed"], r["held_task"])])
            result["comparisons"][f"H5 hybrid k=16 vs stored zero-shot {arm}"] = {"selection": entry}
            acr = entry["across_seeds"]
            print(f"{'H5 hybrid k=16 vs stored ' + arm:48s} {acr['pct']:+6.1f}% [{acr['pct_ci95'][0]:+.1f}, {acr['pct_ci95'][1]:+.1f}] -> {entry['reading']}")
    out.write_text(json.dumps(result, indent=1))
    print("folds:", len(records), "void (anchor):", result["folds_void_anchor"])


if __name__ == "__main__":
    main()
