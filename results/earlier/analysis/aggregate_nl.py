import json
import statistics as st
import sys
from pathlib import Path

from aggregate_n1n2 import paired, reuse_baselines, table

PAIRS = [
    ("(1) primary: nonlinear library vs FiLM", "blind/sal_nl", "blind/film"),
    ("(2) primary: nonlinear vs linear library", "blind/sal_nl", "blind/sal"),
    ("(3) capacity control: vs FiLM + nonlinear readout", "blind/sal_nl", "blind/film_nl"),
    ("(4) capacity control: vs no task input + nonlinear readout", "blind/sal_nl", "blind/plain_nl"),
    ("(5) secondary: penalties", "blind/sal_nl", "blind/sal_nl_noreg"),
    ("(8) mixture-of-experts control: signed codes vs softmax router", "blind/sal_nl", "blind/moe_desc"),
    ("descriptive: FiLM + nonlinear readout vs FiLM", "blind/film_nl", "blind/film"),
    ("descriptive: no task input + nonlinear readout vs no task input", "blind/plain_nl", "blind/plain"),
    ("descriptive: softmax mixture of experts vs FiLM", "blind/moe_desc", "blind/film"),
    ("descriptive: FiLM + nonlinear readout vs no task input + nonlinear readout", "blind/film_nl", "blind/plain_nl"),
    ("exploratory: nonlinear-base library vs FiLM + nonlinear readout", "blind/sal_nlb", "blind/film_nl"),
    ("exploratory: nonlinear-base library vs nonlinear library", "blind/sal_nlb", "blind/sal_nl"),
    ("exploratory: nonlinear-base library vs no task input + nonlinear readout", "blind/sal_nlb", "blind/plain_nl"),
    ("exploratory: nonlinear-base library vs softmax mixture of experts", "blind/sal_nlb", "blind/moe_desc"),
]


def load(runs_dir: Path):
    records = []
    for path in sorted(runs_dir.glob("*/folds.jsonl")):
        if "anchor" in path.parent.name or "SMOKE" in path.parent.name:
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["key"] = f"{r['feature']}/{r['arm']}"
                records.append(r)
    return records


def reading(across):
    if not across or "pct_ci95" not in across:
        return "incomplete"
    lo, hi = across["pct_ci95"]
    return "better" if hi < 0 else "worse" if lo > 0 else "not distinguishable"


def main():
    runs_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    by = table(load(runs_dir))
    result = {"comparisons": {}, "reuse_baselines": {}, "linear_map_residual": {}, "census": {}}
    for key in sorted(k for k in by if k.split("/")[1] in ("sal_nl", "sal_nl_noreg", "film_nl", "plain_nl", "moe_desc", "sal_nlb")):
        result["census"][key] = {str(s): sum(1 for (ss, _) in by[key] if ss == s) for s in sorted({s for s, _ in by[key]})}
    for label, a, b in PAIRS:
        if a not in by or b not in by:
            continue
        entry = {s: paired(by, a, b, score=s) for s in ("main_selection", "main_robustness")}
        entry["reading_selection"] = reading(entry["main_selection"]["across_seeds"])
        entry["reading_robustness"] = reading(entry["main_robustness"]["across_seeds"])
        result["comparisons"][label] = entry
        sel = entry["main_selection"]
        acr = sel["across_seeds"]
        wins = [v["wins"] for v in sel["per_seed"].values()]
        text = (f"{acr['pct']:+.1f}% [{acr['pct_ci95'][0]:+.1f}, {acr['pct_ci95'][1]:+.1f}] seeds={acr['n_seeds']} "
                f"wins {min(wins)}-{max(wins)}/41" if acr and "pct" in acr else f"seeds={len(sel['per_seed'])}")
        print(f"{label:62s} {text}  -> {entry['reading_selection']} (rob: {entry['reading_robustness']})")
    for key in ("blind/sal_nl", "blind/sal_nl_noreg", "blind/sal_nlb"):
        if key in by:
            result["reuse_baselines"][key] = {blk: reuse_baselines(by, key, blk) for blk in ("selection", "robustness")}
            res = [r["linear_map_residual"] for r in by[key].values() if "linear_map_residual" in r]
            if res:
                result["linear_map_residual"][key] = {"folds": len(res), "min": min(res), "median": st.median(res)}
    claim = all(result["comparisons"].get(lbl, {}).get("reading_selection") == "better"
                for lbl in (PAIRS[0][0], PAIRS[2][0], PAIRS[3][0]))
    result["method_claim_library_beats_film_beyond_capacity"] = claim
    result["claim_signed_codes_beat_softmax_router"] = (
        result["comparisons"].get(PAIRS[5][0], {}).get("reading_selection") == "better")
    out_path.write_text(json.dumps(result, indent=1))
    print("method claim (1),(3),(4) all better:", claim)
    print("signed codes beat softmax router (8):", result["claim_signed_codes_beat_softmax_router"])
    print("linear-map residual:", result["linear_map_residual"])


if __name__ == "__main__":
    main()
