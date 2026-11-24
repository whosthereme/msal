import json
import sys
from pathlib import Path

from aggregate_hyb import SHOTS, compare


def load(directory: Path):
    return {(r["seed"], r["held_task"]): r for p in sorted(directory.glob("seed*/folds.jsonl"))
            for r in (json.loads(x) for x in p.read_text().splitlines() if x.strip())}


def main():
    combo_dir, hyb_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    combo, hyb = load(combo_dir), load(hyb_dir)
    valid = [r for r in combo.values() if r["anchor_bit_exact"]]
    repro = {"folds_with_hybrid_record": 0, "hybrid16_equal": 0, "inversion_equal": 0}
    for r in valid:
        h = hyb.get((r["seed"], r["held_task"]))
        if h is None:
            continue
        repro["folds_with_hybrid_record"] += 1
        repro["hybrid16_equal"] += int(r["hybrid_16"]["selection"] == h["hybrid"]["16"]["selection"])
        repro["inversion_equal"] += int(all(r["inversion"][k]["selection"] == h["inversion"][k]["selection"] for k in SHOTS))
    result = {"folds_total": len(combo), "folds_void_anchor": len(combo) - len(valid), "reproduction": repro,
              "comparisons": {}}
    comb = lambda k, s: (lambda r: r["combination"][k][s])
    inv = lambda k, s: (lambda r: r["inversion"][k][s])
    zero = lambda s: (lambda r: r["zero_shot"][s])
    plan = [("C1 combination k=16 vs inversion k=16", "16"), ("C2 combination k=4 vs inversion k=4", "4"),
            ("C3 combination k=1 vs inversion k=1", "1")]
    for label, k in plan:
        result["comparisons"][label] = {s: compare(valid, comb(k, s), inv(k, s)) for s in ("selection", "robustness")}
    for k in SHOTS:
        result["comparisons"][f"curve: combination k={k} vs zero-shot"] = {"selection": compare(valid, comb(k, "selection"), zero("selection"))}
        result["comparisons"][f"curve: combination k={k} vs inversion"] = {"selection": compare(valid, comb(k, "selection"), inv(k, "selection"))}
    paired = [r for r in valid if (r["seed"], r["held_task"]) in hyb]
    for k in SHOTS:
        result["comparisons"][f"curve: combination k={k} vs hybrid"] = {"selection": compare(
            paired, comb(k, "selection"), lambda r, k=k: hyb[(r["seed"], r["held_task"])]["hybrid"][k]["selection"])}
    for label, entry in result["comparisons"].items():
        sel = entry["selection"]
        acr = sel["across_seeds"]
        wins = ", ".join(f"{v['wins']}/{v['folds']}" for v in sel["per_seed"].values())
        rob = f" (rob: {entry['robustness']['reading']})" if "robustness" in entry else ""
        print(f"{label:44s} {acr['pct']:+6.1f}% [{acr['pct_ci95'][0]:+.1f}, {acr['pct_ci95'][1]:+.1f}] wins {wins} -> {sel['reading']}{rob}")
    adds = all(result["comparisons"][lbl]["selection"]["reading"] == "better" for lbl, _ in plan[:2])
    result["library_adds_accuracy_on_top_of_inversion"] = adds
    out.write_text(json.dumps(result, indent=1))
    print("reproduction:", repro, "| void:", result["folds_void_anchor"], "| library adds accuracy (C1 and C2 better):", adds)


if __name__ == "__main__":
    main()
