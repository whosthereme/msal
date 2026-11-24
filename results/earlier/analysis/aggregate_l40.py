import json
import sys
from pathlib import Path

from aggregate_hyb import SHOTS, compare


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    records = [json.loads(x) for p in sorted(root.glob("seed*/folds.jsonl")) for x in p.read_text().splitlines() if x.strip()]
    keyed = {(r["seed"], r["held_task"]): r for r in records}
    records = list(keyed.values())
    comb = lambda k, s: (lambda r: r["combination"][k][s])
    inv = lambda k, s: (lambda r: r["inversion"][k][s])
    hyb = lambda k, s: (lambda r: r["hybrid"][k][s])
    zero = lambda s: (lambda r: r["zero_shot"][s])
    zf = lambda arm, s: (lambda r: r["zero_shot_float32"][arm][s])
    plan = [("L1 combination k=16 vs inversion k=16", comb("16", "selection"), inv("16", "selection"), True),
            ("L2 combination k=4 vs inversion k=4", comb("4", "selection"), inv("4", "selection"), True),
            ("L3 combination k=16 vs zero-shot R", comb("16", "selection"), zero("selection"), True),
            ("L4 combination k=1 vs inversion k=1", comb("1", "selection"), inv("1", "selection"), False),
            ("L5 zero-shot R vs no task input (same readout)", zf("film_nl", "selection"), zf("plain_nl", "selection"), False),
            ("L6 zero-shot R vs linear-head FiLM", zf("film_nl", "selection"), zf("film", "selection"), False)]
    plan += [(f"curve: {name} k={k} vs zero-shot", f(k, "selection"), zero("selection"), False)
             for k in SHOTS for name, f in (("hybrid", hyb), ("inversion", inv), ("combination", comb))]
    result = {"folds": len(records), "seeds": sorted({r["seed"] for r in records}), "comparisons": {}}
    tests = {"L1": (comb("16", "robustness"), inv("16", "robustness")), "L2": (comb("4", "robustness"), inv("4", "robustness")),
             "L3": (comb("16", "robustness"), zero("robustness"))}
    for label, fa, fb, _ in plan:
        entry = {"selection": compare(records, fa, fb)}
        if label[:2] in tests:
            entry["robustness"] = compare(records, *tests[label[:2]])
        result["comparisons"][label] = entry
        acr = entry["selection"]["across_seeds"]
        wins = ", ".join(f"{v['wins']}/{v['folds']}" for v in entry["selection"]["per_seed"].values())
        rob = f" (test: {entry['robustness']['reading']})" if "robustness" in entry else ""
        print(f"{label:48s} {acr['pct']:+6.1f}% [{acr['pct_ci95'][0]:+.1f}, {acr['pct_ci95'][1]:+.1f}] wins {wins} -> "
              f"{entry['selection']['reading']}{rob}")
    confirmed = all(result["comparisons"][lbl]["selection"]["reading"] == "better" for lbl, *_ in plan[:2])
    result["library_adds_accuracy_confirmed"] = confirmed
    with_unseen = [r for r in records if r.get("unseen_words")]
    result["folds_with_unseen_words"] = len(with_unseen)
    out.write_text(json.dumps(result, indent=1))
    print("folds:", len(records), "| with unseen words:", len(with_unseen), "| confirmed (L1 and L2 better):", confirmed)


if __name__ == "__main__":
    main()
