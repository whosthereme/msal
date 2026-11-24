import json
from pathlib import Path

from aggregate_ft import FOLDS, SHOTS, compare, load, pooled_log_ratio, settings_for

ROOT = Path(__file__).resolve().parent.parent


def main():
    ft, msal = load()
    out = {"selected_readout": {}, "comparisons": {}}
    for k in SHOTS:
        keys = [key for key in settings_for(ft, "dev", k) if "_readout_" in key]
        sel = min(keys, key=lambda key: pooled_log_ratio(ft, "dev", key))
        out["selected_readout"][str(k)] = sel
        sc = lambda f, m, sel=sel: f["finetune"][sel]["selection"]
        msal_k = lambda f, m, k=k: m["combination"][str(k)]["selection"]
        zero = lambda f, m: f["zero_shot"]["selection"]
        for family in FOLDS:
            c = out["comparisons"].setdefault(family, {}).setdefault(str(k), {})
            c["readout fine-tuning vs zero-shot"] = compare(ft, msal, family, sc, zero)
            c["MSAL vs readout fine-tuning"] = compare(ft, msal, family, msal_k, sc)
            v, w = c["readout fine-tuning vs zero-shot"], c["MSAL vs readout fine-tuning"]
            print(f"k={k:>2} {family:3s} {sel:28s} readout vs zero {v['pct']:+6.1f}% [{v['pct_ci95'][0]:+.1f}, "
                  f"{v['pct_ci95'][1]:+.1f}] | MSAL vs readout {w['pct']:+6.1f}% [{w['pct_ci95'][0]:+.1f}, "
                  f"{w['pct_ci95'][1]:+.1f}] {w['reading']}, MSAL won {w['folds_won']}/{w['folds']}")
    (ROOT / "analysis" / "ft_readout.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
