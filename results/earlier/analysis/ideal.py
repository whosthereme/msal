import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results_remote" / "ideal"
T975_DF4 = 2.7764451051977987


def load(pattern):
    out = {}
    for p in sorted(glob.glob(str(SRC / pattern))):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                out[(int(r["seed"]), int(r.get("task", r.get("held_task"))))] = r
    return out


def pooled(logr, seeds, tasks):
    per_seed = np.array([np.mean([logr[(s, t)] for t in tasks]) for s in seeds])
    m = per_seed.mean()
    if len(seeds) < 2:
        return {"excess_pct": 100 * (math.exp(m) - 1), "seeds": len(seeds)}
    h = T975_DF4 * per_seed.std(ddof=1) / math.sqrt(len(seeds)) if len(seeds) == 5 else float("nan")
    return {"excess_pct": 100 * (math.exp(m) - 1), "lo_pct": 100 * (math.exp(m - h) - 1),
            "hi_pct": 100 * (math.exp(m + h) - 1), "seeds": len(seeds)}


def main():
    seen, zero = load("seen_seed*.jsonl"), load("zero_seed*.jsonl")
    plain = load("stored/blind_nl_plain_nl_seed*.jsonl")
    keys = sorted(set(seen) & set(zero))
    seeds = sorted({s for s, _ in keys})
    tasks = sorted({t for _, t in keys})
    seeds = [s for s in seeds if all((s, t) in zero and (s, t) in seen for t in tasks)]
    out = {"seeds": seeds, "tasks": len(tasks), "folds": len(seeds) * len(tasks),
           "anchor_mismatch": [k for k in keys if not zero[k]["anchor_bit_exact"]]}
    for metric, field in (("action", "action_selection"), ("flow", "flow_selection"), ("flow_robustness", "flow_robustness")):
        logr = {(s, t): math.log(zero[(s, t)][field] / seen[(s, t)][field]) for s in seeds for t in tasks}
        per_task = {t: 100 * (math.exp(np.mean([logr[(s, t)] for s in seeds])) - 1) for t in tasks}
        vals = np.array([per_task[t] for t in tasks])
        dist = np.array([zero[(seeds[0], t)]["nearest_source_distance_m"] for t in tasks])
        near = dist <= np.median(dist)
        entry = {"pooled": pooled(logr, seeds, tasks), "within10": int((vals <= 10).sum()), "within25": int((vals <= 25).sum()),
                 "per_task_excess_pct": {int(t): round(float(v), 2) for t, v in per_task.items()},
                 "median_excess_pct": float(np.median(vals)), "max_excess_pct": float(vals.max()),
                 "near_half": {"n": int(near.sum()), "median_excess_pct": float(np.median(vals[near])), "within10": int((vals[near] <= 10).sum())},
                 "far_half": {"n": int((~near).sum()), "median_excess_pct": float(np.median(vals[~near])), "within10": int((vals[~near] <= 10).sum())},
                 "mean_error_zero": float(np.mean([zero[(s, t)][field] for s in seeds for t in tasks])),
                 "mean_error_seen": float(np.mean([seen[(s, t)][field] for s in seeds for t in tasks]))}
        out[metric] = entry
    if all((s, t) in plain for s in seeds for t in tasks):
        e_p = np.mean([plain[(s, t)]["main_selection"] for s in seeds for t in tasks])
        e_z, e_s = out["flow"]["mean_error_zero"], out["flow"]["mean_error_seen"]
        out["flow_gap_closed_from_plain"] = float((e_p - e_z) / (e_p - e_s))
        out["flow_mean_error_plain"] = float(e_p)
    a = out["action"]
    tier = "strong" if a["within10"] >= 31 else "moderate" if a["within25"] >= 21 else "not supported"
    f = out["flow"]
    tier_flow = "strong" if f["within10"] >= 31 else "moderate" if f["within25"] >= 21 else "not supported"
    order = ["not supported", "moderate", "strong"]
    out["tier_action"], out["tier_flow"] = tier, tier_flow
    out["tier_claim"] = order[min(order.index(tier), order.index(tier_flow))]
    (ROOT / "analysis" / "ideal.json").write_text(json.dumps(out, indent=1))
    for m in ("action", "flow", "flow_robustness"):
        e = out[m]
        print(m, "pooled", {k: round(v, 2) for k, v in e["pooled"].items()}, "within10", e["within10"], "within25", e["within25"],
              "median", round(e["median_excess_pct"], 1), "max", round(e["max_excess_pct"], 1),
              "near", e["near_half"], "far", e["far_half"], "E_zero", round(e["mean_error_zero"], 5), "E_seen", round(e["mean_error_seen"], 5))
    print("gap closed from plain (flow):", out.get("flow_gap_closed_from_plain"))
    print("seeds", seeds, "tasks", len(tasks), "anchor mismatches", len(out["anchor_mismatch"]),
          "tiers", out["tier_action"], out["tier_flow"], "->", out["tier_claim"])


if __name__ == "__main__":
    main()
