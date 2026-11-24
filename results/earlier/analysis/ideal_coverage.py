import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T = 2.7764451051977987


def main():
    seen, zero = {}, {40: {}}
    for p in glob.glob(str(ROOT / "results_remote/ideal/seen_seed*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            seen[(r["seed"], r["task"])] = r["flow_selection"]
    for p in glob.glob(str(ROOT / "results_remote/ideal/zero_seed*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            zero[40][(r["seed"], r["task"])] = r["flow_selection"]
    for p in glob.glob(str(ROOT / "results_remote/cov/*_n*_s*/folds.jsonl")):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                zero.setdefault(int(r["level"]), {})[(r["seed"], r["held_task"])] = r["film_nl_float32"]
    plain = {}
    for p in glob.glob(str(ROOT / "results_remote/ideal/stored/blind_nl_plain_nl_seed*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            plain[(r["seed"], r["held_task"])] = r["main_selection"]
    seeds = sorted({s for s, _ in seen})
    tasks = sorted({t for _, t in seen})
    out = {"levels": {}}
    per_seed_curve = {s: [] for s in seeds}
    levels = sorted(zero)
    for n in levels:
        assert len(zero[n]) == len(seeds) * len(tasks), (n, len(zero[n]))
        ps = np.array([np.mean([math.log(zero[n][(s, t)] / seen[(s, t)]) for t in tasks]) for s in seeds])
        for s, v in zip(seeds, ps):
            per_seed_curve[s].append(v)
        h = T * ps.std(ddof=1) / math.sqrt(len(seeds))
        per_task = np.array([np.exp(np.mean([math.log(zero[n][(s, t)] / seen[(s, t)]) for s in seeds])) - 1 for t in tasks])
        out["levels"][n] = {"mean_error": float(np.mean(list(zero[n].values()))), "log_ratio": float(ps.mean()),
                            "excess_pct": 100 * (math.exp(ps.mean()) - 1), "lo": 100 * (math.exp(ps.mean() - h) - 1),
                            "hi": 100 * (math.exp(ps.mean() + h) - 1), "within25": int((per_task <= 0.25).sum()),
                            "best_task_excess_pct": float(100 * per_task.min())}
    out["seen_mean_error"] = float(np.mean(list(seen.values())))
    out["plain_n40_mean_error"] = float(np.mean(list(plain.values())))
    x = np.array(levels, dtype=float)
    fits = {}
    for name, fx in (("linear_in_n", x), ("linear_in_log2_n", np.log2(x))):
        slopes, reach25, reach10 = [], [], []
        for s in seeds:
            b, a = np.polyfit(fx, np.array(per_seed_curve[s]), 1)
            slopes.append(b)
            for target, store in ((math.log(1.25), reach25), (math.log(1.10), reach10)):
                v = (target - a) / b
                store.append(v if name == "linear_in_n" else 2 ** v)
        mean_curve = np.mean([per_seed_curve[s] for s in seeds], axis=0)
        b, a = np.polyfit(fx, mean_curve, 1)
        resid = mean_curve - (a + b * fx)
        fits[name] = {"slope": float(np.mean(slopes)), "slope_ci": [float(np.mean(slopes) - T * np.std(slopes, ddof=1) / math.sqrt(5)),
                                                                   float(np.mean(slopes) + T * np.std(slopes, ddof=1) / math.sqrt(5))],
                      "tasks_to_within25": float(np.mean(reach25)), "tasks_to_within10": float(np.mean(reach10)),
                      "rms_residual": float(np.sqrt((resid ** 2).mean()))}
    out["fits"] = fits
    mean_curve = np.mean([per_seed_curve[s] for s in seeds], axis=0)
    out["steps_log_ratio"] = [float(v) for v in np.diff(mean_curve)]
    (ROOT / "analysis" / "ideal_coverage.json").write_text(json.dumps(out, indent=1))
    for n in levels:
        e = out["levels"][n]
        print(n, "err", round(e["mean_error"], 4), "excess vs seen %", round(e["excess_pct"]), [round(e["lo"]), round(e["hi"])],
              "within25", e["within25"], "best task %", round(e["best_task_excess_pct"]))
    print("seen", round(out["seen_mean_error"], 4), "plain n40", round(out["plain_n40_mean_error"], 4))
    print("steps", [round(v, 3) for v in out["steps_log_ratio"]])
    print(json.dumps(fits, indent=1))


if __name__ == "__main__":
    main()
