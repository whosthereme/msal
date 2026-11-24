import csv, json, math, statistics as st
from pathlib import Path
P = Path("D:/manuscript6/paper_package_20261021_windows_handoff/data/tables")
rows = [r for r in csv.DictReader(open(P / "fig2_descriptor_transfer.csv")) if r["panel"] == "a"]
out = {}
T95 = {2: 4.302652729911275, 4: 2.7764451051977987}
for seed in sorted({r["seed"] for r in rows}):
    sub = [r for r in rows if r["seed"] == seed]
    sal = [float(r["sal_mse"]) for r in sub]; nn = [float(r["nearest_mse"]) for r in sub]; mean = [float(r["mean_mse"]) for r in sub]
    lr_nn = [math.log(a / b) for a, b in zip(sal, nn)]; lr_mean = [math.log(a / b) for a, b in zip(sal, mean)]
    lr_best = [math.log(a / min(b, c)) for a, b, c in zip(sal, nn, mean)]
    q = lambda xs, p: sorted(xs)[int(round(p * (len(xs) - 1)))]
    out[seed] = {
        "n_tasks": len(sub),
        "mean_logratio_vs_nn": st.mean(lr_nn), "mean_logratio_vs_mean": st.mean(lr_mean), "mean_logratio_vs_stronger": st.mean(lr_best),
        "median_rel_change_vs_stronger_pct": 100 * (math.exp(st.median(lr_best)) - 1),
        "q10_q90_rel_change_vs_stronger_pct": [100 * (math.exp(q(lr_best, .1)) - 1), 100 * (math.exp(q(lr_best, .9)) - 1)],
        "wins_vs_both": sum(x < 0 for x in lr_best),
        "wins_but_under_1pct": sum(-math.log(1 / 0.99) < x < 0 for x in lr_best),
        "losses": sum(x > 0 for x in lr_best),
        "worst_loss_pct": 100 * (math.exp(max(lr_best)) - 1),
        "mean_mse_sal": st.mean(sal), "mean_mse_nn": st.mean(nn), "mean_mse_mean": st.mean(mean),
    }
seeds = sorted(out)
for key in ("mean_logratio_vs_nn", "mean_logratio_vs_mean", "mean_logratio_vs_stronger"):
    vals = [out[s][key] for s in seeds]; m = st.mean(vals); sd = st.stdev(vals); h = T95[len(vals) - 1] * sd / math.sqrt(len(vals))
    out[f"across_seeds_{key}"] = {"mean": m, "sd": sd, "ci95": [m - h, m + h], "pct_change": 100 * (math.exp(m) - 1)}
print(json.dumps(out, indent=1))
