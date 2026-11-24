from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BOOT, BOOT_SEED = 10_000, 20261101
POLICIES = ("zero", "msal_k1", "msal_k16", "ft_k1", "ft_k16", "seen")
COMPARISONS = [("C1", "msal_k16", "zero", "primary"), ("C2", "msal_k1", "zero", "primary"),
               ("C3", "msal_k16", "ft_k16", "secondary"), ("C4", "msal_k1", "ft_k1", "secondary"),
               ("C5", "ft_k16", "zero", "secondary"), ("C6", "ft_k1", "zero", "secondary")]


def read(paths):
    recs = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def table(recs, key=lambda r: r["policy"]):
    out = defaultdict(dict)
    for r in recs:
        out[key(r)].setdefault(r["task"], {})[r["episode"]] = int(r["success"])
    return out


def stage_a(results: Path) -> dict:
    res = {}
    a1 = read(sorted(results.glob("A1_*.jsonl")))
    if a1:
        t = table(a1, key=lambda r: f"n_exec={r['n_exec']}")
        res["A1"] = {s: {"success_pct": 100 * np.mean([v for task in per.values() for v in task.values()]),
                         "episodes": sum(len(task) for task in per.values())} for s, per in t.items()}
    a2 = read(sorted(results.glob("A2_*.jsonl")))
    if a2:
        per_task = {task: 100 * np.mean(list(eps.values())) for task, eps in table(a2)["seen"].items()}
        mean = float(np.mean(list(per_task.values())))
        n40 = int(sum(v >= 40 for v in per_task.values()))
        res["A2"] = {"tasks": len(per_task), "mean_success_pct": mean, "tasks_at_least_40pct": n40,
                     "per_task_pct": per_task, "gate_passed": len(per_task) == 40 and mean >= 20 and n40 >= 8}
    return res


def compare(t, a, b, index):
    tasks = sorted(set(t.get(a, {})) & set(t.get(b, {})))
    if not tasks:
        return None
    pairs = []
    for task in tasks:
        eps = sorted(set(t[a][task]) & set(t[b][task]))
        pairs.append(np.array([[t[a][task][e], t[b][task][e]] for e in eps], dtype=float))
    diff = 100 * np.mean([p[:, 0].mean() - p[:, 1].mean() for p in pairs])
    rng = np.random.default_rng([BOOT_SEED, index])
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(pairs), len(pairs))
        vals = []
        for j in pick:
            p = pairs[j]
            e = rng.integers(0, len(p), len(p))
            vals.append(p[e, 0].mean() - p[e, 1].mean())
        boot[i] = 100 * np.mean(vals)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"a": a, "b": b, "tasks": len(tasks), "episodes": int(sum(len(p) for p in pairs)),
            "diff_pp": float(diff), "boot95_pp": [float(lo), float(hi)],
            "reading": "better" if lo > 0 else "worse" if hi < 0 else "not distinguishable",
            "tasks_a_better": int(sum(p[:, 0].mean() > p[:, 1].mean() for p in pairs)),
            "tasks_b_better": int(sum(p[:, 0].mean() < p[:, 1].mean() for p in pairs))}


def stage_b(results: Path) -> dict:
    recs = read(sorted(results.glob("B_*.jsonl")))
    if not recs:
        return {}
    t = table(recs)
    success = {p: {task: 100 * np.mean(list(eps.values())) for task, eps in t[p].items()} for p in POLICIES if p in t}
    comps = {name: {**(compare(t, a, b, i) or {}), "role": role} for i, (name, a, b, role) in enumerate(COMPARISONS)}
    desc = {f"{p} vs seen": compare(t, p, "seen", 100 + i) for i, p in enumerate(POLICIES[:-1]) if p in t}
    return {"success_pct": {p: {"overall": float(np.mean(list(v.values()))), "per_task": v} for p, v in success.items()},
            "comparisons": comps, "descriptive": desc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "cl_adapt.json")
    args = ap.parse_args()
    out = {"stage_a": stage_a(args.results), "stage_b": stage_b(args.results)}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
