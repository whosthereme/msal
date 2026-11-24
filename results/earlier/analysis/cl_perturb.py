import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RR = ROOT / "results_remote"
N_BOOT, SEED = 10_000, 20261102
POLICIES = ("msal_k1", "msal_k16", "bf_k16", "ret_replay_k1", "ret_replay_k16", "ft_k16", "seen")
TASKS = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)


def load():
    s = defaultdict(dict)
    def add(path, level, policy=None):
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                p = policy or r["policy"]
                if r["episode"] < 50:
                    s[(p, level)][(r["task"], r["episode"])] = int(r["success"])
    for f in glob.glob(str(RR / "cl_adapt" / "B_*.jsonl")):
        add(f, 0)
    for f in glob.glob(str(RR / "upgrades" / "retrieval" / "results" / "R_ret_replay_k*_task*.jsonl")):
        add(f, 0)
    for f in glob.glob(str(RR / "upgrades" / "behavfit" / "bf_k16_task*.jsonl")):
        add(f, 0)
    for f in glob.glob(str(RR / "perturb" / "P_L*_*_task*.jsonl")):
        name = Path(f).name
        level = int(name[3])
        pol = name[5:name.rindex("_task")]
        add(f, level, pol)
    return s


def contrast(s, terms, index):
    keys = [set(s.get((p, l), {})) for _, p, l in terms]
    common = set.intersection(*keys) if keys else set()
    per_task = defaultdict(list)
    for t, e in sorted(common):
        per_task[t].append(sum(c * s[(p, l)][(t, e)] for c, p, l in terms))
    if not per_task:
        return None
    arrs = [np.array(v, float) for _, v in sorted(per_task.items())]
    est = 100 * np.mean([a.mean() for a in arrs])
    rng = np.random.default_rng([SEED, index])
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(arrs), len(arrs))
        boot[i] = 100 * np.mean([arrs[j][rng.integers(0, len(arrs[j]), len(arrs[j]))].mean() for j in pick])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"estimate_pp": float(est), "ci95_pp": [float(lo), float(hi)], "tasks": len(arrs),
            "episodes": int(sum(len(a) for a in arrs)),
            "reading": "positive" if lo > 0 else "negative" if hi < 0 else "not distinguishable",
            "per_task_pp": {str(t): float(100 * np.mean(v)) for t, v in sorted(per_task.items())}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "analysis" / "cl_perturb.json")
    args = ap.parse_args()
    s = load()
    success = {}
    for p in POLICIES:
        for lev in (0, 1, 2):
            d = s.get((p, lev), {})
            if d:
                per = {t: 100 * np.mean([v for (tt, _), v in d.items() if tt == t]) for t in TASKS
                       if any(tt == t for tt, _ in d)}
                success[f"{p}@L{lev}"] = {"mean_over_tasks": float(np.mean(list(per.values()))), "episodes": len(d),
                                          "per_task": {str(t): float(v) for t, v in per.items()}}
    did = lambda a, b, lev: [(1, a, lev), (-1, b, lev), (-1, a, 0), (1, b, 0)]
    C = {
        "P1 DiD k16 L2: (replay16-msal16)@L2 - @L0": did("ret_replay_k16", "msal_k16", 2),
        "P2 DiD k1 L2: (replay1-msal1)@L2 - @L0": did("ret_replay_k1", "msal_k1", 2),
        "S1 DiD k16 L1": did("ret_replay_k16", "msal_k16", 1),
        "S2 DiD k1 L1": did("ret_replay_k1", "msal_k1", 1),
        "S3 replay16-msal16 @L2": [(1, "ret_replay_k16", 2), (-1, "msal_k16", 2)],
        "S4 replay1-msal1 @L2": [(1, "ret_replay_k1", 2), (-1, "msal_k1", 2)],
        "S5a bf16-replay16 @L2": [(1, "bf_k16", 2), (-1, "ret_replay_k16", 2)],
        "S5b DiD (bf16-replay16) L2 vs L0": did("bf_k16", "ret_replay_k16", 2),
    }
    for p in POLICIES:
        C[f"S6 {p} L2-L0"] = [(1, p, 2), (-1, p, 0)]
        C[f"S6 {p} L1-L0"] = [(1, p, 1), (-1, p, 0)]
    res = {name: contrast(s, terms, i) for i, (name, terms) in enumerate(C.items())}
    out = {"success": success, "contrasts": res}
    args.out.write_text(json.dumps(out, indent=1))
    for k, v in success.items():
        print(f"{k:22s} {v['mean_over_tasks']:6.1f}  ({v['episodes']} eps)")
    for k, v in res.items():
        if v:
            print(f"{k:45s} {v['estimate_pp']:+6.1f} [{v['ci95_pp'][0]:+6.1f}, {v['ci95_pp'][1]:+6.1f}] {v['reading']} ({v['episodes']} eps)")


if __name__ == "__main__":
    main()
