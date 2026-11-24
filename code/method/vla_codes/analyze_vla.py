import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RR = ROOT / "results_remote"
EV = RR / "vla_codes" / "eval"
sys.path.insert(0, str(ROOT / "analysis"))
from cl_adapt import compare, read

TASKS = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)
t = {}
crashes = {}
for p in ("zero", "code_k1", "ft_k1", "code_small_k1"):
    t[p] = {}
    for h in TASKS:
        per = EV / f"{p}_h{h}_per_episode.jsonl"
        info = EV / f"{p}_h{h}" / "eval_info.json"
        if per.exists():
            rows = [json.loads(l) for l in per.read_text().splitlines() if l.strip()]
            t[p][h] = {r["episode"]: int(r["success"]) for r in rows}
            crashes[(p, h)] = sum(r["crash"] for r in rows)
        elif info.exists():
            s = json.loads(info.read_text())["per_task"][0]["metrics"]["successes"]
            t[p][h] = {e: int(v) for e, v in enumerate(s)}
rep = [r for r in read(glob.glob(str(RR / "upgrades" / "retrieval" / "results" / "R_ret_replay_k1_task*.jsonl")))
       if r["episode"] < 50]
t["replay_k1"] = {}
for r in rep:
    t["replay_k1"].setdefault(r["task"], {})[r["episode"]] = int(r["success"])
print("episodes per cell:", {p: {h: len(v) for h, v in d.items()} for p, d in t.items() if p != "replay_k1"})
print("crashes:", crashes)
print("\nsuccess (%), mean over tasks, then per task", TASKS)
for p, d in t.items():
    if d:
        per = [100 * np.mean(list(d[h].values())) for h in TASKS if h in d]
        print(f"{p:10s} {np.mean(per):5.1f}  " + " ".join(f"{x:4.0f}" for x in per))
C = [("code_k1", "zero", "primary"), ("code_k1", "ft_k1", "secondary"), ("ft_k1", "zero", "secondary"),
     ("code_k1", "replay_k1", "descriptive"), ("ft_k1", "replay_k1", "descriptive"), ("zero", "replay_k1", "descriptive"),
     ("code_small_k1", "zero", "post hoc"), ("code_small_k1", "code_k1", "post hoc"), ("code_small_k1", "ft_k1", "post hoc")]
out = {}
print("\npaired comparisons (pp, 95% two-level bootstrap):")
for i, (a, b, role) in enumerate(C):
    if all(h in t[a] and h in t[b] for h in TASKS):
        c = compare(t, a, b, 1100 + i)
        out[f"{a} vs {b}"] = {**c, "role": role}
        print(f"{role:11s} {a:8s} vs {b:10s} {c['diff_pp']:+6.1f} [{c['boot95_pp'][0]:+6.1f}, {c['boot95_pp'][1]:+6.1f}] "
              f"{c['reading']:20s} tasks {c['tasks_a_better']}/{c['tasks_b_better']}")
(RR / "vla_codes" / "analysis.json").write_text(json.dumps({"comparisons": out, "crashes": {f"{k[0]}_{k[1]}": v for k, v in crashes.items()}}, indent=1))
