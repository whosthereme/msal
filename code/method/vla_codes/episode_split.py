import argparse
import glob
import json
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path("/home/ubuntu/work/pilot/data/libero")
TASK_LIST = Path("/home/ubuntu/work/pilot/closed_loop/libero_tasks.json")
L40 = Path("/home/ubuntu/work/manuscript6/repo/sal_v2_identifiable/runs/"
           "20260829_g39_libero40_substrate_v1/libero40_tasks.json")
HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tasks = json.loads(TASK_LIST.read_text())
    l40 = json.loads(L40.read_text())["tasks"]
    rows = []
    for f in sorted(glob.glob(str(DATA / "meta" / "episodes" / "**" / "*.parquet"), recursive=True)):
        t = pq.read_table(f)
        cols = t.column_names
        d = t.to_pydict()
        for i in range(t.num_rows):
            rows.append({"episode_index": int(d["episode_index"][i]),
                         "tasks": list(d["tasks"][i]) if "tasks" in cols else None,
                         "length": int(d["length"][i]) if "length" in cols else None})
    by_task = {}
    for r in rows:
        assert r["tasks"] and len(r["tasks"]) == 1, r
        by_task.setdefault(r["tasks"][0], []).append(r["episode_index"])
    assert len(by_task) == 40 and set(by_task) == set(tasks), "task strings do not match the task list"
    held, held_eps = {}, set()
    for h in HELD:
        instr = l40[str(h)]["instruction"]
        match = [t for t in tasks if t.lower() == instr.lower()]
        assert len(match) == 1, (h, instr)
        eps = sorted(by_task[match[0]])
        held[str(h)] = {"instruction": match[0], "task_index": tasks.index(match[0]), "episodes": len(eps),
                        "demo_episode": eps[0], "suite": l40[str(h)]["suite"]}
        held_eps.update(eps)
    train = sorted(r["episode_index"] for r in rows if r["episode_index"] not in held_eps)
    out = {"held": held, "train_episodes": train, "n_train_episodes": len(train), "n_total_episodes": len(rows),
           "train_tasks": 40 - len(HELD)}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "train_episodes"}, indent=1))


if __name__ == "__main__":
    main()
