from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LEROBOT_LIBERO = Path("/home/ubuntu/work/pilot/data/libero")
SUBSTRATE = Path(__file__).resolve().parents[1] / "runs" / "20260829_g39_libero40_substrate_v1"
FIRST_TASK_ID = 1000

LIBERO_10 = {
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
}


def suite_of(instruction: str) -> str:
    if instruction in LIBERO_10:
        return "libero_10"
    if instruction.startswith("pick up the black bowl"):
        return "libero_spatial"
    if instruction.startswith("pick up the") and instruction.endswith("and place it in the basket"):
        return "libero_object"
    return "libero_goal"


def lerobot_tasks() -> tuple[list[str], Counter]:
    import pandas as pd

    tasks = pd.read_parquet(LEROBOT_LIBERO / "meta" / "tasks.parquet")
    by_index = {int(i): str(t) for t, i in zip(tasks.index, tasks["task_index"])} if "task_index" in tasks else None
    episodes = pd.concat([pd.read_parquet(p) for p in sorted((LEROBOT_LIBERO / "meta" / "episodes").rglob("*.parquet"))])
    names = [t[0] if not isinstance(t, str) else t for t in episodes["tasks"]]
    return sorted(set(by_index.values()) if by_index else set(names)), Counter(names)


def mapping() -> dict[int, dict]:
    instructions, _ = lerobot_tasks()
    per_suite: dict[str, list[str]] = {}
    for text in instructions:
        per_suite.setdefault(suite_of(text), []).append(text)
    if sorted(len(v) for v in per_suite.values()) != [10, 10, 10, 10]:
        raise RuntimeError(f"suite sizes {({k: len(v) for k, v in per_suite.items()})} are not 4 x 10")
    out, task_id = {}, FIRST_TASK_ID
    for suite in sorted(per_suite):
        for text in sorted(per_suite[suite]):
            out[task_id] = {"suite": suite, "instruction": text}
            task_id += 1
    return out


def verify_mapping(tasks: dict[int, dict]) -> dict:
    import numpy as np

    _, lerobot_counts = lerobot_tasks()
    rows = []
    for task_id, entry in tasks.items():
        shard = np.load(SUBSTRATE / "shards" / f"task{task_id}.npz", mmap_mode="r")
        substrate_episodes = len(np.unique(shard["episode_indices"]))
        rows.append((task_id, substrate_episodes, lerobot_counts[entry["instruction"]]))
    agree = sum(a == b for _, a, b in rows)
    return {"tasks": len(rows), "episode_counts_agree": agree, "rows": rows,
            "substrate_total": sum(a for _, a, _ in rows), "lerobot_total": sum(b for _, _, b in rows)}


if __name__ == "__main__":
    tasks = mapping()
    check = verify_mapping(tasks)
    print(f"episode counts agree for {check['episode_counts_agree']} of {check['tasks']} tasks "
          f"(totals {check['substrate_total']} vs {check['lerobot_total']})")
    for task_id, a, b in check["rows"]:
        if a != b:
            print("  mismatch", task_id, tasks[task_id]["instruction"], a, b)
    if check["episode_counts_agree"] == check["tasks"]:
        (SUBSTRATE / "libero40_tasks.json").write_text(json.dumps({"tasks": {str(k): v for k, v in tasks.items()},
                                                                   "verification": check}, indent=1))
        print("written", SUBSTRATE / "libero40_tasks.json")
