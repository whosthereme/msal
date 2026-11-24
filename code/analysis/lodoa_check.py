import json
import sys
from pathlib import Path

new = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
stored = dict()
for p in Path(sys.argv[2]).glob("lodo_*_seed*.jsonl"):
  for x in p.read_text().splitlines():
    if x.strip():
      r = json.loads(x)
      stored[(r["seed"], r["arm"], r["held_task"])] = r
for r in new:
  s = stored.get((r["seed"], r["arm"], r["held_task"]))
  if s is None:
    print(r["seed"], r["arm"], r["held_task"], "no stored record")
    continue
  same = dict(flow_selection=r["score_forward"]["selection"] == s["flow_selection"],
    flow_robustness=r["score_forward"]["robustness"] == s["flow_robustness"],
    action=r["action"]["zero_shot"] == s["action_selection"])
  print(r["seed"], r["arm"], "direction", r["held_direction"], "task", r["held_task"],
    "ALL IDENTICAL" if all(same.values()) else same)
