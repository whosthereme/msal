import json
import sys

new = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
stored = dict()
for p in sys.argv[2:]:
  for x in open(p):
    if x.strip():
      r = json.loads(x)
      stored[(r["seed"], r["arm"], r["held_task"])] = r
for r in new:
  s = stored.get((r["seed"], r["arm"], r["held_task"]))
  if s is None:
    print(r["seed"], r["arm"], r["held_task"], "no stored record")
    continue
  raw = r["raw"]
  same = dict(zero_shot=raw["zero_shot"] == s["zero_shot"], action=raw["action"] == s["action"],
    inversion=raw["inversion"] == s["inversion"], combination=raw["combination"] == s["combination"],
    stage2=r["stage2_final_loss"] == s["stage2_final_loss"], score_forward=r["score_forward"] == s["score_forward"])
  print(r["seed"], r["arm"], r["held_task"], "lane", r["lane"], "vs", s["lane"], "ALL IDENTICAL" if all(same.values()) else same)
