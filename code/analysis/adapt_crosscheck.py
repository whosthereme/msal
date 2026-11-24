import json
import math
import sys

import numpy as np

stored = dict((int(r["held_task"]), r) for r in map(json.loads, open(sys.argv[1])))
rec = dict()
for path in sys.argv[2:]:
  for line in open(path):
    r = json.loads(line)
    rec[(r["arm"], int(r["held_task"]))] = r
GETS = (("zero-shot", lambda r: r["zero_shot"]["selection"]), ("inversion k=16", lambda r: r["inversion"]["16"]["selection"]),
  ("MSAL k=1", lambda r: r["combination"]["1"]["selection"]), ("MSAL k=16", lambda r: r["combination"]["16"]["selection"]))
for name, get in GETS:
  for group in ("local", "nodes"):
    tasks = [t for t in stored if (rec[("film_nl", t)].get("lane") == "local") == (group == "local")]
    v = np.array([math.log(get(rec[("film_nl", t)]) / get(stored[t])) for t in tasks])
    print("%-15s %-5s n=%2d mean %+5.1f%% sd %4.1f%% identical %d" % (name, group, len(v), 100 * (math.exp(v.mean()) - 1),
      100 * v.std(ddof=1), int((v == 0).sum())))
for k in ("1", "16"):
  v = np.array([math.log(stored[t]["combination"][k]["selection"] / stored[t]["zero_shot"]["selection"]) for t in stored])
  print("stored ws2, this seed: MSAL k=%s vs zero-shot %+.1f%%" % (k, 100 * (math.exp(v.mean()) - 1)))
