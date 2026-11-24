import json
import sys
from collections import defaultdict

import numpy as np

recs = [json.loads(x) for x in open(sys.argv[1]) if x.strip()]
by = defaultdict(lambda: defaultdict(list))
for r in recs:
    for m, s in r["scores"].items():
        by[m]["all"].append(s["act"])
        by[m][f"k{r['k']}"].append(s["act"])
        by[m]["flow"].append(s["flow"])
        by[m]["rel_msal"].append(s["act"] / r["scores"]["msal"]["act"] - 1)
print(f"cells: {len(recs)}")
print(f"{'method':18s} {'act_all':>8s} {'act_k1':>8s} {'act_k16':>8s} {'flow':>8s} {'vs msal':>8s} {'wins':>5s}")
for m, v in by.items():
    wins = sum(x < 0 for x in v["rel_msal"])
    print(f"{m:18s} {np.mean(v['all']):8.4f} {np.mean(v['k1']):8.4f} {np.mean(v['k16']):8.4f} {np.mean(v['flow']):8.4f} "
          f"{100 * np.mean(v['rel_msal']):+7.1f}% {wins:3d}/{len(v['all'])}")
bf = {m: np.mean(v["all"]) for m, v in by.items() if m.startswith("bf_")}
best = min(bf, key=bf.get)
print("rule: best (b) setting", best, f"{bf[best]:.4f}; (a) low {np.mean(by['low']['all']):.4f};",
      "run", best if bf[best] <= np.mean(by["low"]["all"]) else "low")
