import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def boot_spearman(x, y, n=20000, seed=0):
  x, y = np.asarray(x, float), np.asarray(y, float)
  rho = float(stats.spearmanr(x, y)[0])
  rng = np.random.default_rng(seed)
  idx = rng.integers(0, len(x), size=(n, len(x)))
  vals = np.array([stats.spearmanr(x[i], y[i])[0] for i in idx])
  vals = vals[np.isfinite(vals)]
  lo, hi = np.percentile(vals, [2.5, 97.5])
  word = "supported" if hi < 0 else ("contradicted" if lo > 0 else "not distinguishable")
  return dict(rho=rho, lo=float(lo), hi=float(hi), n=int(len(x)), reading=word)


def show(name, r):
  print("  %-58s rho %+.3f [%+.3f, %+.3f]  n=%d  -> %s" % (name, r["rho"], r["lo"], r["hi"], r["n"], r["reading"]))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("iota")
  ap.add_argument("libero_root")
  ap.add_argument("lodo", nargs="+")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  io = json.loads(Path(a.iota).read_text())
  rep = {}
  recs = [json.loads(x) for p in sorted(Path(a.libero_root).glob("*/folds.jsonl")) for x in p.read_text().splitlines() if x.strip()]
  keys = set((r["seed"], r["held_task"]) for r in recs)
  assert len(keys) == len(recs) == 200, (len(keys), len(recs))
  out = {}
  for split in ("selection", "robustness"):
    per = {}
    for r in recs:
      z = r["zero_shot_float32"]
      per.setdefault(int(r["held_task"]), []).append(math.log(z["film_nl"][split] / z["plain_nl"][split]))
    assert all(len(v) == 5 for v in per.values()) and len(per) == 40
    tasks = sorted(per)
    x = [io["libero"][str(t)]["iota"] for t in tasks]
    y = [float(np.mean(per[t])) for t in tasks]
    out[split] = boot_spearman(x, y)
    show("A1 LIBERO iota vs log(film_nl/plain_nl), %s" % split, out[split])
    if split == "selection":
      seen = [i for i, t in enumerate(tasks) if not io["libero"][str(t)]["unseen_words"]]
      out["selection_no_unseen_word"] = boot_spearman([x[i] for i in seen], [y[i] for i in seen])
      show("A1 LIBERO, tasks without an unseen word", out["selection_no_unseen_word"])
      out["per_task"] = dict((t, dict(iota=x[i], mean_log_ratio=y[i], unseen=bool(io["libero"][str(t)]["unseen_words"]))) for i, t in enumerate(tasks))
  rep["A1"] = out
  rows = {}
  for d in a.lodo:
    for p in Path(d).glob("lodo_*_seed*.jsonl"):
      for x in p.read_text().splitlines():
        if x.strip():
          r = json.loads(x)
          if r["arm"] in ("lodo_sat", "lodo_plain") and r.get("train_steps") == 12800:
            k = (r["arm"], int(r["seed"]), int(r["held_task"]))
            assert k not in rows, k
            rows[k] = r
  seeds = sorted(set(k[1] for k in rows))
  tasks = sorted(set(k[2] for k in rows))
  assert seeds == [148021, 148022, 148023, 148024, 148025] and len(tasks) == 41, (seeds, len(tasks))
  out = {}
  for field in ("action_selection", "flow_selection", "flow_robustness"):
    y = [float(np.mean([math.log(rows[("lodo_sat", s, t)][field] / rows[("lodo_plain", s, t)][field]) for s in seeds])) for t in tasks]
    x = [io["dev"]["harmonic"]["per_task"][str(t)] for t in tasks]
    out[field] = boot_spearman(x, y)
    show("A2 dev LODO harmonic iota vs log(sat/plain), %s" % field, out[field])
    if field == "action_selection":
      out["per_task"] = dict((t, dict(iota=x[i], mean_log_ratio=y[i])) for i, t in enumerate(tasks))
  rep["A2"] = out
  Path(a.out).write_text(json.dumps(rep, indent=1))
  print("written", a.out)


if __name__ == "__main__":
  main()
