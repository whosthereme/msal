import argparse
import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), str(Path(__file__).resolve().parents[1] / "local_run")]
import tc_combo
import numpy as np
from lib.pool_v5 import build_pool, source_task_ids
import tc_lodo

STOP = {"the", "a", "an", "and", "it", "of", "to"}
ALPHAS = (0.5, 1.0, 2.0)


def iota(D_fit, d):
  P = np.linalg.pinv(D_fit) @ D_fit
  p = P @ d
  return float(p @ p / (d @ d))


def onehot_dir(ids, direction_of):
  out = np.zeros((len(ids), 16))
  for i, t in enumerate(ids):
    out[i, direction_of[int(t)]] = 1.0
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("libero", nargs="+")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  arms = tc_lodo.register()
  ids = [int(t) for t in source_task_ids()]
  direction_of = dict((t.task_id, t.direction_16) for t in build_pool())
  descs = dict(harmonic=arms.DESCRIPTORS["lodo_sat"], xy6=arms.DESCRIPTORS["lodo_xy6"])
  for a in ALPHAS:
    descs["xy6_id%g" % a] = (lambda a: lambda q: np.concatenate([arms.DESCRIPTORS["lodo_xy6"](q), a * onehot_dir(q, direction_of)], axis=1))(a)
  dev = {}
  for name, fn in descs.items():
    per_task = {}
    for t in ids:
      fit = [u for u in ids if direction_of[u] != direction_of[t]]
      per_task[t] = iota(np.asarray(fn(fit), float), np.asarray(fn([t]), float)[0])
    dev[name] = dict(per_task=per_task, mean=float(np.mean(list(per_task.values()))),
      min=float(min(per_task.values())), max=float(max(per_task.values())))
    print("dev LODO %-10s iota mean %.3f  min %.3f  max %.3f" % (name, dev[name]["mean"], dev[name]["min"], dev[name]["max"]))
  instr = {}
  for p in args.libero:
    for line in Path(p).read_text().splitlines():
      if line.strip():
        r = json.loads(line)
        instr[int(r["held_task"])] = r["instruction"]
  vocab = sorted({w for s in instr.values() for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP})
  index = {w: i for i, w in enumerate(vocab)}

  def wd(t):
    row = np.zeros(len(vocab) + 1)
    row[-1] = 1.0
    for w in set(re.findall(r"[a-z]+", instr[t].lower())):
      if w in index:
        row[index[w]] = 1.0
    return row

  lib = {}
  tasks = sorted(instr)
  for t in tasks:
    fit = np.stack([wd(u) for u in tasks if u != t])
    seen = {w for u in tasks if u != t for w in re.findall(r"[a-z]+", instr[u].lower())}
    unseen = sorted({w for w in re.findall(r"[a-z]+", instr[t].lower()) if w not in STOP and w not in seen})
    lib[t] = dict(iota=iota(fit, wd(t)), unseen_words=unseen)
  vals = np.array([v["iota"] for v in lib.values()])
  print("LIBERO tasks %d, vocab %d; iota mean %.3f, min %.3f; iota = 1 on %d tasks; tasks with an unseen word %d"
    % (len(tasks), len(vocab), vals.mean(), vals.min(), int((vals > 1 - 1e-9).sum()), sum(1 for v in lib.values() if v["unseen_words"])))
  print("LIBERO iota < 1 but no unseen word: %d" % sum(1 for v in lib.values() if v["iota"] < 1 - 1e-9 and not v["unseen_words"]))
  Path(args.out).write_text(json.dumps(dict(dev=dev, libero=lib, vocab=vocab), indent=1))
  print("written", args.out)


if __name__ == "__main__":
  main()
