import json
import sys

import numpy as np


def load(npz, key="last"):
  z = np.load(npz)
  ids = [int(i) for i in z["task_ids"]]
  u = z[key].astype(np.float64)
  u = u / np.linalg.norm(u, axis=1, keepdims=True)
  return ids, u


def fold_matrix(ids, u, held):
  tr = [i for i, t in enumerate(ids) if t != held]
  mu = u[tr].mean(0)
  c = u - mu
  s = np.sqrt((np.linalg.norm(c[tr], axis=1) ** 2).mean())
  return np.concatenate([np.ones((len(ids), 1)), 3.0 * c / s], axis=1)


def descriptors(ids, u, held):
  tr = [i for i, t in enumerate(ids) if t != held]
  mu = u[tr].mean(0)
  c = u - mu
  s = np.sqrt((np.linalg.norm(c[tr], axis=1) ** 2).mean())
  d = np.concatenate([np.ones((len(ids), 1)), 3.0 * c / s], axis=1)
  return d[tr], d[ids.index(held)]


def share(train, row):
  q, _ = np.linalg.qr(train.T)
  rank = int(np.linalg.matrix_rank(train))
  p = q[:, :rank] @ (q[:, :rank].T @ row)
  return float(p @ p / (row @ row)), rank, p


if __name__ == "__main__":
  ids, u = load(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "last")
  out = {}
  for h in ids:
    tr, row = descriptors(ids, u, h)
    i, rank, _ = share(tr, row)
    out[h] = dict(iota=i, rank=rank, norm=float(np.linalg.norm(row)))
  v = np.array([o["iota"] for o in out.values()])
  print("share: mean %.3f min %.3f max %.3f; rank %s; held norm mean %.2f" % (v.mean(), v.min(), v.max(),
    sorted(set(o["rank"] for o in out.values())), np.mean([o["norm"] for o in out.values()])))
  if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
