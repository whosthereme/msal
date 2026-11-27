import math
import sys

import numpy as np
import torch

B = "/home/ubuntu/work/manuscript6/trunkcond/libcoord/"
sys.path.insert(0, B)
import lc_tasks

TASKS = {1000 + t["task_id"]: t for t in lc_tasks.tasks()}
IDS = sorted(TASKS)
SEEDS = (148021, 148022, 148023)


def angle_radius(ids):
  t = np.array([TASKS[i]["direction_16"] for i in ids], dtype=np.float64) * 2.0 * math.pi / 16.0
  r = np.array([TASKS[i]["radius_index"] for i in ids], dtype=np.float64) - 1.0
  return t, r


def xy6(ids):
  t, r = angle_radius(ids)
  return np.stack([np.ones_like(t), np.cos(t), np.sin(t), r, r * np.cos(t), r * np.sin(t)], axis=1)


def h20(ids):
  t, r = angle_radius(ids)
  cols = [np.ones_like(t)] + [np.cos(m * t) for m in range(1, 9)] + [np.sin(m * t) for m in range(1, 8)]
  cols += [r, r ** 2, r * np.cos(t), r * np.sin(t)]
  return np.stack(cols, axis=1)


ARMS = {
  "msal_coord": ("heads3/lc_tfilm_xy6_f%d_s%d.pt", xy6, ("trunk_gamma.weight", "trunk_beta.weight")),
  "msal_harm": ("heads4/lc_tfilm_h20_f%d_s%d.pt", h20, ("trunk_gamma.weight", "trunk_beta.weight")),
  "msalh_coord": ("heads/lc_xy6_f%d_s%d.pt", xy6, None),
  "msalh_harm": ("heads2/lc_h20_f%d_s%d.pt", h20, None),
}


def mod_keys(head, c):
  ks = sorted(k for k, v in head.items() if k.endswith(".weight") and ("gamma" in k or "beta" in k)
    and tuple(v.shape) == (512, c))
  g = [k for k in ks if "gamma" in k]
  b = [k for k in ks if "beta" in k]
  assert len(g) == 1 and len(b) == 1, ks
  return g[0], b[0]


def main(out):
  res = dict(ids=np.array(IDS), direction=np.array([TASKS[i]["direction_16"] for i in IDS]),
    ring=np.array([TASKS[i]["radius_index"] for i in IDS]), x=np.array([TASKS[i]["x"] for i in IDS]),
    y=np.array([TASKS[i]["y"] for i in IDS]))
  log = []
  for arm, (pat, fn, keys) in ARMS.items():
    D = fn(IDS)
    c = D.shape[1]
    for f in range(4):
      held = np.array([TASKS[i]["direction_16"] % 4 == f for i in IDS])
      for s in SEEDS:
        o = torch.load(B + pat % (f, s), map_location="cpu", weights_only=False)
        head = o["head"]
        if "descriptors" in o:
          for i, v in o["descriptors"].items():
            assert np.allclose(v.numpy().ravel(), D[IDS.index(int(i))], atol=1e-5), (arm, f, s, i)
        kg, kb = keys if keys else mod_keys(head, c)
        G = head[kg].double().numpy()
        Bt = head[kb].double().numpy()
        Dt = D[~held]
        _, S, Vt = np.linalg.svd(Dt, full_matrices=False)
        r = int((S > S[0] * 1e-9).sum())
        P = Vt[:r].T @ Vt[:r]
        M = np.concatenate([G, Bt], 0)
        Din, Dout = D @ P, D @ (np.eye(c) - P)
        tag = "%s_f%d_s%d" % (arm, f, s)
        res[tag + "_mod"] = (D @ M.T).astype(np.float32)
        res[tag + "_in"] = np.linalg.norm(Din @ M.T, axis=1)
        res[tag + "_out"] = np.linalg.norm(Dout @ M.T, axis=1)
        res[tag + "_iota"] = (Din ** 2).sum(1) / (D ** 2).sum(1)
        res[tag + "_held"] = held
        log.append("%s keys=%s,%s rank=%d held=%d iota_held=%.3f out/in_held=%.3f" % (tag, kg, kb, r, held.sum(),
          res[tag + "_iota"][held].mean(), (res[tag + "_out"][held] / res[tag + "_in"][held]).mean()))
  np.savez_compressed(out, **res)
  print("\n".join(log))


if __name__ == "__main__":
  main(sys.argv[1])
