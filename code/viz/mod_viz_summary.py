import sys
import numpy as np

z = np.load(sys.argv[1])
dirs, ring = z["direction"], z["ring"]
SEEDS = (148021, 148022, 148023)


def neighbours(i):
  step = 2 if ring[i] == 0 else 1
  return [j for j in range(len(dirs)) if ring[j] == ring[i] and (dirs[j] - dirs[i]) % 16 in (step, 16 - step)]


def cos(a, b):
  return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def stats(arm):
  held_sim, ref_sim, ratio, rel_out = [], [], [], []
  for f in range(4):
    for s in SEEDS:
      t = "%s_f%d_s%d" % (arm, f, s)
      m, held = z[t + "_mod"].astype(np.float64), z[t + "_held"]
      for i in range(len(dirs)):
        nb = neighbours(i)
        sim = cos(m[i], m[nb].mean(0))
        if held[i]:
          assert not held[nb].any()
          held_sim.append(sim)
          ratio.append(z[t + "_out"][i] / z[t + "_in"][i])
          rel_out.append(z[t + "_out"][i] / np.median(np.linalg.norm(m[~held], axis=1)))
        elif not held[nb].any():
          ref_sim.append(sim)
  return np.array(held_sim), np.array(ref_sim), np.array(ratio), np.array(rel_out)


for arm in ("msal_coord", "msal_harm", "msalh_coord", "msalh_harm"):
  h, r, q, o = stats(arm)
  print("%-12s held cos to neighbours: median %.3f [%.3f, %.3f] n=%d | trained-to-trained median %.3f n=%d | "
    "out/in median %.2f | out-of-span norm / median trained modulation %.2f" % (arm, np.median(h), h.min(), h.max(),
    len(h), np.median(r) if len(r) else float("nan"), len(r), np.median(q), np.median(o)))
