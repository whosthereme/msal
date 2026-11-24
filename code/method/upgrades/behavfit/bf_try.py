import sys
import time

import torch

from bf_core import RUNS, FHead, action_error, action_fit, action_fit_lm, demo_rows, noise

h, k, dev = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
art = torch.load(RUNS / f"held{h}.pt", weights_only=False)
head = FHead(art["head"], art["library"], dev)
r = demo_rows(h, k)
sh = head.encode(r["state"])
A, M = r["actions"].to(dev), r["masks"].to(dev)
eps = noise(sh.shape[0], 8 if k == 1 else 2, 7, dev)
other = noise(sh.shape[0], 4, 8, dev)
a = art["adapt"][k]
d0, z0 = a["d"].to(dev), a["z"].to(dev)
print("msal fresh-noise demo err", action_error(head, sh, A, M, d0, z0, other), flush=True)
for spec in sys.argv[4:]:
    t0 = time.time()
    if spec.startswith("lm"):
        out = action_fit_lm(head, sh, A, M, d0, z0, eps, iters=int(spec[2:]), checkpoints=(2, 5, 10, 20), log=print)
    else:
        lr_d, lr_z, steps = spec.split(":")
        out = action_fit(head, sh, A, M, d0, z0.float(), eps, steps=int(steps), lr_d=float(lr_d), lr_z=float(lr_z),
                         checkpoints=(50, 150, 400), log=print)
    print(spec, f"{time.time() - t0:.0f}s", flush=True)
    for s, (d, z) in out.items():
        print("  ", s, "fresh-noise demo err", round(action_error(head, sh, A, M, d, z, other), 5), "|z|",
              round(float(z.norm()), 4), "|d-dmsal|", round(float((d.cpu() - a["d"]).norm()), 2), flush=True)
