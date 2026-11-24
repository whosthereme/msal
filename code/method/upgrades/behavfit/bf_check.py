import sys
import time

import torch

from bf_core import RUNS, FHead, action_error, demo_rows, load_rows, noise, SPLIT_TRAIN

h = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
dev = sys.argv[2] if len(sys.argv) > 2 else "cpu"
art = torch.load(RUNS / f"held{h}.pt", weights_only=False)
pr = art["probe"]
head = FHead(art["head"], art["library"], dev)
v = head.velocity_raw(pr["state"], pr["x"], pr["t"], art["d_star"]).cpu()
print("zero probe", float((v - pr["zero_shot_v"]).abs().max()))
for k in (1, 16):
    a = art["adapt"][k]
    v = head.velocity_raw(pr["state"], pr["x"], pr["t"], a["d"], a["z"].to(dev)).cpu()
    print(f"msal_k{k} probe", float((v - a["probe_v"]).abs().max()), "|z|", float(a["z"].norm()),
          "|d-d*|", float((a["d"] - art["d_star"]).norm()))
    fh = FHead(art["finetune"][k]["head"], None, dev)
    v = fh.velocity_raw(pr["state"], pr["x"], pr["t"], art["d_star"]).cpu()
    print(f"ft_k{k} probe", float((v - art["finetune"][k]["probe_v"]).abs().max()))
r16 = load_rows(h, split=SPLIT_TRAIN, first_episodes=16)
for k in (1, 16):
    r = demo_rows(h, k, r16)
    sh = head.encode(r["state"])
    A, M = r["actions"].to(dev), r["masks"].to(dev)
    eps = noise(sh.shape[0], 4, 99, dev)
    a = art["adapt"][k]
    with torch.no_grad():
        hh = head.hidden(sh, eps[0], torch.full((sh.shape[0],), 0.5))
        at = head.atoms(hh)
        base = head.base_from_hidden(hh, a["d"])
    print(f"k={k} rows={sh.shape[0]} base rms {float(base.pow(2).mean().sqrt()):.3f} atom rms per k",
          [round(float(x), 2) for x in at.pow(2).mean(dim=(0, 2, 3)).sqrt()])
    print("   z*atoms rms", float(torch.einsum('k,bktd->btd', a['z'].float(), at).pow(2).mean().sqrt()))
    t0 = time.time()
    print("   action err demo: zero", action_error(head, sh, A, M, art["d_star"], None, eps),
          "msal", action_error(head, sh, A, M, a["d"], a["z"].to(dev), eps),
          "inv only", action_error(head, sh, A, M, a["d"], None, eps), f"{time.time()-t0:.1f}s")
    fh = FHead(art["finetune"][k]["head"], None, dev)
    print("   ft", action_error(fh, fh.encode(r["state"]), A, M, art["d_star"], None, eps))
