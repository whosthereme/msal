from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ret_rollout import ACTION, CHUNK, Head

HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)


def pad(acts, masks):
    acts = acts.copy()
    for i in range(len(acts)):
        valid = np.flatnonzero(masks[i] > 0.5)
        last = valid[-1] if len(valid) else 0
        acts[i, last + 1:] = acts[i, last]
    return acts


def sample(head, s, d, z, eps, a0=None, tau0=1.0, steps=10):
    i0 = int(round((1.0 - tau0) * steps))
    t0 = 1.0 - i0 / steps
    x = eps if a0 is None else t0 * eps + (1 - t0) * a0
    with torch.no_grad():
        for i in range(i0, steps):
            t = torch.full((s.shape[0],), 1.0 - i / steps)
            x = x - head.velocity(s, x, t, d, z) / steps
    return x.clamp(-1, 1)


def nearest(qs, bs):
    d = (qs * qs).sum(1, keepdim=True) - 2 * qs @ bs.T + (bs * bs).sum(1)[None]
    return d.argmin(1)


def mse(pred, true, m):
    return float((((pred - true) ** 2).mean(-1) * m).sum() / m.sum())


def evaluate(art, feats, acts, masks, q_idx, b_idx, k, chunk=256):
    hz = Head(art["head"])
    hm = Head(art["head"], art["library"])
    hf = Head(art["finetune"][k]["head"])
    dm, zm = art["adapt"][k]["d"], art["adapt"][k]["z"]
    mean, std = hz.sd["state_mean"], hz.sd["state_std"]
    bs = (feats[b_idx] - mean) / std
    bact = torch.from_numpy(pad(acts[b_idx].numpy(), masks[b_idx].numpy()))
    g = torch.Generator().manual_seed(4321)
    out = {key: [] for key in ("retrieved", "zero", "msal", "ret_msal", "ft")}
    w = []
    for start in range(0, len(q_idx), chunk):
        qi = q_idx[start:start + chunk]
        s, a, m = feats[qi], acts[qi], masks[qi]
        a_r = bact[nearest((s - mean) / std, bs)]
        eps = torch.randn(len(qi), CHUNK, ACTION, generator=g)
        preds = {"retrieved": a_r.clamp(-1, 1), "zero": sample(hz, s, art["d_star"], None, eps),
                 "msal": sample(hm, s, dm, zm, eps), "ret_msal": sample(hm, s, dm, zm, eps, a_r, 0.5),
                 "ft": sample(hf, s, art["d_star"], None, eps)}
        for key, p in preds.items():
            out[key].append(mse(p, a, m) * float(m.sum()))
        w.append(float(m.sum()))
    return {key: sum(v) / sum(w) for key, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--banks", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    res = {}
    for h in HELD:
        art = torch.load(args.runs / f"held{h}.pt", weights_only=False, map_location="cpu")
        b = np.load(args.banks / f"bank_h{h}_k16.npz")
        feats = torch.from_numpy(b["features"]).float()
        acts, masks, eps_ids = torch.from_numpy(b["actions"]), torch.from_numpy(b["masks"]), b["episodes"]
        demos = np.unique(eps_ids)
        k1 = {"bank": np.flatnonzero(eps_ids == demos[0]), "query": np.flatnonzero(eps_ids != demos[0])}
        r1 = evaluate(art, feats, acts, masks, k1["query"], k1["bank"], 1)
        loo = []
        for j in demos:
            r = evaluate(art, feats, acts, masks, np.flatnonzero(eps_ids == j), np.flatnonzero(eps_ids != j), 16)
            loo.append((r, int((masks[np.flatnonzero(eps_ids == j)]).sum())))
        r16 = {key: sum(r[key] * n for r, n in loo) / sum(n for _, n in loo) for key in loo[0][0]}
        res[h] = {"k1": r1, "k16_loo": r16}
        print(h, {k: {m: round(v, 4) for m, v in r.items()} for k, r in res[h].items()}, flush=True)
    res["mean"] = {k: {m: float(np.mean([res[h][k][m] for h in HELD])) for m in res[HELD[0]][k]}
                   for k in ("k1", "k16_loo")}
    print("mean", res["mean"], flush=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
