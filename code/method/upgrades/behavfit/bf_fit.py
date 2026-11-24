from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bf_core import (HELD, RUNS, SPLIT_TRAIN, FHead, action_error, action_fit, action_fit_lm, demo_rows,
                     flow_blocks, invert_and_code, load_rows, noise, sha256, SHARDS)


def fit_one(head, rows, d_msal, z_msal, d_star, method, k, dev, seed):
    sh = head.encode(rows["state"])
    A, M = rows["actions"].to(dev), rows["masks"].to(dev)
    n = sh.shape[0]
    eps = noise(n, 8 if k == 1 else 2, seed, dev)
    kind, *par = method.split(":")
    t0 = time.time()
    if kind == "lm":
        d, z = action_fit_lm(head, sh, A, M, d_msal, z_msal.float(), eps, iters=int(par[0]))[int(par[0])]
    elif kind == "adam":
        s = int(par[2])
        d, z = action_fit(head, sh, A, M, d_msal, z_msal.float(), eps, steps=s, lr_d=float(par[0]),
                          lr_z=float(par[1]))[s]
    elif kind == "low":
        d, z = invert_and_code(head, sh, A, M, d_star, flow_blocks(n, 4, seed, dev, (0.001, 0.3)))
    else:
        raise ValueError(method)
    sec = time.time() - t0
    other = noise(n, 4, seed + 1, dev)
    info = {"method": method, "rows": n, "fit_seconds": sec,
            "demo_act_fit_draws": {"msal": action_error(head, sh, A, M, d_msal, z_msal, eps),
                                   "bf": action_error(head, sh, A, M, d, z, eps)},
            "demo_act_fresh_draws": {"msal": action_error(head, sh, A, M, d_msal, z_msal, other),
                                     "bf": action_error(head, sh, A, M, d, z, other)}}
    return d, z, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--held", default=",".join(map(str, HELD)))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    for h in (int(x) for x in args.held.split(",")):
        if (args.out / f"held{h}.pt").exists():
            continue
        art = torch.load(RUNS / f"held{h}.pt", weights_only=False)
        head = FHead(art["head"], art["library"], dev)
        cpu = FHead(art["head"], art["library"], "cpu")
        r16 = load_rows(h, split=SPLIT_TRAIN, first_episodes=16)
        new = {k: v for k, v in art.items() if k != "finetune"}
        new["finetune_dropped"] = {k: {"selection": v["selection"], "setting": v["setting"]} for k, v in art["finetune"].items()}
        new["adapt"], new["behavfit"] = {}, {"shard_sha256": sha256(SHARDS / f"task{h}.npz"), "fits": {}}
        for k in (1, 16):
            rows = demo_rows(h, k, r16)
            a = art["adapt"][k]
            d, z, info = fit_one(head, rows, a["d"].to(dev), a["z"].to(dev), art["d_star"].to(dev), args.method, k,
                                 dev, seed=80_000 + 1000 * k + h)
            d, z = d.detach().cpu().float().reshape(1, -1), z.detach().cpu().double()
            pr = art["probe"]
            with torch.no_grad():
                probe_v = cpu.velocity_raw(pr["state"], pr["x"], pr["t"], d, z)
            new["adapt"][k] = {"d": d, "z": z, "selection": None, "probe_v": probe_v, "msal_d": a["d"], "msal_z": a["z"],
                               "msal_selection": a["selection"]}
            info.update(d_shift=float((d - a["d"]).norm()), z_norm=float(z.norm()), msal_z_norm=float(a["z"].norm()))
            new["behavfit"]["fits"][k] = info
            print(json.dumps({"held": h, "k": k, **info}), flush=True)
        torch.save(new, args.out / f"held{h}.pt")
    print("fit done", flush=True)


if __name__ == "__main__":
    main()
