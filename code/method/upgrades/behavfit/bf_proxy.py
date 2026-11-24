from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bf_core import (HELD, RUNS, SPLIT_TRAIN, SPLIT_VAL, FHead, action_error, action_fit, action_fit_lm, demo_rows, flow_blocks,
                     flow_mse, invert_and_code, load_rows, noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--shots", default="1,16")
    ap.add_argument("--settings", default="lm:2,5,10;adam:1e-1:1e-2:50,150,400",
                    help="';'-separated: lm:CHECKPOINTS or adam:LR_D:LR_Z:CHECKPOINTS (checkpoints ','-separated)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    tasks = [int(t) for t in args.tasks.split(",")]
    assert not set(tasks) & set(HELD), "proxy tasks must not be held tasks"
    settings = [s.split(":") for s in args.settings.split(";")]
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["fold"], r["task"], r["k"]))
    data = {t: (load_rows(t, split=SPLIT_TRAIN, first_episodes=16), load_rows(t, split=SPLIT_VAL)) for t in tasks}
    for h in (int(f) for f in args.folds.split(",")):
        art = torch.load(RUNS / f"held{h}.pt", weights_only=False)
        head = FHead(art["head"], art["library"], dev)
        d0 = torch.zeros(1, art["d_star"].shape[1], device=dev)
        d0[0, -1] = 1.0
        for t in tasks:
            r16, val = data[t]
            vsh = head.encode(val["state"])
            vA, vM = val["actions"].to(dev), val["masks"].to(dev)
            veps = noise(vsh.shape[0], 4, 424242 + t, dev)
            vblk = flow_blocks(vsh.shape[0], 4, 525252 + t, dev)
            score = lambda d, z: {"act": action_error(head, vsh, vA, vM, d, z, veps),
                                  "flow": flow_mse(head, vsh, vA, vM, d, z, vblk)}
            for k in (int(s) for s in args.shots.split(",")):
                if (h, t, k) in done:
                    continue
                t0 = time.time()
                r = demo_rows(t, k, r16)
                sh = head.encode(r["state"])
                A, M = r["actions"].to(dev), r["masks"].to(dev)
                n = sh.shape[0]
                rec = {"fold": h, "task": t, "k": k, "rows": n, "scores": {}}
                rec["scores"]["zero"] = score(d0, None)
                d_m, z_m = invert_and_code(head, sh, A, M, d0, flow_blocks(n, 4, 60_000 + t, dev))
                rec["scores"]["msal"] = score(d_m, z_m)
                d_a, z_a = invert_and_code(head, sh, A, M, d0, flow_blocks(n, 4, 60_000 + t, dev, (0.001, 0.3)))
                rec["scores"]["low"] = score(d_a, z_a)
                eps = noise(n, 8 if k == 1 else 2, 70_000 + t, dev)
                for st in settings:
                    cks = tuple(int(c) for c in st[-1].split(","))
                    if st[0] == "lm":
                        fits = action_fit_lm(head, sh, A, M, d_m, z_m.float(), eps, iters=max(cks), checkpoints=cks)
                        tag = "bf_lm"
                    else:
                        lr_d, lr_z = float(st[1]), float(st[2])
                        fits = action_fit(head, sh, A, M, d_m, z_m.float(), eps, steps=max(cks), lr_d=lr_d,
                                          lr_z=lr_z, checkpoints=cks)
                        tag = f"bf_adam{lr_d:g}"
                    for s, (d, z) in fits.items():
                        rec["scores"][f"{tag}_{s}"] = {**score(d, z), "d_shift": float((d - d_m).norm())}
                rec["seconds"] = time.time() - t0
                with args.out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(json.dumps({"fold": h, "task": t, "k": k, "sec": round(rec["seconds"]),
                                  **{m: round(v["act"], 4) for m, v in rec["scores"].items()}}), flush=True)
    print("proxy done", flush=True)


if __name__ == "__main__":
    main()
