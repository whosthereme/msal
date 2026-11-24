from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/home/ubuntu/work/manuscript6/upgrades/behavfit")
from bf_core import (HELD, RUNS, SPLIT_TRAIN, SPLIT_VAL, FHead, action_error, action_fit_lm,
                     demo_rows, flow_blocks, invert_and_code, load_rows, noise)

BF = Path("/home/ubuntu/work/manuscript6/upgrades/behavfit/art")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tasks", default=",".join(map(str, HELD)))
    args = ap.parse_args()
    dev = args.device
    seen = torch.load(RUNS / "seen.pt", weights_only=False)
    with args.out.open("a") as fh:
        for h in [int(x) for x in args.tasks.split(",")]:
            art = torch.load(RUNS / f"held{h}.pt", weights_only=False)
            val = load_rows(h, split=SPLIT_VAL)
            A, M = val["actions"].to(dev), val["masks"].to(dev)
            eps = noise(A.shape[0], 8, 31337 + h, dev)
            base = FHead(art["head"], art["library"], dev)
            sh_val = base.encode(val["state"])
            rec = {"task": h, "val_rows": int(A.shape[0])}

            def score(head, d, z, sh=None):
                sh = head.encode(val["state"]) if sh is None else sh
                return action_error(head, sh, A, M, d.to(dev), None if z is None else z.to(dev), eps)

            rec["zero"] = score(base, art["d_star"], None, sh_val)
            for k in (1, 16):
                rec[f"msal_k{k}"] = score(base, art["adapt"][k]["d"], art["adapt"][k]["z"], sh_val)
                ft = FHead(art["finetune"][k]["head"], None, dev)
                rec[f"ft_k{k}"] = score(ft, art["d_star"], None)
                dm = demo_rows(h, k)
                Ad, Md = dm["actions"].to(dev), dm["masks"].to(dev)
                epd = noise(Ad.shape[0], 4, 61337 + h + k, dev)
                rec[f"msal_k{k}_train"] = action_error(base, base.encode(dm["state"]), Ad, Md,
                                                       art["adapt"][k]["d"].to(dev), art["adapt"][k]["z"].to(dev), epd)
                rec[f"ft_k{k}_train"] = action_error(ft, ft.encode(dm["state"]), Ad, Md, art["d_star"].to(dev), None, epd)
            if (BF / f"held{h}.pt").exists():
                bf = torch.load(BF / f"held{h}.pt", weights_only=False)
                rec["bf_k16"] = score(base, bf["adapt"][16]["d"], bf["adapt"][16]["z"], sh_val)
            sd = FHead(seen["head"], None, dev)
            rec["seen"] = score(sd, seen["descriptors"][h], None)
            tr = load_rows(h, split=SPLIT_TRAIN)
            At, Mt = tr["actions"].to(dev), tr["masks"].to(dev)
            sh_tr = base.encode(tr["state"])
            d, z = invert_and_code(base, sh_tr, At, Mt, art["d_star"].to(dev), flow_blocks(At.shape[0], 4, 71337 + h, dev))
            rec["msal_alltrain"] = score(base, d, z, sh_val)
            rec["alltrain_rows"] = int(At.shape[0])
            d, z = invert_and_code(base, sh_val, A, M, art["d_star"].to(dev), flow_blocks(A.shape[0], 8, 81337 + h, dev))
            rec["ceil_flow"] = score(base, d, z, sh_val)
            fit = action_fit_lm(base, sh_val, A, M, art["adapt"][16]["d"].to(dev), art["adapt"][16]["z"].to(dev).float(),
                                eps, iters=10)
            d, z = fit[10]
            rec["ceil_act"] = score(base, d, z, sh_val)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(json.dumps(rec), flush=True)
    print("capacity done", flush=True)


if __name__ == "__main__":
    main()
