from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bf_core import HELD, RUNS, SPLIT_VAL, FHead, flow_blocks, flow_mse, load_rows, masked_mse, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bf-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    with args.out.open("w") as fh:
        for h in HELD:
            art = torch.load(RUNS / f"held{h}.pt", weights_only=False)
            val = load_rows(h, split=SPLIT_VAL)
            A, M = val["actions"].to(dev), val["masks"].to(dev)
            n = A.shape[0]
            eps = noise(n, 8, 31337 + h, dev)
            blk = flow_blocks(n, 8, 41337 + h, dev)
            blk_low = flow_blocks(n, 8, 51337 + h, dev, (0.001, 0.3))
            base = FHead(art["head"], art["library"], dev)
            pols = {"zero": (base, art["d_star"], None)}
            for k in (1, 16):
                pols[f"msal_k{k}"] = (base, art["adapt"][k]["d"], art["adapt"][k]["z"])
                pols[f"ft_k{k}"] = (FHead(art["finetune"][k]["head"], None, dev), art["d_star"], None)
            if args.bf_dir is not None and (args.bf_dir / f"held{h}.pt").exists():
                bf = torch.load(args.bf_dir / f"held{h}.pt", weights_only=False)
                for k in (1, 16):
                    pols[f"bf_k{k}"] = (base, bf["adapt"][k]["d"], bf["adapt"][k]["z"])
            for name, (head, d, z) in pols.items():
                d = d.to(dev)
                z = None if z is None else z.to(dev)
                sh = head.encode(val["state"])
                acts, acts8 = [], []
                with torch.no_grad():
                    for j in range(eps.shape[0]):
                        x = head.sample(sh, eps[j], d, z).clamp(-1, 1)
                        acts.append(float(masked_mse(x, A, M)))
                        acts8.append(float(masked_mse(x[:, :8], A[:, :8], M[:, :8])))
                rec = {"task": h, "policy": name, "rows": n, "act": sum(acts) / len(acts),
                       "act_first8": sum(acts8) / len(acts8),
                       "flow": flow_mse(head, sh, A, M, d, z, blk), "flow_low": flow_mse(head, sh, A, M, d, z, blk_low)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(json.dumps(rec), flush=True)
    print("diag done", flush=True)


if __name__ == "__main__":
    main()
