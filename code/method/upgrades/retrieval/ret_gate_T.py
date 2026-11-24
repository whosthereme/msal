from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from msal_ext.hyb import held_episode_rows
from lib import g39_data

HELD = (1019, 1025, 1000, 1020, 1023, 1030, 1033, 1037)
SHOTS = (1, 16)
Q = 95.0
RUNS = Path("/home/ubuntu/work/manuscript6/runs/cl_adapt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    data = g39_data.load_g39_data(verify_hash=True)
    res = {"q": Q, "T": {}, "stats": {}}
    for h in HELD:
        sd = torch.load(RUNS / f"held{h}.pt", weights_only=False, map_location="cpu")["head"]
        mean, std = sd["state_mean"].float().to(args.device), sd["state_std"].float().to(args.device)
        pooled = {k: [] for k in SHOTS}
        with torch.no_grad():
            for t in [t for t in data.unique_task_ids() if t != h]:
                val = data.rows_for_task(t, g39_data.SPLIT_VAL)
                q = (torch.from_numpy(data.state_features[val]).to(args.device).float() - mean) / std
                qq = (q * q).sum(1)
                for k in SHOTS:
                    rows = held_episode_rows(data.episode_indices, data, t, k)
                    b = (torch.from_numpy(data.state_features[rows]).to(args.device).float() - mean) / std
                    d = (qq[:, None] - 2.0 * q @ b.T + (b * b).sum(1)[None]).min(1).values
                    pooled[k].append(d.cpu().numpy())
                    del b
                del q
        res["T"][h], res["stats"][h] = {}, {}
        for k in SHOTS:
            d = np.concatenate(pooled[k])
            res["T"][h][k] = float(np.percentile(d, Q))
            res["stats"][h][k] = {"n": int(d.size), **{f"p{p}": float(np.percentile(d, p)) for p in (5, 25, 50, 75, 90, 95, 99)}}
        print(f"[gate_T] fold {h}: T_k1={res['T'][h][1]:.1f} T_k16={res['T'][h][16]:.1f} "
              f"(n={res['stats'][h][1]['n']}, median k1 {res['stats'][h][1]['p50']:.1f}, k16 {res['stats'][h][16]['p50']:.1f})",
              flush=True)
    res["seconds"] = time.time() - t0
    args.out.write_text(json.dumps(res, indent=1))
    print(f"[gate_T] done ({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
